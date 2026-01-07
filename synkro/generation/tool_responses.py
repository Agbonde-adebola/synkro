"""Tool call response generation with JSON mode for structured outputs."""

import json
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.core import Scenario, Trace, Message
from synkro.types.tool import ToolCall, ToolFunction, ToolDefinition
from synkro.prompts.tool_templates import (
    MULTI_TURN_TOOL_DECISION_PROMPT,
    MULTI_TURN_TOOL_SYNTHESIS_PROMPT,
)

if TYPE_CHECKING:
    from synkro.generation.tool_simulator import ToolSimulator
    from synkro.generation.follow_ups import FollowUpGenerator


# =============================================================================
# Pydantic models for structured JSON output
# =============================================================================

class ToolCallRequest(BaseModel):
    """A single tool call request from the LLM."""
    
    name: str = Field(description="Name of the tool to call")
    arguments: str = Field(description="Arguments as a JSON string, e.g. '{\"query\": \"test\"}'")
    
    def get_arguments_dict(self) -> dict:
        """Parse arguments JSON string to dict."""
        return json.loads(self.arguments)


class ToolCallDecision(BaseModel):
    """
    Structured output for the LLM's tool calling decision.
    
    The LLM outputs this to indicate whether tools are needed
    and which ones to call.
    """
    
    needs_tool: bool = Field(
        description="Whether a tool call is needed to answer the user's request"
    )
    reasoning: str = Field(
        description="Brief explanation of why tool is/isn't needed"
    )
    tool_calls: list[ToolCallRequest] = Field(
        default_factory=list,
        description="List of tool calls to make (empty if needs_tool is False)"
    )
    direct_response: str | None = Field(
        default=None,
        description="Direct response if no tool is needed"
    )


class FinalSynthesis(BaseModel):
    """Structured output for synthesizing tool results into a response."""
    
    response: str = Field(
        description="Natural response incorporating the tool results"
    )


# =============================================================================
# Tool Call Response Generator
# =============================================================================

class ToolCallResponseGenerator:
    """
    Generates tool call training traces using JSON mode for structured outputs.
    
    Produces traces in OpenAI function calling format:
    - system message with tool descriptions
    - user message with request
    - assistant message with tool_calls (or direct response)
    - tool response messages
    - final assistant message synthesizing results
    
    Example:
        >>> gen = ToolCallResponseGenerator(
        ...     tools=[web_search_tool, db_tool],
        ...     llm=LLM(model=OpenAI.GPT_4O),
        ...     simulator=tool_simulator,
        ... )
        >>> trace = await gen.generate_single(policy_text, scenario)
    """
    
    def __init__(
        self,
        tools: list[ToolDefinition],
        llm: LLM | None = None,
        simulator: "ToolSimulator | None" = None,
        model: Model = OpenAI.GPT_4O_MINI,
    ):
        """
        Initialize the tool call response generator.
        
        Args:
            tools: List of available tool definitions
            llm: LLM client to use (creates one if not provided)
            simulator: Tool simulator for generating tool responses
            model: Model to use if creating LLM
        """
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.llm = llm or LLM(model=model)
        self.simulator = simulator
        self._follow_up_gen: "FollowUpGenerator | None" = None

    @property
    def follow_up_generator(self) -> "FollowUpGenerator":
        """Lazy initialization of follow-up generator for multi-turn."""
        if self._follow_up_gen is None:
            from synkro.generation.follow_ups import FollowUpGenerator
            self._follow_up_gen = FollowUpGenerator(llm=self.llm)
        return self._follow_up_gen
    
    def _get_tools_description(self) -> str:
        """Get formatted description of all tools for system prompt."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.to_system_prompt())
        return "\n\n".join(descriptions)
    
    def _get_tools_json_schema(self) -> str:
        """Get JSON schema representation of tools."""
        tools_json = [tool.to_openai_format() for tool in self.tools]
        return json.dumps(tools_json, indent=2)
    
    def _generate_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"
    
    async def generate_single(
        self,
        policy_text: str,
        scenario: Scenario,
        target_turns: int = 1,
    ) -> Trace:
        """
        Generate a single tool call trace.

        Args:
            policy_text: The policy/guidelines text
            scenario: The scenario to respond to
            target_turns: Number of conversation turns (1 for single-turn,
                >1 for multi-turn with follow-up questions)

        Returns:
            Trace with proper tool calling format
        """
        if target_turns > 1:
            return await self._generate_multi_turn(policy_text, scenario, target_turns)

        # Single-turn generation
        tools_desc = self._get_tools_description()

        # Step 1: Get LLM decision on tool usage
        decision = await self._get_tool_decision(policy_text, scenario, tools_desc)

        # Step 2: Build the message sequence
        messages = await self._build_message_sequence(
            policy_text, scenario, tools_desc, decision
        )

        return Trace(messages=messages, scenario=scenario)
    
    async def _get_tool_decision(
        self,
        policy_text: str,
        scenario: Scenario,
        tools_desc: str,
    ) -> ToolCallDecision:
        """
        Get the LLM's decision on whether to use tools.
        
        Uses JSON mode to force structured output.
        """
        prompt = f"""You are a customer support agent deciding whether to use tools.

AVAILABLE TOOLS:
{tools_desc}

TOOL USAGE GUIDELINES:
{policy_text}

USER REQUEST:
{scenario.description}

CONTEXT:
{scenario.context}

Analyze this request and decide:
1. Does this require calling a tool, or can you answer directly?
2. If tools are needed, which ones and with what arguments?
3. If no tools needed, provide the direct response.

Important rules:
- Only call tools when necessary (don't call for information you already know)
- Use correct tool names and parameter types
- If multiple tools are needed, list them all
- Provide clear reasoning for your decision"""

        return await self.llm.generate_structured(prompt, ToolCallDecision)
    
    async def _build_message_sequence(
        self,
        policy_text: str,
        scenario: Scenario,
        tools_desc: str,
        decision: ToolCallDecision,
    ) -> list[Message]:
        """Build the full message sequence based on the tool decision."""
        messages = []
        
        # System message with tool descriptions
        system_content = f"""You are a helpful customer support agent. You have access to the following tools:

{tools_desc}

Follow the tool usage guidelines provided to assist customers effectively."""
        
        messages.append(Message(role="system", content=system_content))
        
        # User message
        messages.append(Message(role="user", content=scenario.description))
        
        if decision.needs_tool and decision.tool_calls:
            # Assistant message with tool_calls
            tool_calls = []
            for tc in decision.tool_calls:
                call_id = self._generate_call_id()
                tool_calls.append(ToolCall(
                    id=call_id,
                    type="function",
                    function=ToolFunction(
                        name=tc.name,
                        arguments=tc.arguments  # Already a JSON string
                    )
                ))
            
            messages.append(Message(
                role="assistant",
                content=None,
                tool_calls=tool_calls
            ))
            
            # Tool response messages
            tool_results = []
            for tc in tool_calls:
                result = await self._simulate_tool_call(tc)
                tool_results.append(result)
                
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id
                ))
            
            # Final assistant message synthesizing results
            final_response = await self._synthesize_response(
                scenario.description, tool_calls, tool_results, policy_text
            )
            messages.append(Message(role="assistant", content=final_response))
        
        else:
            # Direct response without tools
            response = decision.direct_response or await self._generate_direct_response(
                policy_text, scenario, tools_desc
            )
            messages.append(Message(role="assistant", content=response))
        
        return messages
    
    async def _simulate_tool_call(self, tool_call: ToolCall) -> str:
        """Simulate a tool response."""
        if self.simulator:
            return await self.simulator.simulate(tool_call)
        
        # Fallback: generate a mock response based on tool definition
        tool_name = tool_call.function.name
        if tool_name in self.tools_by_name:
            tool = self.tools_by_name[tool_name]
            if tool.mock_responses:
                # Use a mock response
                import random
                return random.choice(tool.mock_responses)
        
        # Default mock response
        args = json.loads(tool_call.function.arguments)
        return json.dumps({
            "status": "success",
            "result": f"Simulated response for {tool_name}",
            "query": args
        })
    
    async def _synthesize_response(
        self,
        user_request: str,
        tool_calls: list[ToolCall],
        tool_results: list[str],
        policy_text: str,
    ) -> str:
        """Synthesize a natural response from tool results."""
        # Build context of tool calls and results
        tools_context = []
        for tc, result in zip(tool_calls, tool_results):
            tools_context.append(f"Tool: {tc.function.name}")
            tools_context.append(f"Arguments: {tc.function.arguments}")
            tools_context.append(f"Result: {result}")
            tools_context.append("")
        
        prompt = f"""Based on the tool results, provide a helpful response to the user.

USER REQUEST:
{user_request}

TOOL RESULTS:
{chr(10).join(tools_context)}

GUIDELINES:
{policy_text}

Synthesize the tool results into a natural, helpful response. 
- Incorporate the information from the tool results
- Don't expose raw JSON or technical details
- Be conversational and helpful
- If a tool returned an error, acknowledge it and offer alternatives"""

        synthesis = await self.llm.generate_structured(prompt, FinalSynthesis)
        return synthesis.response
    
    async def _generate_direct_response(
        self,
        policy_text: str,
        scenario: Scenario,
        tools_desc: str,
    ) -> str:
        """Generate a direct response when no tools are needed."""
        prompt = f"""Provide a helpful response to the user's request.

USER REQUEST:
{scenario.description}

CONTEXT:
{scenario.context}

GUIDELINES:
{policy_text}

Note: No tools are needed for this request. Provide a direct, helpful response
based on your knowledge and the guidelines."""

        synthesis = await self.llm.generate_structured(prompt, FinalSynthesis)
        return synthesis.response

    # =========================================================================
    # MULTI-TURN TOOL CALLING
    # =========================================================================

    async def _generate_multi_turn(
        self,
        policy_text: str,
        scenario: Scenario,
        target_turns: int,
    ) -> Trace:
        """
        Generate multi-turn tool call trace.

        Each turn can independently decide if new tool calls are needed
        based on the follow-up question and conversation history.

        Args:
            policy_text: The policy/guidelines text
            scenario: The initial scenario to respond to
            target_turns: Number of conversation turns

        Returns:
            Trace with multi-turn tool calling conversation
        """
        tools_desc = self._get_tools_description()

        # Step 1: Generate initial response (Turn 1)
        decision = await self._get_tool_decision(policy_text, scenario, tools_desc)
        messages = await self._build_message_sequence(
            policy_text, scenario, tools_desc, decision
        )

        # Step 2: Generate follow-up turns
        for turn in range(1, target_turns):
            # Generate follow-up question based on conversation so far
            follow_up = await self.follow_up_generator.generate(
                policy_text=policy_text,
                messages=messages,
                turn_index=turn,
            )

            # Add user message with follow-up question
            messages.append(Message(role="user", content=follow_up.question))

            # Get tool decision for this follow-up
            follow_up_decision = await self._get_follow_up_tool_decision(
                policy_text=policy_text,
                messages=messages,
                follow_up_question=follow_up.question,
                tools_desc=tools_desc,
            )

            # Build response for this turn (may include new tool calls)
            turn_messages = await self._build_follow_up_message_sequence(
                policy_text=policy_text,
                messages=messages,
                follow_up_question=follow_up.question,
                tools_desc=tools_desc,
                decision=follow_up_decision,
            )

            # Extend conversation with this turn's messages
            messages.extend(turn_messages)

        return Trace(messages=messages, scenario=scenario)

    def _format_conversation_with_tools(self, messages: list[Message]) -> str:
        """
        Format conversation including tool calls and results.

        This provides context for follow-up tool decisions so the LLM knows:
        - What tools were already called
        - What results were obtained
        - What information is already available
        """
        formatted = []
        for msg in messages:
            role = msg.role.upper()

            if msg.role == "assistant" and msg.tool_calls:
                # Format assistant message with tool calls
                tool_strs = []
                for tc in msg.tool_calls:
                    if hasattr(tc, "function"):
                        tool_strs.append(
                            f"  - {tc.function.name}({tc.function.arguments})"
                        )
                    elif isinstance(tc, dict) and "function" in tc:
                        func = tc["function"]
                        tool_strs.append(
                            f"  - {func.get('name', 'unknown')}({func.get('arguments', '{}')})"
                        )
                    else:
                        tool_strs.append(f"  - {tc}")
                formatted.append(f"ASSISTANT: [Tool Calls]\n" + "\n".join(tool_strs))
            elif msg.role == "tool":
                # Format tool response
                formatted.append(f"TOOL RESULT [{msg.tool_call_id}]: {msg.content}")
            else:
                content = msg.content or "[No content]"
                formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    async def _get_follow_up_tool_decision(
        self,
        policy_text: str,
        messages: list[Message],
        follow_up_question: str,
        tools_desc: str,
    ) -> ToolCallDecision:
        """
        Get tool decision for a follow-up question with full conversation context.

        The LLM can see previous tool calls and results to decide if new
        tools are needed or if existing results can answer the follow-up.
        """
        conversation_history = self._format_conversation_with_tools(messages)

        prompt = MULTI_TURN_TOOL_DECISION_PROMPT.format(
            tools_desc=tools_desc,
            policy_text=policy_text,
            conversation_history=conversation_history,
            follow_up_question=follow_up_question,
        )

        return await self.llm.generate_structured(prompt, ToolCallDecision)

    async def _build_follow_up_message_sequence(
        self,
        policy_text: str,
        messages: list[Message],
        follow_up_question: str,
        tools_desc: str,
        decision: ToolCallDecision,
    ) -> list[Message]:
        """
        Build message sequence for a follow-up turn.

        Returns only the new messages for this turn (not the full conversation).
        May include: assistant with tool_calls, tool responses, final assistant.
        Or just: assistant with direct response.
        """
        new_messages = []

        if decision.needs_tool and decision.tool_calls:
            # Assistant message with new tool_calls
            tool_calls = []
            for tc in decision.tool_calls:
                call_id = self._generate_call_id()
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=ToolFunction(
                            name=tc.name,
                            arguments=tc.arguments,
                        ),
                    )
                )

            new_messages.append(
                Message(role="assistant", content=None, tool_calls=tool_calls)
            )

            # Tool response messages
            tool_results = []
            for tc in tool_calls:
                result = await self._simulate_tool_call(tc)
                tool_results.append(result)
                new_messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )

            # Final assistant message synthesizing new results
            final_response = await self._synthesize_follow_up_response(
                policy_text=policy_text,
                messages=messages,
                follow_up_question=follow_up_question,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            new_messages.append(Message(role="assistant", content=final_response))

        else:
            # Direct response without new tools
            if decision.direct_response:
                response = decision.direct_response
            else:
                # Generate response using existing context
                response = await self._synthesize_follow_up_response(
                    policy_text=policy_text,
                    messages=messages,
                    follow_up_question=follow_up_question,
                    tool_calls=[],
                    tool_results=[],
                )
            new_messages.append(Message(role="assistant", content=response))

        return new_messages

    async def _synthesize_follow_up_response(
        self,
        policy_text: str,
        messages: list[Message],
        follow_up_question: str,
        tool_calls: list[ToolCall],
        tool_results: list[str],
    ) -> str:
        """Synthesize response for a follow-up turn."""
        conversation_history = self._format_conversation_with_tools(messages)

        # Format new tool results if any
        if tool_calls and tool_results:
            new_tool_results = []
            for tc, result in zip(tool_calls, tool_results):
                new_tool_results.append(f"Tool: {tc.function.name}")
                new_tool_results.append(f"Arguments: {tc.function.arguments}")
                new_tool_results.append(f"Result: {result}")
                new_tool_results.append("")
            new_results_str = "\n".join(new_tool_results)
        else:
            new_results_str = "None (using existing information from conversation)"

        prompt = MULTI_TURN_TOOL_SYNTHESIS_PROMPT.format(
            conversation_history=conversation_history,
            follow_up_question=follow_up_question,
            new_tool_results=new_results_str,
            policy_text=policy_text,
        )

        synthesis = await self.llm.generate_structured(prompt, FinalSynthesis)
        return synthesis.response

    async def generate(
        self,
        policy_text: str,
        scenarios: list[Scenario],
    ) -> list[Trace]:
        """
        Generate traces for multiple scenarios.
        
        Args:
            policy_text: The policy/guidelines text
            scenarios: List of scenarios to respond to
            
        Returns:
            List of traces with tool calling format
        """
        traces = []
        for scenario in scenarios:
            trace = await self.generate_single(policy_text, scenario)
            traces.append(trace)
        return traces


__all__ = [
    "ToolCallResponseGenerator",
    "ToolCallDecision",
    "ToolCallRequest",
    "FinalSynthesis",
]

