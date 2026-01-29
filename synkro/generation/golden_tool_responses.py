"""Golden Tool Response Generator - The Thinker for Tool Calls.

Generates tool call traces with grounded reasoning and rule citations.
This is Stage 3 of the Golden Trace pipeline for TOOL_CALL datasets.
"""

import asyncio
import json
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.tool_templates import (
    GOLDEN_MULTI_TURN_TOOL_DECISION_PROMPT,
    GOLDEN_MULTI_TURN_TOOL_SYNTHESIS_PROMPT,
)
from synkro.types.core import Message, Trace
from synkro.types.logic_map import GoldenScenario, LogicMap
from synkro.types.tool import ToolCall, ToolDefinition, ToolFunction

if TYPE_CHECKING:
    from synkro.generation.follow_ups import FollowUpGenerator
    from synkro.generation.tool_simulator import ToolSimulator


# =============================================================================
# Pydantic models for structured JSON output
# =============================================================================


class GoldenToolCallRequest(BaseModel):
    """A tool call request with rule citation."""

    name: str = Field(description="Name of the tool to call")
    arguments: str = Field(description="Arguments as JSON string")
    rule_id: str = Field(description="Rule ID that requires this tool call")
    reasoning: str = Field(description="Why this tool is needed for the rule")


class GoldenToolDecision(BaseModel):
    """Structured output for tool calling decision with rule grounding."""

    needs_tool: bool = Field(description="Whether a tool call is needed")
    reasoning: str = Field(description="Rule-based explanation of decision")
    rule_ids_evaluated: list[str] = Field(
        default_factory=list, description="Rule IDs that were evaluated"
    )
    tool_calls: list[GoldenToolCallRequest] = Field(
        default_factory=list, description="Tool calls with rule citations"
    )
    direct_response: str | None = Field(
        default=None, description="Direct response if no tool needed"
    )


class GoldenToolSynthesis(BaseModel):
    """Structured output for synthesizing tool results."""

    response: str = Field(description="Natural response incorporating tool results")
    rules_applied: list[str] = Field(
        default_factory=list, description="Rule IDs applied in the response"
    )
    rules_excluded: list[str] = Field(
        default_factory=list, description="Rule IDs explicitly excluded"
    )


class GoldenMultiTurnToolDecision(BaseModel):
    """Tool decision for a follow-up turn with rule grounding."""

    needs_tool: bool = Field(description="Whether a tool call is needed")
    reasoning: str = Field(description="Rule-based explanation of decision")
    rule_ids_evaluated: list[str] = Field(
        default_factory=list, description="Rule IDs evaluated for this turn"
    )
    tool_calls: list[GoldenToolCallRequest] = Field(
        default_factory=list, description="Tool calls with rule citations"
    )
    direct_response: str | None = Field(
        default=None, description="Direct response if no tool needed"
    )
    rules_applied_this_turn: list[str] = Field(
        default_factory=list, description="Rules applied in this turn's response"
    )
    rules_excluded_this_turn: list[str] = Field(
        default_factory=list, description="Rules excluded in this turn"
    )


class GoldenMultiTurnToolSynthesis(BaseModel):
    """Structured output for synthesizing follow-up responses with rule tracking."""

    response: str = Field(description="Natural response for follow-up")
    rules_applied_this_turn: list[str] = Field(
        default_factory=list, description="Rule IDs applied in this turn"
    )
    rules_excluded_this_turn: list[str] = Field(
        default_factory=list, description="Rule IDs excluded in this turn"
    )


# =============================================================================
# Golden Tool Call Response Generator
# =============================================================================


class GoldenToolCallResponseGenerator:
    """
    The Thinker for Tool Calls - Generates tool traces with grounded reasoning.

    Produces tool call traces with:
    - Rule citations for tool selection decisions
    - Explicit reasoning linking rules to tool usage
    - DAG-compliant evaluation order
    - Verification-ready metadata

    Examples:
        >>> generator = GoldenToolCallResponseGenerator(
        ...     tools=[web_search_tool],
        ...     llm=LLM(model=OpenAI.GPT_4O_MINI),
        ...     simulator=tool_simulator,
        ... )
        >>> trace = await generator.generate_single(
        ...     policy_text="...",
        ...     logic_map=logic_map,
        ...     scenario=scenario,
        ... )
    """

    # Instruction to inject when thinking mode is enabled
    THINKING_INSTRUCTION = """
THINKING MODE:
Your assistant response MUST include reasoning wrapped in <think> and </think> tags.
Place your step-by-step reasoning inside the think tags BEFORE your actual response.

Format:
<think>
[Your reasoning about which rules apply, tool usage decisions, etc.]
</think>

[Your actual response to the user]
"""

    def __init__(
        self,
        tools: list[ToolDefinition],
        llm: LLM | None = None,
        simulator: "ToolSimulator | None" = None,
        model: Model = OpenAI.GPT_4O_MINI,
        thinking: bool = False,
    ):
        """
        Initialize the Golden Tool Call Response Generator.

        Args:
            tools: List of available tool definitions
            llm: LLM client to use (creates one if not provided)
            simulator: Tool simulator for generating tool responses
            model: Model to use if creating LLM
            thinking: Enable thinking mode with <think> tags in responses
        """
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.llm = llm or LLM(model=model, temperature=0.7)
        self.simulator = simulator
        self.thinking = thinking
        self._follow_up_gen: "FollowUpGenerator | None" = None

    @property
    def follow_up_generator(self) -> "FollowUpGenerator":
        """Lazy initialization of follow-up generator for multi-turn."""
        if self._follow_up_gen is None:
            from synkro.generation.follow_ups import FollowUpGenerator

            self._follow_up_gen = FollowUpGenerator(llm=self.llm)
        return self._follow_up_gen

    def _get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(tool.to_system_prompt())
        return "\n\n".join(descriptions)

    def _generate_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for prompt inclusion."""
        lines = []
        lines.append("RULES:")
        for rule in logic_map.rules:
            deps = f" [depends on: {', '.join(rule.dependencies)}]" if rule.dependencies else ""
            lines.append(f"  {rule.rule_id} ({rule.category.value}): {rule.text}{deps}")
            lines.append(f"    IF: {rule.condition}")
            lines.append(f"    THEN: {rule.action}")
        return "\n".join(lines)

    async def generate_single(
        self,
        policy_text: str,
        logic_map: LogicMap,
        scenario: GoldenScenario,
        target_turns: int = 1,
    ) -> Trace:
        """
        Generate a single tool call trace with grounded reasoning.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map (DAG of rules)
            scenario: The golden scenario to respond to
            target_turns: Number of conversation turns (1 for single-turn,
                >1 for multi-turn with follow-up questions)

        Returns:
            Trace with proper tool calling format and rule citations
        """
        if target_turns > 1:
            return await self._generate_multi_turn(policy_text, logic_map, scenario, target_turns)

        # Single-turn generation
        tools_desc = self._get_tools_description()
        logic_map_str = self._format_logic_map(logic_map)

        # Step 1: Get LLM decision on tool usage with rule grounding
        decision = await self._get_tool_decision(policy_text, logic_map_str, scenario, tools_desc)

        # Step 2: Build the message sequence
        messages = await self._build_message_sequence(
            policy_text, logic_map_str, scenario, tools_desc, decision
        )

        # Convert GoldenScenario to base Scenario
        base_scenario = scenario.to_base_scenario()

        return Trace(messages=messages, scenario=base_scenario)

    async def _get_tool_decision(
        self,
        policy_text: str,
        logic_map_str: str,
        scenario: GoldenScenario,
        tools_desc: str,
    ) -> GoldenToolDecision:
        """Get the LLM's rule-grounded decision on tool usage."""
        prompt = f"""You are a customer support agent deciding whether to use tools.
Your decisions must be GROUNDED in the Logic Map rules.

AVAILABLE TOOLS:
{tools_desc}

LOGIC MAP (Rules to Apply):
{logic_map_str}

POLICY GUIDELINES:
{policy_text}

SCENARIO:
Type: {scenario.scenario_type.value.upper()}
Request: {scenario.description}
Context: {scenario.context}
Target Rules: {', '.join(scenario.target_rule_ids)}

YOUR TASK:
1. Evaluate which rules from the Logic Map apply to this scenario
2. Determine if any rule requires information that a tool can provide
3. If tools are needed, specify which rule requires each tool call
4. If no tools needed, explain based on which rules why direct response is sufficient

TOOL CALLING RULES:
- Only call a tool if a SPECIFIC RULE requires information the tool can provide
- Cite the Rule ID that necessitates each tool call
- If the scenario is IRRELEVANT type, no tools should be needed
- If information is already in the context, don't call a tool for it"""

        return await self.llm.generate_structured(prompt, GoldenToolDecision)

    async def _build_message_sequence(
        self,
        policy_text: str,
        logic_map_str: str,
        scenario: GoldenScenario,
        tools_desc: str,
        decision: GoldenToolDecision,
    ) -> list[Message]:
        """Build the full message sequence based on the tool decision."""
        messages = []

        # System message with tool descriptions
        system_content = f"""You are a helpful customer support agent. You have access to the following tools:

{tools_desc}

Follow the policy guidelines to assist customers effectively."""

        messages.append(Message(role="system", content=system_content))

        # User message
        messages.append(Message(role="user", content=scenario.description))

        if decision.needs_tool and decision.tool_calls:
            # Assistant message with tool_calls
            tool_calls = []
            for tc in decision.tool_calls:
                call_id = self._generate_call_id()
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=ToolFunction(name=tc.name, arguments=tc.arguments),
                    )
                )

            messages.append(Message(role="assistant", content=None, tool_calls=tool_calls))

            # Tool response messages
            tool_results = []
            for tc in tool_calls:
                result = await self._simulate_tool_call(tc)
                tool_results.append(result)

                messages.append(Message(role="tool", content=result, tool_call_id=tc.id))

            # Final assistant message synthesizing results
            final_response = await self._synthesize_response(
                scenario, tool_calls, tool_results, decision, policy_text, logic_map_str
            )
            messages.append(Message(role="assistant", content=final_response))

        else:
            # Direct response without tools
            response = decision.direct_response or await self._generate_direct_response(
                policy_text, logic_map_str, scenario
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
                import random

                return random.choice(tool.mock_responses)

        # Default mock response
        args = json.loads(tool_call.function.arguments)
        return json.dumps(
            {"status": "success", "result": f"Simulated response for {tool_name}", "query": args}
        )

    async def _synthesize_response(
        self,
        scenario: GoldenScenario,
        tool_calls: list[ToolCall],
        tool_results: list[str],
        decision: GoldenToolDecision,
        policy_text: str,
        logic_map_str: str,
    ) -> str:
        """Synthesize a natural response from tool results with rule grounding."""
        # Build context of tool calls and results
        tools_context = []
        for tc, result in zip(tool_calls, tool_results):
            tools_context.append(f"Tool: {tc.function.name}")
            tools_context.append(f"Arguments: {tc.function.arguments}")
            tools_context.append(f"Result: {result}")
            tools_context.append("")

        prompt = f"""Based on the tool results and rules, provide a helpful response.

USER REQUEST:
{scenario.description}

SCENARIO TYPE: {scenario.scenario_type.value.upper()}
TARGET RULES: {', '.join(scenario.target_rule_ids)}

TOOL RESULTS:
{chr(10).join(tools_context)}

LOGIC MAP:
{logic_map_str}

RULES EVALUATED: {', '.join(decision.rule_ids_evaluated)}

Synthesize the tool results into a natural, helpful response.
- Apply the relevant rules from the Logic Map
- Incorporate the information from the tool results
- Don't expose raw JSON or technical details
- Be conversational and helpful"""

        # Inject thinking instruction if enabled
        if self.thinking:
            prompt = prompt + self.THINKING_INSTRUCTION

        synthesis = await self.llm.generate_structured(prompt, GoldenToolSynthesis)
        return synthesis.response

    async def _generate_direct_response(
        self,
        policy_text: str,
        logic_map_str: str,
        scenario: GoldenScenario,
    ) -> str:
        """Generate a direct response when no tools are needed."""
        prompt = f"""Provide a helpful response based on the rules.

USER REQUEST:
{scenario.description}

CONTEXT:
{scenario.context}

SCENARIO TYPE: {scenario.scenario_type.value.upper()}
TARGET RULES: {', '.join(scenario.target_rule_ids)}

LOGIC MAP:
{logic_map_str}

POLICY GUIDELINES:
{policy_text}

No tools are needed for this request. Provide a direct, helpful response
applying the relevant rules from the Logic Map."""

        # Inject thinking instruction if enabled
        if self.thinking:
            prompt = prompt + self.THINKING_INSTRUCTION

        synthesis = await self.llm.generate_structured(prompt, GoldenToolSynthesis)
        return synthesis.response

    # =========================================================================
    # MULTI-TURN TOOL CALLING WITH RULE TRACKING
    # =========================================================================

    async def _generate_multi_turn(
        self,
        policy_text: str,
        logic_map: LogicMap,
        scenario: GoldenScenario,
        target_turns: int,
    ) -> Trace:
        """
        Generate multi-turn golden tool call trace with cumulative rule tracking.

        Each turn can independently decide if new tool calls are needed.
        Rules applied/excluded are tracked across all turns.

        Args:
            policy_text: The policy/guidelines text
            logic_map: The extracted Logic Map
            scenario: The golden scenario to respond to
            target_turns: Number of conversation turns

        Returns:
            Trace with multi-turn tool calling and cumulative rule metadata
        """
        tools_desc = self._get_tools_description()
        logic_map_str = self._format_logic_map(logic_map)

        # Track cumulative rules across turns
        cumulative_rules_applied: list[str] = []
        cumulative_rules_excluded: list[str] = []

        # Step 1: Generate initial response (Turn 1)
        decision = await self._get_tool_decision(policy_text, logic_map_str, scenario, tools_desc)
        messages = await self._build_message_sequence(
            policy_text, logic_map_str, scenario, tools_desc, decision
        )

        # Track rules from initial turn
        cumulative_rules_applied.extend(decision.rule_ids_evaluated)

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

            # Get rule-grounded tool decision for this follow-up
            follow_up_decision = await self._get_follow_up_tool_decision(
                policy_text=policy_text,
                logic_map_str=logic_map_str,
                messages=messages,
                follow_up_question=follow_up.question,
                tools_desc=tools_desc,
                cumulative_rules_applied=cumulative_rules_applied,
            )

            # Build response for this turn
            (
                turn_messages,
                turn_rules_applied,
                turn_rules_excluded,
            ) = await self._build_follow_up_message_sequence(
                policy_text=policy_text,
                logic_map_str=logic_map_str,
                messages=messages,
                follow_up_question=follow_up.question,
                tools_desc=tools_desc,
                decision=follow_up_decision,
                cumulative_rules_applied=cumulative_rules_applied,
            )

            messages.extend(turn_messages)

            # Update cumulative rule tracking
            cumulative_rules_applied.extend(turn_rules_applied)
            cumulative_rules_excluded.extend(turn_rules_excluded)

        # Deduplicate rules
        unique_rules_applied = list(dict.fromkeys(cumulative_rules_applied))
        unique_rules_excluded = list(dict.fromkeys(cumulative_rules_excluded))

        base_scenario = scenario.to_base_scenario()

        return Trace(
            messages=messages,
            scenario=base_scenario,
            rules_applied=unique_rules_applied,
            rules_excluded=unique_rules_excluded,
        )

    def _format_conversation_with_tools(self, messages: list[Message]) -> str:
        """Format conversation including tool calls and results for context."""
        formatted = []
        for msg in messages:
            role = msg.role.upper()

            if msg.role == "assistant" and msg.tool_calls:
                tool_strs = []
                for tc in msg.tool_calls:
                    if hasattr(tc, "function"):
                        tool_strs.append(f"  - {tc.function.name}({tc.function.arguments})")
                    elif isinstance(tc, dict) and "function" in tc:
                        func = tc["function"]
                        tool_strs.append(
                            f"  - {func.get('name', 'unknown')}({func.get('arguments', '{}')})"
                        )
                    else:
                        tool_strs.append(f"  - {tc}")
                formatted.append("ASSISTANT: [Tool Calls]\n" + "\n".join(tool_strs))
            elif msg.role == "tool":
                formatted.append(f"TOOL RESULT [{msg.tool_call_id}]: {msg.content}")
            else:
                content = msg.content or "[No content]"
                formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    async def _get_follow_up_tool_decision(
        self,
        policy_text: str,
        logic_map_str: str,
        messages: list[Message],
        follow_up_question: str,
        tools_desc: str,
        cumulative_rules_applied: list[str],
    ) -> GoldenMultiTurnToolDecision:
        """Get rule-grounded tool decision for a follow-up question."""
        conversation_history = self._format_conversation_with_tools(messages)

        prompt = GOLDEN_MULTI_TURN_TOOL_DECISION_PROMPT.format(
            tools_desc=tools_desc,
            logic_map_str=logic_map_str,
            policy_text=policy_text,
            conversation_history=conversation_history,
            cumulative_rules_applied=", ".join(cumulative_rules_applied) or "None yet",
            follow_up_question=follow_up_question,
        )

        return await self.llm.generate_structured(prompt, GoldenMultiTurnToolDecision)

    async def _build_follow_up_message_sequence(
        self,
        policy_text: str,
        logic_map_str: str,
        messages: list[Message],
        follow_up_question: str,
        tools_desc: str,
        decision: GoldenMultiTurnToolDecision,
        cumulative_rules_applied: list[str],
    ) -> tuple[list[Message], list[str], list[str]]:
        """
        Build message sequence for a follow-up turn with rule tracking.

        Returns:
            Tuple of (new_messages, rules_applied_this_turn, rules_excluded_this_turn)
        """
        new_messages = []
        rules_applied: list[str] = []
        rules_excluded: list[str] = []

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

            new_messages.append(Message(role="assistant", content=None, tool_calls=tool_calls))

            # Tool response messages
            tool_results = []
            for tc in tool_calls:
                result = await self._simulate_tool_call(tc)
                tool_results.append(result)
                new_messages.append(Message(role="tool", content=result, tool_call_id=tc.id))

            # Final assistant message with rule-grounded synthesis
            response, rules_applied, rules_excluded = await self._synthesize_follow_up_response(
                policy_text=policy_text,
                logic_map_str=logic_map_str,
                messages=messages,
                follow_up_question=follow_up_question,
                tool_calls=tool_calls,
                tool_results=tool_results,
                cumulative_rules_applied=cumulative_rules_applied,
            )
            new_messages.append(Message(role="assistant", content=response))

        else:
            # Direct response without new tools
            if decision.direct_response:
                response = decision.direct_response
                rules_applied = decision.rules_applied_this_turn
                rules_excluded = decision.rules_excluded_this_turn
            else:
                response, rules_applied, rules_excluded = await self._synthesize_follow_up_response(
                    policy_text=policy_text,
                    logic_map_str=logic_map_str,
                    messages=messages,
                    follow_up_question=follow_up_question,
                    tool_calls=[],
                    tool_results=[],
                    cumulative_rules_applied=cumulative_rules_applied,
                )
            new_messages.append(Message(role="assistant", content=response))

        return new_messages, rules_applied, rules_excluded

    async def _synthesize_follow_up_response(
        self,
        policy_text: str,
        logic_map_str: str,
        messages: list[Message],
        follow_up_question: str,
        tool_calls: list[ToolCall],
        tool_results: list[str],
        cumulative_rules_applied: list[str],
    ) -> tuple[str, list[str], list[str]]:
        """
        Synthesize response for a follow-up turn with rule tracking.

        Returns:
            Tuple of (response, rules_applied_this_turn, rules_excluded_this_turn)
        """
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

        prompt = GOLDEN_MULTI_TURN_TOOL_SYNTHESIS_PROMPT.format(
            logic_map_str=logic_map_str,
            conversation_history=conversation_history,
            follow_up_question=follow_up_question,
            new_tool_results=new_results_str,
            cumulative_rules_applied=", ".join(cumulative_rules_applied) or "None yet",
            policy_text=policy_text,
        )

        # Inject thinking instruction if enabled
        if self.thinking:
            prompt = prompt + self.THINKING_INSTRUCTION

        synthesis = await self.llm.generate_structured(prompt, GoldenMultiTurnToolSynthesis)
        return (
            synthesis.response,
            synthesis.rules_applied_this_turn,
            synthesis.rules_excluded_this_turn,
        )

    async def generate(
        self,
        policy_text: str,
        logic_map: LogicMap,
        scenarios: list[GoldenScenario],
        target_turns: int = 1,
        concurrency: int = 50,
    ) -> list[Trace]:
        """
        Generate traces for multiple scenarios.

        Args:
            policy_text: The policy document text
            logic_map: The extracted Logic Map
            scenarios: List of golden scenarios
            target_turns: Number of conversation turns
            concurrency: Maximum concurrent LLM calls (default: 50)

        Returns:
            List of traces with tool calling format
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_limit(scenario):
            async with semaphore:
                return await self.generate_single(policy_text, logic_map, scenario, target_turns)

        tasks = [generate_with_limit(s) for s in scenarios]
        return await asyncio.gather(*tasks)


__all__ = [
    "GoldenToolCallResponseGenerator",
    "GoldenToolDecision",
    "GoldenToolCallRequest",
    "GoldenToolSynthesis",
    "GoldenMultiTurnToolDecision",
    "GoldenMultiTurnToolSynthesis",
]
