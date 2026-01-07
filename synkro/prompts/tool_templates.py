"""Prompt templates for tool call trace generation."""

# =============================================================================
# TOOL SCENARIO GENERATION
# =============================================================================

TOOL_SCENARIO_PROMPT = """You are an expert at creating realistic scenarios that require tool usage.

Given a set of available tools and usage guidelines, generate diverse scenarios that test when and how to use these tools correctly.

AVAILABLE TOOLS:
{TOOLS_DESCRIPTION}

USAGE GUIDELINES:
{GUIDELINES}

Generate scenarios that cover:

1. **Clear Tool Use Cases** - Situations where a specific tool is clearly needed
2. **Tool Selection** - Scenarios requiring choosing between multiple tools
3. **No Tool Needed** - Cases where the assistant should respond directly without tools
4. **Multi-Tool Workflows** - Complex tasks requiring multiple tool calls
5. **Parameter Variations** - Different parameter combinations and edge cases
6. **Error Handling** - What to do when tools return errors or unexpected results

Each scenario should include:
- A realistic user request
- Context about what information is available vs what needs to be looked up
- Expected tool usage pattern (or lack thereof)

Focus on creating "golden traces" - perfect examples of correct tool usage."""

TOOL_CATEGORY_SCENARIO_PROMPT = """You are an expert at creating realistic scenarios for tool usage.

Generate scenarios specifically for the following CATEGORY:
**Category Name**: {CATEGORY_NAME}
**Category Description**: {CATEGORY_DESCRIPTION}

AVAILABLE TOOLS:
{TOOLS_DESCRIPTION}

USAGE GUIDELINES:
{GUIDELINES}

Create scenarios that:
- Are deeply relevant to this specific category
- Test the nuances of tool usage in this context
- Include realistic user requests with appropriate context
- Cover both happy paths and edge cases within this category"""

# =============================================================================
# TOOL RESPONSE GENERATION
# =============================================================================

TOOL_RESPONSE_PROMPT = """You are generating a training example for teaching an AI assistant to use tools correctly.

AVAILABLE TOOLS:
{TOOLS_DESCRIPTION}

USAGE GUIDELINES:
{GUIDELINES}

SCENARIO:
{SCENARIO}

USER REQUEST:
{USER_REQUEST}

Generate a complete conversation that demonstrates correct tool usage:

1. If a tool should be called:
   - The assistant's first response should include appropriate tool_calls
   - Include the simulated tool response
   - The assistant should then synthesize the tool results into a helpful response

2. If no tool is needed:
   - The assistant should respond directly with helpful information
   - Explain why no tool lookup was necessary

The assistant should:
- Only call tools when necessary (don't call tools for information you already know)
- Use correct parameters with proper types
- Wait for tool results before providing final answers
- Synthesize tool results naturally without exposing raw data
- Handle missing or partial information gracefully

Output as JSON with this structure:
{{
  "messages": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": null, "tool_calls": [...]}},  // if tool needed
    {{"role": "tool", "tool_call_id": "...", "content": "..."}},    // tool result
    {{"role": "assistant", "content": "..."}}                        // final response
  ]
}}"""

# =============================================================================
# TOOL GRADING
# =============================================================================

TOOL_GRADE_PROMPT = """You are a strict evaluator of tool usage in AI assistant responses.

AVAILABLE TOOLS:
{TOOLS_DESCRIPTION}

USAGE GUIDELINES:
{GUIDELINES}

SCENARIO:
{SCENARIO}

CONVERSATION TO GRADE:
{CONVERSATION}

Evaluate the assistant's tool usage on these criteria:

1. **Tool Selection** (Did they use the right tool?)
   - Chose appropriate tool for the task
   - Didn't use tools when not needed
   - Used all necessary tools

2. **Parameter Accuracy** (Were the parameters correct?)
   - Correct parameter types
   - Sensible parameter values
   - Required parameters included

3. **Response Synthesis** (Did they use tool results correctly?)
   - Accurately incorporated tool results
   - Didn't hallucinate beyond tool data
   - Provided helpful, complete response

4. **Timing** (Did they call tools at the right time?)
   - Called tools before making claims
   - Didn't call tools for known information
   - Efficient tool call ordering

A response PASSES only if ALL criteria are met.

Grade this response."""

# =============================================================================
# TOOL REFINEMENT
# =============================================================================

TOOL_REFINE_PROMPT = """You are improving a tool-calling conversation that failed quality checks.

AVAILABLE TOOLS:
{TOOLS_DESCRIPTION}

USAGE GUIDELINES:
{GUIDELINES}

ORIGINAL SCENARIO:
{SCENARIO}

FAILED CONVERSATION:
{CONVERSATION}

ISSUES FOUND:
{ISSUES}

GRADER FEEDBACK:
{FEEDBACK}

Generate an IMPROVED conversation that fixes all the issues while maintaining the same user request.

Focus on:
- Correct tool selection
- Accurate parameters
- Proper synthesis of tool results
- No hallucination beyond tool data

Output the corrected conversation as JSON."""

# =============================================================================
# TOOL SIMULATION
# =============================================================================

TOOL_SIMULATION_PROMPT = """You are simulating a tool response for training data generation.

TOOL BEING CALLED:
Name: {TOOL_NAME}
Description: {TOOL_DESCRIPTION}
Parameters: {TOOL_PARAMETERS}

CALL ARGUMENTS:
{ARGUMENTS}

EXAMPLE RESPONSES (for reference):
{MOCK_RESPONSES}

Generate a realistic, plausible response that this tool would return for the given arguments.

The response should:
- Be realistic and internally consistent
- Match the type of data this tool would return
- Include appropriate detail level
- Handle edge cases gracefully (e.g., no results found)

Return only the tool response content as a string."""

# =============================================================================
# MULTI-TURN TOOL CALLING
# =============================================================================

MULTI_TURN_TOOL_DECISION_PROMPT = """You are a customer support agent deciding whether to use tools for a follow-up question.

AVAILABLE TOOLS:
{tools_desc}

TOOL USAGE GUIDELINES:
{policy_text}

CONVERSATION HISTORY (including previous tool calls and results):
{conversation_history}

NEW FOLLOW-UP QUESTION:
{follow_up_question}

Analyze this follow-up question and decide:
1. Can this be answered using information from previous tool results?
2. Does this require NEW tool calls?
3. If new tools are needed, which ones and with what arguments?

Important rules:
- If previous tool results contain the needed information, DON'T call tools again
- If the follow-up asks about something different, you MAY need new tools
- Use correct tool names and parameter types
- Provide clear reasoning for your decision"""

MULTI_TURN_TOOL_SYNTHESIS_PROMPT = """Based on the conversation and tool results, respond to the follow-up question.

CONVERSATION HISTORY:
{conversation_history}

LATEST FOLLOW-UP QUESTION:
{follow_up_question}

NEW TOOL RESULTS (if any):
{new_tool_results}

GUIDELINES:
{policy_text}

Synthesize a response that:
- Directly addresses the follow-up question
- Incorporates relevant information from ALL tool results (previous and new)
- Maintains consistency with previous responses
- Is conversational and helpful
- Does not expose raw JSON or technical details"""

# =============================================================================
# GOLDEN MULTI-TURN TOOL CALLING
# =============================================================================

GOLDEN_MULTI_TURN_TOOL_DECISION_PROMPT = """You are a customer support agent deciding whether to use tools for a follow-up.
Your decisions must be GROUNDED in the Logic Map rules.

AVAILABLE TOOLS:
{tools_desc}

LOGIC MAP (Rules to Apply):
{logic_map_str}

POLICY GUIDELINES:
{policy_text}

CONVERSATION HISTORY (including previous tool calls and results):
{conversation_history}

RULES ALREADY APPLIED: {cumulative_rules_applied}

NEW FOLLOW-UP QUESTION:
{follow_up_question}

YOUR TASK:
1. Identify which NEW rules from the Logic Map apply to this follow-up
2. Determine if any rule requires information that a tool can provide
3. Consider if previous tool results satisfy the rule requirements
4. If new tools needed, specify which rule requires each tool call

TOOL CALLING RULES:
- Only call a tool if a SPECIFIC RULE requires information not yet available
- Cite the Rule ID that necessitates each tool call
- If previous tool results satisfy the rule, don't call tools again
- Explain your reasoning in terms of rule evaluation"""

GOLDEN_MULTI_TURN_TOOL_SYNTHESIS_PROMPT = """Based on the conversation and tool results, respond to the follow-up question.
Your response must be GROUNDED in the Logic Map rules.

LOGIC MAP (Rules to Apply):
{logic_map_str}

CONVERSATION HISTORY:
{conversation_history}

LATEST FOLLOW-UP QUESTION:
{follow_up_question}

NEW TOOL RESULTS (if any):
{new_tool_results}

RULES ALREADY APPLIED: {cumulative_rules_applied}

POLICY GUIDELINES:
{policy_text}

Synthesize a response that:
- Directly addresses the follow-up question
- Cites the Rule IDs that apply to this response
- Incorporates relevant information from ALL tool results (previous and new)
- Maintains consistency with previous responses
- Does not expose raw JSON or technical details

Also identify:
- Which rules were applied in THIS turn's response
- Which rules were explicitly excluded and why"""