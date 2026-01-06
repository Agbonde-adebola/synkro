"""Prompt templates for Golden Trace generation.

These prompts implement the 4-stage Golden Trace pipeline:
1. Logic Extraction (The Cartographer) - Extract rules as DAG
2. Scenario Synthesis (The Adversary) - Generate typed scenarios
3. Trace Synthesis (The Thinker) - Produce grounded reasoning
4. Verification (The Auditor) - Verify trace against Logic Map
"""

# =============================================================================
# STAGE 1: LOGIC EXTRACTION (The Cartographer)
# =============================================================================

LOGIC_EXTRACTION_PROMPT = """You are a policy analyst tasked with extracting a Logic Map from a policy document.

A Logic Map is a Directed Acyclic Graph (DAG) where:
- Each node is a RULE with a unique ID (R001, R002, etc.)
- Edges represent DEPENDENCIES between rules
- Root rules have no dependencies (they are entry points)

POLICY DOCUMENT:
{policy_text}

EXTRACTION INSTRUCTIONS:

1. **Identify All Rules**: Extract every distinct rule, condition, or requirement from the policy.
   - Look for: "must", "shall", "should", "can", "cannot", "if...then", "unless", "except"
   - Each rule should be atomic (one condition -> one action)

2. **Categorize Each Rule**:
   - CONSTRAINT: Must/must not conditions (e.g., "Refunds must be requested within 30 days")
   - PERMISSION: Allowed actions (e.g., "Customers can request store credit")
   - PROCEDURE: Step-by-step processes (e.g., "To cancel, first verify identity, then...")
   - EXCEPTION: Special cases that override other rules (e.g., "VIP customers are exempt from...")

3. **Identify Dependencies**:
   - If Rule B can only be evaluated after Rule A is known, then B depends on A
   - Example: "If refund is approved (R001), customer can choose cash or credit (R002)" - R002 depends on R001
   - Root rules are those that can be evaluated independently

4. **Ensure DAG Properties**:
   - No circular dependencies (A -> B -> A is invalid)
   - All rules must be reachable from root rules

5. **CRITICAL - Rule Precision Requirements**:

   a) **Explicit Scope**: Each rule must clearly state WHO or WHAT it applies to.
      - BAD: "Maximum $75 per person" (ambiguous - applies to what?)
      - GOOD: "Team events have a maximum of $75 per person. Client meals have no per-person limit."

   b) **Boundary Clarity**: For thresholds, specify inclusive vs exclusive.
      - BAD: "Expenses over $50 need approval" (is $50 exactly included?)
      - GOOD: "Expenses of $50 or more require manager approval" (inclusive)
      - GOOD: "Expenses exceeding $50 require manager approval" (exclusive, $50 does not need approval)

   c) **Distinguish Similar Rules**: If a policy treats categories differently, create SEPARATE rules.
      - Example: If "client meals" and "team events" have different limits, they need separate rule IDs
      - R008a: "Client meals: no per-person spending limit"
      - R008b: "Team events: maximum $75 per person"

   d) **No Ambiguous Groupings**: Avoid rules that bundle unrelated constraints.
      - BAD: "Meals have various limits depending on type"
      - GOOD: Separate rules for each meal type with specific limits

OUTPUT FORMAT:
Provide the Logic Map with:
- rules: List of all extracted rules with their IDs, text, conditions, actions, dependencies, and categories
- root_rules: List of rule IDs that have no dependencies (entry points)
- reasoning: Brief explanation of the extraction process and key relationships identified"""


# =============================================================================
# STAGE 2: SCENARIO SYNTHESIS (The Adversary)
# =============================================================================

GOLDEN_SCENARIO_PROMPT = """You are a scenario generator creating {scenario_type} test cases for a policy.

POLICY DOCUMENT:
{policy_text}

LOGIC MAP (Extracted Rules):
{logic_map}

CATEGORY: {category}
COUNT: Generate exactly {count} scenarios

SCENARIO TYPES:
- POSITIVE (Happy Path): User meets ALL criteria, rules should approve/allow
- NEGATIVE (Violation): User fails EXACTLY ONE criterion, rules should deny/reject
- EDGE_CASE (Boundary): User is at exact limits (e.g., day 30 of 30-day window)
- IRRELEVANT: Query not covered by the policy at all

YOUR TASK - Generate {scenario_type} scenarios:

{type_specific_instructions}

REQUIREMENTS FOR EACH SCENARIO:
1. description: The user's EXACT words - a realistic request/question
   - This is LITERALLY what the user says, nothing more
   - Should be natural and conversational
   - Example: "I'd like to submit an expense for a client lunch"

2. context: Background facts for evaluation that the user has NOT stated
   - Include specific details: amounts, dates, receipt status, approval status
   - These details inform the assistant's reasoning but are NOT in the user's message
   - Example: "Expense amount: $180, Purchase date: 5 days ago, Has digital receipt, No manager approval yet"

3. target_rule_ids: Which rules from the Logic Map this scenario tests
4. expected_outcome: What the correct response should do based on the rules

CRITICAL - DESCRIPTION VS CONTEXT SEPARATION:
- The description should NOT contain specific amounts, dates, or status details
- Those details belong in context ONLY
- The assistant will need to either:
  a) Ask the user for these details, OR
  b) Use them for reasoning if the scenario implies they're known

BAD EXAMPLE:
  description: "I want to submit a $180 expense from last week with receipt"  ← Too specific!
  context: "Has manager approval"

GOOD EXAMPLE:
  description: "I'd like to submit an expense for a client lunch"
  context: "Expense amount: $180, Purchase date: 5 days ago, Has digital receipt, Has manager approval"

IMPORTANT:
- Each scenario must reference specific rule IDs from the Logic Map
- Scenarios should be diverse within the category
- {scenario_type} scenarios should clearly demonstrate the expected behavior"""

POSITIVE_SCENARIO_INSTRUCTIONS = """For POSITIVE scenarios:
- The user's situation should satisfy ALL relevant rule conditions
- The expected outcome should be approval/success/fulfillment
- Include clear context showing why all rules pass
- Example: A customer requesting a refund on day 5 of a 30-day window with receipt"""

NEGATIVE_SCENARIO_INSTRUCTIONS = """For NEGATIVE scenarios:
- The user's situation should FAIL exactly ONE criterion
- Clearly identify which rule fails and why
- The expected outcome should be denial/rejection with explanation
- Example: A customer requesting a refund on day 45 of a 30-day window (violates R001)"""

EDGE_CASE_SCENARIO_INSTRUCTIONS = """For EDGE_CASE scenarios:
- The user's situation should be at EXACT boundaries
- Test limits, thresholds, and edge conditions
- The expected outcome depends on whether boundary is inclusive/exclusive
- Example: A customer requesting a refund on EXACTLY day 30 of a 30-day window"""

IRRELEVANT_SCENARIO_INSTRUCTIONS = """For IRRELEVANT scenarios:
- The user's query should NOT be addressed by ANY rule in the policy
- The expected outcome is a polite explanation that this is outside policy scope
- Should still be a reasonable customer inquiry, just unrelated
- Example: Asking about company history when policy only covers refunds"""


# =============================================================================
# STAGE 3: TRACE SYNTHESIS (The Thinker)
# =============================================================================

GOLDEN_TRACE_PROMPT = """You are a customer support agent generating a response with explicit reasoning.

POLICY DOCUMENT:
{policy_text}

LOGIC MAP (Rules to Apply):
{logic_map}

SCENARIO:
{scenario_description}

CONTEXT:
{scenario_context}

TARGET RULES: {target_rule_ids}
SCENARIO TYPE: {scenario_type}
EXPECTED OUTCOME: {expected_outcome}

YOUR TASK:
Generate a response with GROUNDED Chain-of-Thought reasoning.

CHAIN-OF-THOUGHT REQUIREMENTS:
1. For EACH relevant rule in the Logic Map:
   - State the rule (with Rule ID)
   - Evaluate whether it applies to this scenario
   - Explain WHY it applies or doesn't apply
   - If it doesn't apply, list which rules are EXCLUDED as a result

2. Follow the dependency order:
   - Evaluate root rules first
   - Then evaluate dependent rules only if their dependencies are satisfied

3. Be EXPLICIT about exclusions:
   - When a rule doesn't apply, state "R00X does NOT apply because..."
   - This prevents hallucination of non-applicable rules

RESPONSE REQUIREMENTS:
- messages: The conversation (system, user, assistant)
- reasoning_chain: Step-by-step reasoning with Rule IDs
- rules_applied: List of Rule IDs that were applied
- rules_excluded: List of Rule IDs that were explicitly excluded

CRITICAL - MESSAGE CONSTRUCTION RULES:

USER MESSAGE:
- Must contain ONLY the scenario_description text (the user's exact words)
- Must NOT include any information from the CONTEXT section
- Should read as a realistic query from someone who hasn't shared specific details yet

ASSISTANT MESSAGE:
- Use CONTEXT for internal reasoning (in reasoning_chain) only
- The assistant should respond as if it does NOT already know context details
- If context contains specific amounts/dates but user didn't state them:
  * Either ASK the user for those details, OR
  * Provide general policy guidance that would apply
- Do NOT act as if you magically know unstated information

EXAMPLE OF WHAT TO AVOID:
  User says: "I'd like to submit an expense"
  Context has: "$180, has receipt, 5 days ago"
  BAD response: "Your $180 expense with receipt from 5 days ago is approved!"  ← Knows unstated info!
  GOOD response: "I can help with that! Could you tell me the amount and whether you have a receipt?"

The assistant response should:
- Be professional and helpful
- Reference the policy naturally (without exposing Rule IDs to user)
- Provide clear next steps or explanations
- Only reference details the user actually stated"""


GOLDEN_TRACE_MULTI_TURN_PROMPT = """You are a customer support agent generating a multi-turn conversation with explicit reasoning.

POLICY DOCUMENT:
{policy_text}

LOGIC MAP (Rules to Apply):
{logic_map}

INITIAL SCENARIO:
{scenario_description}

CONTEXT:
{scenario_context}

TARGET RULES: {target_rule_ids}
SCENARIO TYPE: {scenario_type}
TARGET TURNS: {target_turns}

YOUR TASK:
Generate a {target_turns}-turn conversation where:
- Turn 1: Address the initial query with grounded reasoning
- Subsequent turns: Handle follow-up questions that probe deeper into the policy

MULTI-TURN GUIDELINES:
1. Each assistant response should have its own reasoning chain
2. Follow-up questions should test:
   - Clarifications (what about X?)
   - Edge cases (what if I...?)
   - Related rules (does this affect Y?)
3. Maintain context consistency across turns
4. Each turn should cite relevant Rule IDs in its reasoning

CRITICAL - MESSAGE CONSTRUCTION RULES:

TURN 1 - USER MESSAGE:
- Must contain ONLY the scenario_description (the user's exact words)
- Must NOT include details from CONTEXT
- Natural, conversational query without specific amounts/dates

TURN 1 - ASSISTANT MESSAGE:
- Use CONTEXT for reasoning but respond as if you don't know unstated details
- Either ask for needed details OR provide general guidance
- Do NOT "magically know" information the user didn't provide

SUBSEQUENT TURNS:
- User follow-ups may naturally reveal more details from CONTEXT
- This creates realistic information-gathering flow
- Assistant can reference details once user has stated them
- Each turn builds on previously shared information

GOOD MULTI-TURN FLOW:
  Turn 1 User: "I need to submit an expense"
  Turn 1 Assistant: "I can help! What type of expense and the amount?"
  Turn 2 User: "It's a client lunch for $180"
  Turn 2 Assistant: "For $180, you'll need manager approval. Do you have a receipt?"
  Turn 3 User: "Yes, I have a digital receipt"
  Turn 3 Assistant: "Great! Digital receipts are accepted. With manager approval and receipt, you're all set."

The final output should include:
- Complete conversation messages
- Reasoning chain for EACH assistant turn
- Cumulative rules_applied and rules_excluded"""


# =============================================================================
# STAGE 4: VERIFICATION (The Auditor)
# =============================================================================

VERIFICATION_PROMPT = """You are a verification system checking a trace against the Logic Map.

LOGIC MAP (Ground Truth):
{logic_map}

SCENARIO:
Type: {scenario_type}
Description: {scenario_description}
Target Rules: {target_rule_ids}
Expected Outcome: {expected_outcome}

GENERATED TRACE:
{trace_messages}

REASONING CHAIN PROVIDED:
{reasoning_chain}

RULES CLAIMED APPLIED: {rules_applied}
RULES CLAIMED EXCLUDED: {rules_excluded}

VERIFICATION CHECKS:

1. **No Skipped Rules**: Were all rules in target_rule_ids evaluated?
   - Check: Every rule in target_rule_ids should appear in reasoning_chain
   - Violation: If a target rule was never mentioned

2. **No Hallucinated Rules**: Were only valid rules cited?
   - Check: Every rule_id in reasoning_chain should exist in Logic Map
   - Violation: If a non-existent rule was referenced

3. **No Contradictions**: Is the reasoning internally consistent?
   - Check: If R001 is excluded, dependent rules shouldn't be applied
   - Check: Final response should match the reasoning conclusion
   - Violation: Saying "refund approved" when reasoning shows denial

4. **DAG Compliance**: Was dependency order followed?
   - Check: Parent rules evaluated before child rules
   - Violation: Evaluating R003 before its dependency R001

5. **Outcome Alignment**: Does response match expected outcome?
   - For POSITIVE: Should approve/allow
   - For NEGATIVE: Should deny/reject with reason
   - For EDGE_CASE: Should handle boundary correctly
   - For IRRELEVANT: Should redirect/explain scope

OUTPUT:
- passed: true/false
- issues: List of specific problems found
- skipped_rules: Rules that should have been applied but weren't
- hallucinated_rules: Rules cited that don't exist/don't apply
- contradictions: Logical contradictions found
- rules_verified: Rules that were correctly applied
- feedback: Summary of verification result"""


# =============================================================================
# GOLDEN REFINEMENT
# =============================================================================

GOLDEN_REFINE_PROMPT = """You are refining a trace that failed verification.

ORIGINAL TRACE:
{original_trace}

VERIFICATION FAILURE:
{verification_result}

LOGIC MAP (Ground Truth):
{logic_map}

SCENARIO:
{scenario_description}

ISSUES TO FIX:
- Skipped Rules: {skipped_rules}
- Hallucinated Rules: {hallucinated_rules}
- Contradictions: {contradictions}

YOUR TASK:
Generate a CORRECTED trace that:
1. Addresses ALL skipped rules in the reasoning chain
2. Removes references to hallucinated rules
3. Resolves all contradictions
4. Follows the DAG dependency order
5. Produces a response that matches the reasoning

REQUIREMENTS:
- Include complete reasoning_chain covering all target rules
- Ensure rules_applied only contains actually applicable rules
- Maintain professional, helpful tone in response
- Preserve the scenario context"""


# =============================================================================
# TOOL CALL SPECIFIC PROMPTS
# =============================================================================

GOLDEN_TOOL_TRACE_PROMPT = """You are a customer support agent with tools, generating a response with explicit reasoning.

POLICY DOCUMENT:
{policy_text}

LOGIC MAP (Rules to Apply):
{logic_map}

AVAILABLE TOOLS:
{tools_description}

SCENARIO:
{scenario_description}

CONTEXT:
{scenario_context}

TARGET RULES: {target_rule_ids}
SCENARIO TYPE: {scenario_type}

YOUR TASK:
Generate a response that may use tools, with GROUNDED reasoning.

TOOL USAGE REASONING:
When deciding whether to call a tool:
1. Reference which RULE requires this information
2. Explain why the tool is necessary to evaluate the rule
3. State what you expect to learn from the tool call

Example reasoning:
"To evaluate R002 (verify purchase date), I need the order details.
Calling get_order(order_id) to retrieve purchase date.
This will determine if the 30-day window applies."

RESPONSE STRUCTURE:
1. Reasoning chain with tool decisions tied to rules
2. Tool calls (if needed) with rule citations
3. Final response synthesizing tool results
4. rules_applied and rules_excluded lists

CRITICAL - MESSAGE CONSTRUCTION RULES:

USER MESSAGE:
- Must contain ONLY the scenario_description (the user's exact words)
- Must NOT include details from CONTEXT
- Natural query without specific amounts/dates the user hasn't stated

ASSISTANT MESSAGE:
- Use CONTEXT for reasoning but respond as if you don't know unstated details
- Tool calls should gather information the user hasn't provided
- Do NOT act as if you already know context details

The trace should include:
- System message with tool descriptions
- User message (scenario_description ONLY)
- Assistant message (with tool_calls if needed)
- Tool response messages (if tools were called)
- Final assistant response"""


__all__ = [
    "LOGIC_EXTRACTION_PROMPT",
    "GOLDEN_SCENARIO_PROMPT",
    "POSITIVE_SCENARIO_INSTRUCTIONS",
    "NEGATIVE_SCENARIO_INSTRUCTIONS",
    "EDGE_CASE_SCENARIO_INSTRUCTIONS",
    "IRRELEVANT_SCENARIO_INSTRUCTIONS",
    "GOLDEN_TRACE_PROMPT",
    "GOLDEN_TRACE_MULTI_TURN_PROMPT",
    "VERIFICATION_PROMPT",
    "GOLDEN_REFINE_PROMPT",
    "GOLDEN_TOOL_TRACE_PROMPT",
]
