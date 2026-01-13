"""Prompt templates for interactive Logic Map editing."""

LOGIC_MAP_REFINEMENT_PROMPT = """You are a Logic Map editor. Your task is to modify a Logic Map based on user feedback.

CURRENT LOGIC MAP:
{current_logic_map}

ORIGINAL POLICY (for reference):
{policy_text}

PREVIOUS FEEDBACK IN THIS SESSION:
{conversation_history}

USER FEEDBACK:
{user_feedback}

INSTRUCTIONS:
Interpret the user's natural language request and modify the Logic Map accordingly.

SUPPORTED OPERATIONS:

1. **ADD**: Create a new rule
   - User might say: "add a rule for...", "include a rule about...", "there should be a rule for..."
   - Generate a new unique rule_id (use the next available number, e.g., if R008 exists, use R009)
   - Extract condition, action, and dependencies from context
   - Determine category based on rule type (CONSTRAINT, PERMISSION, PROCEDURE, EXCEPTION)

2. **REMOVE**: Delete a rule
   - User might say: "remove R005", "delete the rule about...", "R003 is not needed"
   - Remove the specified rule
   - Update dependencies in other rules that referenced the removed rule
   - Update root_rules if the removed rule was a root

3. **MERGE**: Combine two or more rules
   - User might say: "merge R002 and R003", "combine these rules into one"
   - Create a new rule that captures both conditions/actions
   - Remove the original rules
   - Update all dependencies that referenced the merged rules

4. **MODIFY**: Change an existing rule
   - User might say: "change R001 to...", "the condition for R002 should be...", "update R003's text"
   - Update the specified fields (text, condition, action, category)
   - Preserve rule_id and update dependencies if needed

5. **SPLIT**: Divide a rule into multiple rules
   - User might say: "split R001 into separate rules for X and Y"
   - Create new rules with sequential IDs
   - Remove original rule and update dependencies

6. **REORDER DEPENDENCIES**: Change rule relationships
   - User might say: "R003 should depend on R001", "remove dependency on R002 from R004"
   - Update the dependencies arrays accordingly
   - Ensure no circular dependencies are created

CRITICAL REQUIREMENTS:
- Maintain valid DAG structure (no circular dependencies)
- Ensure all rule_ids are unique
- Update root_rules list when dependencies change (root rules have no dependencies)
- Preserve existing rules that aren't affected by the change
- If the user's request is unclear, make a reasonable interpretation based on context

OUTPUT:
Return the complete updated Logic Map with ALL rules (both modified and unmodified).
Provide a brief changes_summary explaining what was done.
Provide reasoning explaining how you interpreted the user's feedback."""


HITL_INTENT_CLASSIFIER_PROMPT = """You are classifying user feedback in an interactive training data generation session.

CURRENT STATE:
- Conversation turns: {current_turns} ({complexity_level} complexity)
- Logic Map has {rule_count} rules
- Scenarios: {scenario_count} total
- Coverage: {coverage_summary}

PREVIOUS FEEDBACK IN THIS SESSION:
{conversation_history}

USER FEEDBACK: "{user_input}"

CLASSIFY THE INTENT:

1. "turns" - User wants to adjust conversation length/turns
   Examples: "shorter", "more thorough", "I want 5 turns", "make them brief", "longer conversations"
   → Set intent_type="turns", target_turns (1-6), and turns_reasoning
   Guidelines for target_turns:
   - "shorter" / "brief" / "quick" / "simple" → 1-2 turns
   - "normal" / "moderate" / "standard" → 3-4 turns
   - "longer" / "deeper" / "thorough" / "more detail" → 5-6 turns
   - Specific numbers like "3 turns" or "I want 4" → use that exact number

2. "rules" - User wants to modify the Logic Map rules
   Examples: "remove R005", "add a rule for...", "merge R002 and R003", "change R001 to..."
   → Set intent_type="rules" and rule_feedback to the original user input

3. "scenarios" - User wants to add/delete/modify scenarios or adjust distribution
   Examples:
   - "add a scenario for late submissions" → scenario_operation="add"
   - "delete S3" → scenario_operation="delete", scenario_target="S3"
   - "remove the refund scenario" → scenario_operation="delete", scenario_target="the refund scenario"
   - "change S2 to test edge cases" → scenario_operation="modify", scenario_target="S2"
   - "more negative scenarios" → scenario_operation="distribution"
   - "fewer edge cases" → scenario_operation="distribution"
   - "delete all irrelevant scenarios" → scenario_operation="delete", scenario_target="all irrelevant"
   → Set intent_type="scenarios", scenario_operation, scenario_target (if applicable), and scenario_feedback

4. "compound" - User wants BOTH rule changes AND scenario changes in one request
   Examples:
   - "add a rule for alcohol refunds and create 2 scenarios for it"
   - "add a rule about late fees, then add some negative scenarios testing that rule"
   - "create a rule for VIP discounts and add edge case scenarios for the boundary conditions"
   - "remove R005 and delete all scenarios that reference it"
   → Set intent_type="compound", rule_feedback (the rule part), AND scenario_feedback (the scenario part)
   → The system will execute rules first, then scenarios, so scenarios can reference newly added rules

5. "coverage" - User wants to view or improve coverage metrics
   Examples:
   - "show coverage" → coverage_operation="view", coverage_view_mode="summary"
   - "show coverage gaps" / "what's missing" → coverage_operation="view", coverage_view_mode="gaps"
   - "show heatmap" → coverage_operation="view", coverage_view_mode="heatmap"
   - "increase coverage for refunds by 20%" → coverage_operation="increase", coverage_target_sub_category="refunds", coverage_increase_amount=20
   - "add more negative scenarios for time eligibility" → coverage_operation="increase", coverage_target_sub_category="time eligibility", coverage_scenario_type="negative"
   - "improve coverage of amount thresholds" → coverage_operation="increase", coverage_target_sub_category="amount thresholds"
   - "get amount thresholds to 80%" → coverage_operation="target", coverage_target_sub_category="amount thresholds", coverage_target_percent=80
   - "fully cover time rules" / "100% coverage for X" → coverage_operation="target", coverage_target_sub_category="X", coverage_target_percent=100
   - "do the suggestions" / "apply suggestions" / "do what you suggested" → coverage_operation="increase" (apply the coverage suggestions shown)
   - "fix the gaps" / "fill the gaps" / "address the gaps" → coverage_operation="increase" (improve coverage for gap areas)
   - "improve coverage" / "increase coverage" / "better coverage" → coverage_operation="increase"
   → Set intent_type="coverage" and the appropriate coverage_* fields
   → When user says "do suggestions" or "fix gaps", set coverage_operation="increase" without specific target (system picks lowest coverage)

6. "command" - User typed a built-in command (done, undo, reset, help, show Rxxx, show Sxxx)
   → Set intent_type="command", leave other fields null
   Note: Commands are handled separately, but classify them if they appear

7. "unclear" - Cannot determine intent
   → Set intent_type="unclear"

IMPORTANT:
- Set confidence based on how clear the intent is (0.0 to 1.0)
- Use "compound" when the user explicitly wants BOTH rule AND scenario changes in ONE request
- Use "coverage" for any requests about viewing or improving coverage metrics
- Default to "rules" if ambiguous between rules and unclear
- Default to "scenarios" if ambiguous between scenarios and unclear"""


SCENARIO_REFINEMENT_PROMPT = """You are a scenario editor for training data generation. Your task is to modify scenarios based on user feedback.

LOGIC MAP (for rule references):
{logic_map}

CURRENT SCENARIOS:
{scenarios_formatted}

CURRENT DISTRIBUTION:
{distribution}

ORIGINAL POLICY (for context):
{policy_text}

PREVIOUS FEEDBACK IN THIS SESSION:
{conversation_history}

USER FEEDBACK:
{user_feedback}

INSTRUCTIONS:
Interpret the user's natural language request and modify the scenarios accordingly.

SUPPORTED OPERATIONS:

1. **ADD**: Create a new scenario
   - User might say: "add a scenario for...", "include a test case for...", "there should be a scenario about..."
   - Create scenario with appropriate type (positive, negative, edge_case, irrelevant)
   - Set target_rule_ids to rules this scenario tests
   - Write expected_outcome based on rule evaluation

2. **DELETE**: Remove scenario(s)
   - User might say: "delete S3", "remove the refund scenario", "delete all irrelevant scenarios"
   - Match by ID (S1, S2...) or by description/content
   - Can delete multiple scenarios if user requests

3. **MODIFY**: Change an existing scenario
   - User might say: "change S2 to...", "update S5 to test edge cases", "S3 should be negative"
   - Update specified fields while preserving scenario_id
   - Ensure target_rule_ids are updated if scenario focus changes

4. **DISTRIBUTION**: Adjust type distribution
   - User might say: "more negative scenarios", "fewer edge cases", "add more positive examples"
   - Add/remove scenarios to achieve requested distribution
   - Maintain total count unless user specifies otherwise

SCENARIO ID MAPPING:
Scenarios are displayed as S1, S2, S3... (1-indexed).
User may reference by:
- ID: "S3", "S5"
- Description: "the refund scenario", "the one about late submissions"
- Type: "all negative scenarios", "edge cases"

CRITICAL REQUIREMENTS:
- Ensure target_rule_ids reference valid rules from the Logic Map
- Maintain scenario type validity (positive, negative, edge_case, irrelevant)
- Write clear, testable expected_outcome for each scenario
- Preserve scenarios not affected by the change

OUTPUT:
Return the complete updated scenarios list with ALL scenarios (both modified and unmodified).
Provide a brief changes_summary explaining what was done.
Provide reasoning explaining how you interpreted the user's feedback."""


__all__ = [
    "LOGIC_MAP_REFINEMENT_PROMPT",
    "HITL_INTENT_CLASSIFIER_PROMPT",
    "SCENARIO_REFINEMENT_PROMPT",
]
