"""Prompt templates for interactive Logic Map editing."""

LOGIC_MAP_REFINEMENT_PROMPT = """You are a Logic Map editor. Your task is to modify a Logic Map based on user feedback.

CURRENT LOGIC MAP:
{current_logic_map}

ORIGINAL POLICY (for reference):
{policy_text}

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


HITL_INTENT_CLASSIFIER_PROMPT = """You are classifying user feedback in an interactive SFT data generation session.

CURRENT STATE:
- Conversation turns: {current_turns} ({complexity_level} complexity)
- Logic Map has {rule_count} rules

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

3. "command" - User typed a built-in command (done, undo, reset, help, show Rxxx)
   → Set intent_type="command", leave other fields null
   Note: Commands are handled separately, but classify them if they appear

4. "unclear" - Cannot determine intent
   → Set intent_type="unclear"

IMPORTANT:
- Set confidence based on how clear the intent is (0.0 to 1.0)
- If the user mentions both turns and rules, prioritize the more prominent intent
- Default to "rules" if slightly ambiguous between rules and unclear"""


__all__ = ["LOGIC_MAP_REFINEMENT_PROMPT", "HITL_INTENT_CLASSIFIER_PROMPT"]
