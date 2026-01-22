"""Prompt templates for coverage tracking and analysis.

These prompts support the coverage tracking system:
1. Sub-category extraction - Extract testable sub-categories from policy
2. Scenario tagging - Tag scenarios with sub-category IDs
3. Coverage suggestions - Generate improvement suggestions
4. Coverage commands - Parse natural language coverage commands
"""

# =============================================================================
# SUB-CATEGORY EXTRACTION
# =============================================================================

SUBCATEGORY_EXTRACTION_PROMPT = """You are a policy analyst extracting a sub-category taxonomy from a policy document.

POLICY DOCUMENT:
{policy_text}

EXTRACTED LOGIC MAP (Rules):
{logic_map}

CATEGORIES (from planning):
{categories}

YOUR TASK:
Extract a hierarchical sub-category taxonomy that organizes the policy into testable units.

EXTRACTION GUIDELINES:

1. **Identify Distinct Concepts**: Within each category, find specific testable aspects.
   - Look for: thresholds, time limits, permissions, restrictions, procedures, exceptions
   - Each sub-category should represent a distinct concept that can be tested independently

2. **Map to Rules**: Link each sub-category to relevant rules from the Logic Map.
   - A sub-category can relate to multiple rules
   - A rule can belong to multiple sub-categories if it spans concepts

3. **Assign Priority**:
   - HIGH: Critical business rules, security-related, compliance-required, frequently triggered
   - MEDIUM: Standard operational rules, common scenarios
   - LOW: Edge cases, rare situations, nice-to-have coverage

4. **Aim for Granularity**:
   - Target 2-5 sub-categories per parent category
   - Each sub-category should be specific enough to generate targeted test scenarios
   - Avoid sub-categories that are too broad or overlap significantly

5. **Use Clear Names**:
   - Names should be concise (2-5 words)
   - Names should indicate what aspect is being tested

EXAMPLES:

For a "Refund Policy" category, sub-categories might be:
- SC001: "Time-based eligibility" - Rules about the 30-day refund window
- SC002: "Amount thresholds" - Rules about maximum refund amounts
- SC003: "Method restrictions" - Rules about cash vs. credit refunds
- SC004: "Exception cases" - VIP overrides, manager discretion

For an "Expense Approval" category:
- SC005: "Dollar thresholds" - Approval limits by amount
- SC006: "Category restrictions" - What can/cannot be expensed
- SC007: "Documentation requirements" - Receipt and justification rules
- SC008: "Approver hierarchy" - Who can approve what

OUTPUT FORMAT:
Provide a taxonomy with:
- sub_categories: List of SubCategory objects, each with:
  - id: Unique identifier (SC001, SC002, etc.)
  - name: Short descriptive name
  - description: What this sub-category covers
  - parent_category: Which category this belongs to
  - related_rule_ids: Rule IDs from the Logic Map
  - priority: high/medium/low
- reasoning: Brief explanation of how you organized the taxonomy"""


# =============================================================================
# SCENARIO TAGGING
# =============================================================================

SCENARIO_SUBCATEGORY_TAGGING_PROMPT = """You are tagging scenarios with the sub-categories they cover.

AVAILABLE SUB-CATEGORIES:
{sub_categories_formatted}

LOGIC MAP (for reference):
{logic_map}

SCENARIOS TO TAG:
{scenarios_formatted}

YOUR TASK:
For each scenario, identify which sub-category(ies) it covers based on:
1. The scenario's target_rule_ids - if a rule belongs to a sub-category, the scenario likely covers it
2. The scenario's description - what aspect of the policy is being tested
3. The scenario's expected_outcome - what rules/concepts are being validated

TAGGING GUIDELINES:

1. **Use Rule Mappings First**: If a scenario targets rules R001 and R003, and those rules
   are in sub-categories SC001 and SC002, tag the scenario with those sub-categories.

2. **Consider Implicit Coverage**: A scenario might cover a sub-category even without
   directly targeting its rules. For example, a scenario testing time-based eligibility
   implicitly covers the "time eligibility" sub-category.

3. **Multiple Tags Are OK**: A scenario can cover multiple sub-categories if it tests
   multiple concepts. This is common for complex scenarios.

4. **At Least One Tag**: Every scenario should cover at least one sub-category.
   If none seem to fit, choose the closest match.

OUTPUT FORMAT:
For each scenario, provide:
- scenario_index: The index (0-based) of the scenario
- sub_category_ids: List of sub-category IDs this scenario covers"""


# =============================================================================
# COVERAGE SUGGESTIONS
# =============================================================================

COVERAGE_SUGGESTIONS_PROMPT = """You are a coverage advisor analyzing scenario coverage gaps.

COVERAGE REPORT:
- Total Sub-Categories: {total_sub_categories}
- Covered (>={covered_threshold}%): {covered_count}
- Partial ({partial_threshold}-{covered_threshold}%): {partial_count}
- Uncovered (<{partial_threshold}%): {uncovered_count}
- Overall Coverage: {overall_coverage_percent:.1f}%

DETAILED COVERAGE BY SUB-CATEGORY:
{coverage_details}

GAPS IDENTIFIED (Uncovered or Partial):
{gaps}

POLICY CONTEXT:
{policy_summary}

LOGIC MAP SUMMARY:
{logic_map_summary}

YOUR TASK:
Generate actionable suggestions to improve coverage.

SUGGESTION GUIDELINES:

1. **Prioritize by Impact**:
   - HIGH priority uncovered sub-categories first
   - Then MEDIUM priority uncovered
   - Then partial coverage sub-categories
   - Then LOW priority

2. **Be Specific**:
   - Name the exact sub-category to target
   - Specify the scenario type to add (positive/negative/edge_case)
   - Reference specific rules that need testing
   - Suggest how many scenarios to add

3. **Balance Coverage**:
   - If a sub-category only has positive scenarios, suggest negative/edge cases
   - If missing edge cases, highlight boundary conditions to test
   - Ensure type diversity within each sub-category

4. **Quick Wins First**:
   - Suggest easy-to-add scenarios before complex ones
   - Prioritize scenarios that can cover multiple gaps

SUGGESTION FORMAT:
Each suggestion should follow this pattern:
"Add [N] [type] scenario(s) for '[sub-category name]' testing [specific rule/concept]"

Examples:
- "Add 2 negative scenarios for 'Time eligibility' testing the 30-day boundary"
- "Add 1 edge_case scenario for 'Amount thresholds' testing the exact $500 limit"
- "Add 3 positive scenarios for 'Documentation requirements' covering receipt uploads"

OUTPUT:
Provide 5-10 prioritized, actionable suggestions."""


# =============================================================================
# COVERAGE COMMAND PARSING
# =============================================================================

COVERAGE_COMMAND_PROMPT = """You are interpreting a natural language coverage command.

CURRENT COVERAGE STATE:
{coverage_summary}

AVAILABLE SUB-CATEGORIES:
{sub_categories_list}

USER COMMAND: "{user_input}"

YOUR TASK:
Parse the user's command into a structured coverage intent.

COMMAND TYPES:

1. **VIEW Commands** - User wants to see coverage information:
   - "show coverage" -> view, mode=summary
   - "show gaps" / "what's missing" -> view, mode=gaps
   - "show heatmap" / "show coverage map" -> view, mode=heatmap
   - "show coverage for [X]" -> view, mode=detail, target=[X]

2. **INCREASE Commands** - User wants to add scenarios:
   - "increase coverage for [X] by [N]%" -> increase, target=[X], amount=[N]
   - "add more [type] scenarios for [X]" -> increase, target=[X], scenario_type=[type]
   - "improve coverage of [X]" -> increase, target=[X]
   - "boost [X]" -> increase, target=[X]

3. **TARGET Commands** - User wants to reach a specific coverage level:
   - "get [X] to [N]% coverage" -> target, target=[X], percent=[N]
   - "fully cover [X]" / "100% coverage for [X]" -> target, target=[X], percent=100
   - "ensure [X] is covered" -> target, target=[X], percent=80 (default)

MATCHING SUB-CATEGORIES:
When the user mentions a sub-category by name or concept:
- Match case-insensitively
- Allow partial matches ("refunds" matches "Time-based refund eligibility")
- If ambiguous, choose the most relevant match

OUTPUT FORMAT:
Provide:
- operation: "view" | "increase" | "target"
- view_mode: (for view) "summary" | "gaps" | "heatmap" | "detail"
- target_sub_category: (for increase/target) sub-category name or ID
- target_percent: (for target) target percentage
- increase_amount: (for increase with percentage) amount to increase
- scenario_type: (if specified) "positive" | "negative" | "edge_case"

If the command is unclear, set operation to "view" and view_mode to "summary" as a safe default."""


# =============================================================================
# TARGETED SCENARIO GENERATION
# =============================================================================

TARGETED_SCENARIO_GENERATION_PROMPT = """You are generating scenarios to improve coverage for a specific sub-category.

POLICY DOCUMENT:
{policy_text}

LOGIC MAP:
{logic_map}

TARGET SUB-CATEGORY:
- ID: {sub_category_id}
- Name: {sub_category_name}
- Description: {sub_category_description}
- Related Rules: {related_rule_ids}
- Priority: {priority}

CURRENT COVERAGE:
- Current scenarios covering this sub-category: {current_count}
- Current coverage percentage: {current_percent}%
- Existing scenario types: {existing_types}

TARGET:
- Generate {count} new scenarios
- Preferred type(s): {preferred_types}
- Avoid duplicating: {existing_descriptions}

REQUIREMENTS:

1. **Focus on the Target Sub-Category**: All scenarios should test aspects of this sub-category.
   - Reference the related rules
   - Test the specific concept this sub-category covers

2. **Diversify Scenario Types**:
   - If existing scenarios are mostly positive, generate negative/edge_case
   - Aim for type balance within the sub-category

3. **Create Distinct Scenarios**:
   - Each scenario should test a different aspect or combination
   - Avoid scenarios too similar to existing ones
   - Vary the context, user situation, and specific details

4. **Maintain Quality**:
   - Realistic user descriptions (natural language)
   - Clear expected outcomes
   - Accurate rule targeting

OUTPUT FORMAT:
For each scenario, provide:
- description: The user's exact words
- context: Background information
- scenario_type: positive/negative/edge_case/irrelevant
- target_rule_ids: Rules this scenario tests
- expected_outcome: What should happen based on rules
- sub_category_ids: Should include {sub_category_id}"""


# =============================================================================
# COVERAGE TARGET GENERATION (LLM-driven)
# =============================================================================

COVERAGE_TARGET_GENERATION_PROMPT = """You are generating scenarios to reach a target coverage percentage.

CURRENT COVERAGE STATE:
- Overall Coverage: {current_overall:.0f}%
- Target Coverage: {target_percent}%
- Gap to Close: {gap:.0f} percentage points

PER-SUB-CATEGORY BREAKDOWN:
{sub_category_coverage_table}

EXISTING SCENARIOS (to avoid duplicating):
{existing_scenarios_summary}

LOGIC MAP (Rules to test):
{logic_map}

POLICY DOCUMENT:
{policy_text}

YOUR TASK:
Generate scenarios to improve overall coverage from {current_overall:.0f}% to {target_percent}%.

STEP 1 - CALCULATE SCENARIOS NEEDED:
Before generating any scenarios, analyze the coverage breakdown and determine:
- How many sub-categories are uncovered (0%) or partial (<80%)?
- For each gap sub-category, how many scenarios would bring it to adequate coverage?
- What is the TOTAL number of scenarios needed to reach {target_percent}% overall?

Consider:
- Each sub-category typically needs 2-4 scenarios to be "covered" (80%+)
- Uncovered sub-categories (0%) need more scenarios than partial ones
- HIGH priority sub-categories should get more scenarios
- Overall coverage = average of sub-category coverages

STEP 2 - PLAN THE DISTRIBUTION:
Based on your calculation, decide:
- Which sub-categories to target (prioritize uncovered, then partial, then HIGH priority)
- How many scenarios for each sub-category
- What types to generate (positive/negative/edge_case) based on what's missing

STEP 3 - GENERATE SCENARIOS:
Now generate EXACTLY the number of scenarios you calculated in Step 1.

GUIDELINES:
- Spread scenarios across multiple sub-categories (don't target just one)
- Each scenario should specify which sub_category_ids it covers
- Reference specific rules from the Logic Map
- Write realistic user descriptions in natural language
- Ensure type diversity (don't generate all positive or all negative)

OUTPUT FORMAT:
For each scenario, provide:
- description: The user's exact words (realistic customer/user query)
- context: Background information about the situation
- scenario_type: positive/negative/edge_case
- target_rule_ids: Rules this scenario tests (from the Logic Map)
- expected_outcome: What should happen based on the rules
- sub_category_ids: Which sub-categories this scenario covers"""


# =============================================================================
# COVERAGE IMPROVEMENT 3-CALL WORKFLOW
# =============================================================================

COVERAGE_PLANNING_PROMPT = """You are analyzing coverage gaps and creating a MINIMAL plan to reach a specific target.

CURRENT COVERAGE STATE:
- Overall Coverage: {current_overall:.0f}%
- Target Coverage: {target_percent}%
- Gap to Close: {gap:.0f} percentage points

PER-SUB-CATEGORY BREAKDOWN:
{sub_category_coverage_table}

EXISTING SCENARIOS COUNT: {existing_count}

YOUR TASK:
Create a plan to generate the MINIMUM scenarios needed to reach EXACTLY {target_percent}% coverage.

CRITICAL CONSTRAINTS:
1. DO NOT exceed the target - if target is 30%, stop at ~30%
2. DO NOT try to cover all gaps - only enough to reach the target
3. Generate the MINIMUM scenarios needed, not the maximum

COVERAGE MATH:
- Overall coverage = average of all sub-category coverages
- With N sub-categories, improving 1 sub-category from 0% to 100% adds ~(100/N)% to overall
- Example: 5 sub-categories at 0%, target 20% → only need to cover 1 sub-category (1/5 = 20%)
- Example: 5 sub-categories at 0%, target 40% → only need to cover 2 sub-categories (2/5 = 40%)

PLANNING STEPS:

1. **Calculate**: How many sub-categories need improvement to reach {target_percent}%?
   - Current overall: {current_overall:.0f}%
   - Need to gain: {gap:.0f} percentage points
   - Each sub-category at 100% contributes equally to the average

2. **Select Targets**: Pick the MINIMUM sub-categories to improve:
   - Prioritize HIGH priority over MEDIUM over LOW
   - Prefer uncovered (0%) sub-categories - more impact per scenario

3. **Plan Scenarios**: For ONLY the selected sub-categories:
   - 2-3 scenarios typically brings a sub-category to 80%+
   - Include type diversity (positive/negative/edge_case)

IMPORTANT: Your total_scenarios in the plan should be the minimum needed.
If target is low (e.g., 20-30%), you may only need 2-4 scenarios total.

OUTPUT:
Provide a structured plan targeting ONLY the sub-categories needed to reach {target_percent}%."""


COVERAGE_EXECUTION_PROMPT = """You are generating scenarios according to a coverage improvement plan.

PLAN TO EXECUTE:
{plan_summary}

SPECIFIC TARGETS:
{plan_details}

POLICY DOCUMENT:
{policy_text}

LOGIC MAP (Rules to test):
{logic_map}

YOUR TASK:
Generate scenarios exactly as specified in the plan above.

FOR EACH SCENARIO:
- Match the target sub-category from the plan
- Use the specified scenario type (positive/negative/edge_case)
- Focus on the aspects mentioned in the plan
- Reference specific rules from the Logic Map
- Write realistic user descriptions in natural language

QUALITY REQUIREMENTS:
- Each scenario must be distinct and test different aspects
- Descriptions should sound like real user queries
- Expected outcomes must align with the rules
- Context should provide relevant background

OUTPUT FORMAT:
For each scenario, provide:
- description: The user's exact words (realistic query)
- context: Background information
- scenario_type: As specified in plan
- target_rule_ids: Rules this scenario tests
- expected_outcome: What should happen based on rules
- sub_category_ids: The target sub-category from plan"""


SCENARIO_DEDUPLICATION_PROMPT = """You are reviewing generated scenarios to remove duplicates and near-duplicates.

EXISTING SCENARIOS (already in the dataset):
{existing_scenarios}

NEWLY GENERATED SCENARIOS (to review):
{generated_scenarios}

YOUR TASK:
Identify which newly generated scenarios should be REMOVED because they are:
1. Exact duplicates of existing scenarios
2. Near-duplicates (testing the same thing with minor wording changes)
3. Redundant (covering the exact same rules/edge cases as existing scenarios)

KEEP scenarios that:
- Test genuinely different aspects or edge cases
- Approach the same rule from a meaningfully different angle
- Add value by covering gaps not addressed by existing scenarios

REMOVE scenarios that:
- Are too similar to existing scenarios in intent and coverage
- Would not add new testing value to the dataset
- Duplicate the exact same test case with different wording

Be CONSERVATIVE with removals - only remove if clearly redundant.
When in doubt, KEEP the scenario.

OUTPUT:
- kept_indices: List of indices (0-based) of generated scenarios to KEEP
- removed_indices: List of indices (0-based) of generated scenarios to REMOVE
- removal_reasons: For each removed scenario, explain why it was removed"""
