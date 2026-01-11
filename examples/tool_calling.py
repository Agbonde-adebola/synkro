"""
Synkro Tool Calling Example
============================

Generate training data for teaching models to use tools correctly.

This example shows how to:
1. Define tools with ToolDefinition
2. Create a TOOL_CALL pipeline
3. Generate traces with specialized tool grading & refinement
4. Save traces in OpenAI function calling format

Tool call traces now use specialized ToolCallGrader 
and ToolCallRefiner that evaluate tool-specific criteria:
- Tool Selection: Did they use the right tool?
- Parameter Accuracy: Were the parameters correct?
- Response Synthesis: Did they use tool results correctly?
- Timing: Did they call tools at the right time?

Output format follows OpenAI's function calling format:
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": null, "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "...", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from synkro import create_pipeline, ToolDefinition, DatasetType
from synkro.models.google import Google
from synkro.reporting import FileLoggingReporter

# =============================================================================
# Step 1: Define your tools
# =============================================================================

# Web search tool
web_search = ToolDefinition(
    name="web_search",
    description="Search the web for current information. Use for real-time data, news, prices, weather, etc.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query - be specific and include relevant keywords"
            }
        },
        "required": ["query"]
    },
    # Example responses help the LLM generate realistic simulated tool outputs
    mock_responses=[
        "Current weather in NYC: 72°F, sunny with light clouds",
        "Bitcoin price: $67,234.56 USD (as of 2:30 PM EST)",
        "Python 3.12 was released on October 2, 2023",
    ]
)

# Database query tool
customer_lookup = ToolDefinition(
    name="lookup_customer",
    description="Look up customer information by ID or email. Returns account details, tier, and recent activity.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Customer ID (format: CUST-XXXXX)"
            },
            "email": {
                "type": "string",
                "description": "Customer email address"
            }
        }
    },
    mock_responses=[
        '{"id": "CUST-12345", "name": "Jane Smith", "tier": "Premium", "since": "2022-01-15"}',
        '{"id": "CUST-67890", "name": "John Doe", "tier": "Basic", "since": "2024-06-01"}',
        '{"error": "Customer not found"}',
    ]
)

# Internal API tool
create_ticket = ToolDefinition(
    name="create_support_ticket",
    description="Create a support ticket in the internal system. Use for issues that need escalation or tracking.",
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Customer ID for the ticket"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],
                "description": "Ticket priority level"
            },
            "category": {
                "type": "string",
                "description": "Issue category (billing, technical, account, other)"
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the issue"
            }
        },
        "required": ["customer_id", "priority", "category", "description"]
    },
    mock_responses=[
        '{"ticket_id": "TKT-98765", "status": "created", "estimated_response": "2 hours"}',
        '{"ticket_id": "TKT-11111", "status": "created", "estimated_response": "24 hours"}',
    ]
)

# =============================================================================
# Step 2: Define usage guidelines (your "policy" for tool usage)
# =============================================================================

TOOL_USAGE_GUIDELINES = """
Customer Support Agent Tool Usage Guidelines
=============================================

You are a customer support agent with access to tools. Follow these rules:

1. ALWAYS look up customer information before providing account-specific help
2. Use web_search for questions about:
   - Current prices, rates, or market data
   - Recent news or events
   - Information that changes frequently
3. Do NOT use tools when you can answer from general knowledge:
   - Company policies (use your training)
   - General product features
   - Common troubleshooting steps
4. Create support tickets for:
   - Issues you cannot resolve directly
   - Requests requiring backend changes
   - Complaints that need documentation
5. Priority levels:
   - urgent: System outages, security issues
   - high: Billing errors, locked accounts
   - medium: Feature requests, general bugs
   - low: Questions, feedback

IMPORTANT: Never hallucinate tool results. If a tool returns an error or
no results, tell the customer honestly and offer alternatives.
"""

# =============================================================================
# Step 3: Create pipeline and generate traces
# =============================================================================

# Use FileLoggingReporter for both CLI output and file logging
reporter = FileLoggingReporter(log_dir="./logs")

pipeline = create_pipeline(
    dataset_type=DatasetType.TOOL_CALL,
    tools=[web_search, customer_lookup, create_ticket],
    model=Google.GEMINI_25_FLASH,
    grading_model=Google.GEMINI_25_FLASH,
    max_iterations=2,
    reporter=reporter,                  # Log to both CLI and file
)

# Generate training traces
# Each trace will demonstrate correct tool usage patterns
dataset = pipeline.generate(TOOL_USAGE_GUIDELINES, traces=20)

# =============================================================================
# Step 4: Save and inspect
# =============================================================================

# Save with tool_call format
dataset.save("tool_training.jsonl", format="tool_call")

# View summary
print("\n" + dataset.summary())

# Show grading stats
passed = sum(1 for t in dataset.traces if t.grade and t.grade.passed)
total = len(dataset.traces)
print(f"\n📊 Grading: {passed}/{total} traces passed ({passed/total*100:.0f}%)")

# Show a sample trace
if len(dataset) > 0:
    print("\n--- Sample Trace ---")
    trace = dataset[0]
    for msg in trace.messages:
        print(f"[{msg.role}]", end=" ")
        if msg.tool_calls:
            print(f"<tool_calls: {[tc.function.name for tc in msg.tool_calls]}>")
        elif msg.tool_call_id:
            print(f"<tool_response for {msg.tool_call_id}>: {msg.content[:50]}...")
        else:
            content = msg.content or ""
            print(content[:80] + "..." if len(content) > 80 else content)
    
    # Show grade info if available
    if trace.grade:
        print(f"\n📝 Grade: {'✓ PASS' if trace.grade.passed else '✗ FAIL'}")
        if trace.grade.feedback:
            print(f"   Feedback: {trace.grade.feedback[:100]}...")

