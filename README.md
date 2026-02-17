# Examples-BigQuery-Agent-Analytics-Plugin
Example code repo for using BigQuery Agent Analytics Plugin in different scenarios.

## Repository Structure

```
.
├── local_test_agent/                  # Local ADK runner tests
│   ├── bq_test_agent/                 # Multi-agent app with BQ analytics plugin
│   │   ├── __init__.py
│   │   └── agent.py                   # 3-level agent hierarchy + plugin config
│   └── test_scripts/
│       ├── run_multi_turn_test.py     # Multi-turn multi-subagent BQ validation
│       └── run_comprehensive_pr_test.py  # Full PR #4491 feature verification
│
├── agent_engine_test_agent/           # Vertex AI Agent Engine tests
│   ├── bq_test_agent/                 # Same agent, configured for Agent Engine
│   │   ├── __init__.py
│   │   ├── agent.py                   # Uses os.environ for project config
│   │   └── requirements.txt           # Dependencies for Agent Engine
│   └── test_scripts/
│       ├── run_multi_turn_test.py     # Multi-turn test against deployed agent
│       ├── run_concurrent_test.py     # Concurrent users + user isolation
│       └── run_full_audit.py          # Full audit: multimodal + data quality
│
├── agent-engine-example.py            # Standalone Agent Engine example
├── fast_api_example.py                # FastAPI integration example
└── test-agent-engine-example.py       # Agent Engine test example
```

## Agent Architecture

Both test agents use the same 3-level multi-agent hierarchy:

```
orchestrator (LlmAgent, root)
  ├── data_team (SequentialAgent)
  │     ├── schema_explorer (LlmAgent, BQ tools)
  │     └── query_analyst (LlmAgent, BQ tools)
  └── image_describer (LlmAgent, multimodal tool)
```

## Features Tested

The test scripts validate the following BigQuery Agent Analytics Plugin features:

1. **EventData dataclass** - Typed container replacing `**kwargs` in `_log_event`
2. **STATE_DELTA logging** - State changes logged via `on_event_callback`
3. **Session metadata** - `log_session_metadata` captures session state
4. **API surface compatibility** - `batch_processor`, `write_client`, `write_stream` attributes
5. **Multi-subagent tool attribution** - Tool events attributed to correct agent
6. **Tool content structure** - `tool`, `args`, `result` keys in content
7. **Latency tracking** - `total_ms` on TOOL_COMPLETED and LLM_RESPONSE
8. **Span/trace consistency** - `span_id` and `parent_span_id` on completed events
9. **Session continuity** - Same `session_id` across multi-turn conversations
10. **Custom tags** - Static tags propagated to all BQ rows
11. **User isolation** - No cross-contamination between concurrent users

## Quick Start

### Local Testing

```bash
# Set up environment
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1

# Run the comprehensive test
cd local_test_agent
python test_scripts/run_comprehensive_pr_test.py
```

### Agent Engine Testing

```bash
# Deploy the agent
adk deploy agent_engine \
  --project=your-project-id \
  --region=us-central1 \
  --adk_app_object=app \
  agent_engine_test_agent/bq_test_agent

# Update AGENT_ENGINE_ID in the test script, then run
cd agent_engine_test_agent
python test_scripts/run_multi_turn_test.py
```
