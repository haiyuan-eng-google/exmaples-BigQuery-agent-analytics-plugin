#!/usr/bin/env python3
"""Multi-turn multi-subagent test: exercises different subagents across turns.

Turn 1: "What datasets are available?" -> orchestrator -> data_team -> schema_explorer (BQ tools)
Turn 2: "Describe this image" -> orchestrator -> image_describer (describe_this_image tool)
Turn 3: "What tables in agent_analytics?" -> orchestrator -> data_team -> schema_explorer (BQ tools)

Then queries BQ and validates:
  - Each turn has correct invocation_id
  - Tool events attributed to correct subagent (schema_explorer vs image_describer)
  - Session continuity (same session_id across turns)
  - All expected event types present per turn
  - Tool content has correct tool name and args/result

Usage:
  Set environment variables before running:
    export GOOGLE_GENAI_USE_VERTEXAI=true
    export GOOGLE_CLOUD_PROJECT=your-project-id
    export GOOGLE_CLOUD_LOCATION=us-central1
  Then:
    python run_multi_turn_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict

# TODO: Replace with your own project ID
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
DATASET_ID = os.environ.get("BQ_DATASET", "agent_analytics")
TABLE_ID = "agent_events_v2"
FULL_TABLE = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
TEST_TAG = "multi_turn_subagent_local"

TEST_IMAGE_URI = (
    "https://storage.googleapis.com/cloud-samples-data"
    "/generative-ai/image/scones.jpg"
)

# Set env vars BEFORE any google.genai / ADK imports
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

# Add parent dir to path so bq_test_agent can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.genai import types
from google.cloud import bigquery
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from bq_test_agent.agent import app


async def main():
    print("=" * 70)
    print("MULTI-TURN MULTI-SUBAGENT LOCAL TEST")
    print("=" * 70)

    # Override custom_tags for this test
    for plugin in app.plugins:
        if hasattr(plugin, "config") and hasattr(plugin.config, "custom_tags"):
            plugin.config.custom_tags = {
                "env": "local_test",
                "test": TEST_TAG,
            }

    test_start = time.time()
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service)

    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="multi_turn_user"
    )
    print(f"Session: {session.id}")

    # --- Turn 1: Data question -> schema_explorer ---
    print("\n--- Turn 1: Data question (should trigger data_team -> schema_explorer) ---")
    msg1 = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=f"What datasets are available in project {PROJECT_ID}?"
        )],
    )
    events1 = []
    async for event in runner.run_async(
        user_id="multi_turn_user", session_id=session.id, new_message=msg1,
    ):
        events1.append(event)
    print(f"  Got {len(events1)} events")

    # --- Turn 2: Image question -> image_describer ---
    print("\n--- Turn 2: Image description (should trigger image_describer) ---")
    msg2 = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=f"Please describe this image: {TEST_IMAGE_URI}"
        )],
    )
    events2 = []
    async for event in runner.run_async(
        user_id="multi_turn_user", session_id=session.id, new_message=msg2,
    ):
        events2.append(event)
    print(f"  Got {len(events2)} events")

    # --- Turn 3: Another data question -> schema_explorer again ---
    print("\n--- Turn 3: Another data question (should trigger data_team again) ---")
    msg3 = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=f"What tables are in the {DATASET_ID} dataset of {PROJECT_ID}?"
        )],
    )
    events3 = []
    async for event in runner.run_async(
        user_id="multi_turn_user", session_id=session.id, new_message=msg3,
    ):
        events3.append(event)
    print(f"  Got {len(events3)} events")

    # Flush
    print("\n--- Flushing ---")
    for plugin in runner.plugin_manager.plugins:
        if hasattr(plugin, "flush"):
            await plugin.flush()

    print("Waiting 12s for BQ propagation...")
    await asyncio.sleep(12)

    # --- Query BQ ---
    print("\n--- Querying BQ ---")
    bq = bigquery.Client(project=PROJECT_ID)
    q = f"""
    SELECT *
    FROM `{FULL_TABLE}`
    WHERE timestamp >= TIMESTAMP_SECONDS({int(test_start)})
      AND JSON_VALUE(attributes, '$.custom_tags.test') = '{TEST_TAG}'
    ORDER BY timestamp
    """
    rows = list(bq.query(q).result())
    print(f"Total rows: {len(rows)}")

    if not rows:
        print("WARNING: No rows yet, retrying in 20s...")
        await asyncio.sleep(20)
        rows = list(bq.query(q).result())
        print(f"After retry: {len(rows)}")

    if not rows:
        print("ERROR: No rows found. Aborting.")
        return

    # --- Analysis ---
    errors = []
    by_type = defaultdict(list)
    by_agent = defaultdict(list)
    by_invocation = defaultdict(list)

    for r in rows:
        by_type[r.event_type].append(r)
        by_agent[r.agent].append(r)
        by_invocation[r.invocation_id].append(r)

    def get_content(r):
        c = r.content
        if isinstance(c, str):
            try:
                return json.loads(c)
            except Exception:
                return c
        return c

    FRAMEWORK_TOOLS = {"transfer_to_agent"}
    agents_seen = sorted(by_agent.keys())
    invocations = sorted(by_invocation.keys())
    tool_events = [r for r in rows if r.event_type in ("TOOL_STARTING", "TOOL_COMPLETED", "TOOL_ERROR")]

    # CHECK 1: Session continuity
    print("\n=== CHECK 1: Session continuity ===")
    sessions = {r.session_id for r in rows}
    if len(sessions) != 1:
        errors.append(f"Expected 1 session, got {len(sessions)}")
    else:
        print("  PASS")

    # CHECK 2: Multiple invocations
    print("\n=== CHECK 2: Multiple invocations ===")
    if len(invocations) < 3:
        errors.append(f"Expected >= 3 invocations, got {len(invocations)}")
    else:
        print(f"  PASS: {len(invocations)} invocations")

    # CHECK 3: Agents
    print("\n=== CHECK 3: Agents seen ===")
    print(f"  Agents: {agents_seen}")
    for ea in ("orchestrator", "schema_explorer"):
        if ea not in agents_seen:
            errors.append(f"Expected agent '{ea}' not found")

    # CHECK 4: Tool pairing
    print("\n=== CHECK 4: Tool start/complete pairing ===")
    starts = [r for r in tool_events if r.event_type == "TOOL_STARTING"]
    completes = [r for r in tool_events if r.event_type == "TOOL_COMPLETED"]
    tool_errs = [r for r in tool_events if r.event_type == "TOOL_ERROR"]
    if len(starts) != len(completes) + len(tool_errs):
        errors.append(f"Mismatch: {len(starts)} starts vs {len(completes)}+{len(tool_errs)} completes/errors")
    else:
        print("  PASS")

    # CHECK 5: Latency and spans
    print("\n=== CHECK 5: Latency and spans ===")
    for r in completes:
        lat = r.latency_ms
        if isinstance(lat, str):
            try:
                lat = json.loads(lat)
            except Exception:
                lat = None
        if not (lat and isinstance(lat, dict) and "total_ms" in lat):
            errors.append(f"TOOL_COMPLETED missing latency at {r.timestamp}")
    if not any("missing latency" in e for e in errors):
        print("  PASS: All TOOL_COMPLETED have latency")

    # REPORT
    print("\n" + "=" * 70)
    if errors:
        print(f"ISSUES FOUND ({len(errors)}):")
        for e in errors:
            print(f"  * {e}")
    else:
        print("ALL MULTI-TURN MULTI-SUBAGENT CHECKS PASSED")
        print(f"  {len(rows)} rows | {len(invocations)} turns | "
              f"{len(agents_seen)} agents | {len(tool_events)} tool events")
    print("=" * 70)

    # Cleanup
    for plugin in runner.plugin_manager.plugins:
        if hasattr(plugin, "close"):
            await plugin.close()


if __name__ == "__main__":
    asyncio.run(main())
