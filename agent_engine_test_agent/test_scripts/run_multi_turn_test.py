#!/usr/bin/env python3
"""Multi-turn multi-subagent test on Agent Engine.

Turn 1: Data question -> orchestrator -> data_team -> schema_explorer (BQ tools)
Turn 2: Image question -> orchestrator -> image_describer (describe_this_image tool)
Turn 3: Another data question -> orchestrator -> data_team again

Then queries BQ and validates the same checks as the local test.

Prerequisites:
  Deploy the agent first:
    adk deploy agent_engine \
      --project=YOUR_PROJECT_ID \
      --region=us-central1 \
      --adk_app_object=app \
      ../bq_test_agent

  Then update AGENT_ENGINE_ID below with the deployed ID.

Usage:
  python run_multi_turn_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict

import vertexai
from vertexai import agent_engines
from google.cloud import bigquery

# TODO: Replace with your own values
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DATASET_ID = os.environ.get("BQ_DATASET", "agent_analytics")
TABLE_ID = "agent_events_v2"
FULL_TABLE = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
AGENT_ENGINE_ID = "YOUR_AGENT_ENGINE_ID"  # TODO: Update after deployment
TEST_TAG = "multi_turn_subagent_ae"

TEST_IMAGE_URI = (
    "https://storage.googleapis.com/cloud-samples-data"
    "/generative-ai/image/scones.jpg"
)


def query_agent(remote_app, user_id, session_id, message):
    print(f"  [{user_id}] Sending: {message[:70]}...")
    try:
        response = remote_app.stream_query(
            user_id=user_id, session_id=session_id, message=message,
        )
        events = list(response)
        print(f"  [{user_id}] Got {len(events)} events")
        return {"user_id": user_id, "events": events, "error": None}
    except Exception as e:
        print(f"  [{user_id}] ERROR: {e}")
        return {"user_id": user_id, "events": [], "error": str(e)}


async def main():
    print("=" * 70)
    print("MULTI-TURN MULTI-SUBAGENT AGENT ENGINE TEST")
    print("=" * 70)

    vertexai.init(project=PROJECT_ID, location=REGION)
    resource_name = (
        f"projects/{PROJECT_ID}/locations/{REGION}"
        f"/reasoningEngines/{AGENT_ENGINE_ID}"
    )
    remote_app = agent_engines.AgentEngine(resource_name)
    print(f"Connected: {resource_name}")

    test_start = time.time()
    loop = asyncio.get_event_loop()
    user_id = "ae_multi_turn_user"

    # Create a session
    session = await loop.run_in_executor(
        None, lambda: remote_app.create_session(user_id=user_id),
    )
    session_id = session["id"]
    print(f"Session: {session_id}")

    # --- Turn 1: Data question ---
    print("\n--- Turn 1: Data question ---")
    result1 = await loop.run_in_executor(
        None,
        lambda: query_agent(
            remote_app, user_id, session_id,
            f"What datasets are available in project {PROJECT_ID}?"
        ),
    )

    # --- Turn 2: Image question ---
    print("\n--- Turn 2: Image description ---")
    result2 = await loop.run_in_executor(
        None,
        lambda: query_agent(
            remote_app, user_id, session_id,
            f"Please describe this image: {TEST_IMAGE_URI}"
        ),
    )

    # --- Turn 3: Another data question ---
    print("\n--- Turn 3: Another data question ---")
    result3 = await loop.run_in_executor(
        None,
        lambda: query_agent(
            remote_app, user_id, session_id,
            f"What tables are in the {DATASET_ID} dataset of {PROJECT_ID}?"
        ),
    )

    for i, res in enumerate([result1, result2, result3], 1):
        if res["error"]:
            print(f"  Turn {i} had error: {res['error']}")

    print("\nWaiting 25s for BQ write propagation...")
    await asyncio.sleep(25)

    # --- Query BQ ---
    print("\n--- Querying BQ ---")
    bq = bigquery.Client(project=PROJECT_ID)
    q = f"""
    SELECT *
    FROM `{FULL_TABLE}`
    WHERE timestamp >= TIMESTAMP_SECONDS({int(test_start)})
      AND JSON_VALUE(attributes, '$.custom_tags.test') = 'eventdata_refactor'
      AND JSON_VALUE(attributes, '$.custom_tags.env') = 'agent_engine'
      AND user_id = '{user_id}'
    ORDER BY timestamp
    """
    rows = list(bq.query(q).result())
    print(f"Total rows: {len(rows)}")

    if not rows:
        print("WARNING: No rows yet, retrying in 30s...")
        await asyncio.sleep(30)
        rows = list(bq.query(q).result())
        print(f"After retry: {len(rows)}")

    if not rows:
        print("ERROR: No rows found. Aborting.")
        return

    # --- Analysis ---
    errors = []
    by_agent = defaultdict(list)
    by_invocation = defaultdict(list)

    for r in rows:
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

    agents_seen = sorted(by_agent.keys())
    invocations = sorted(by_invocation.keys())
    tool_events = [r for r in rows if r.event_type in ("TOOL_STARTING", "TOOL_COMPLETED", "TOOL_ERROR")]

    # Checks
    if len({r.session_id for r in rows}) != 1:
        errors.append("Session continuity failed")
    if len(invocations) < 3:
        errors.append(f"Expected >= 3 invocations, got {len(invocations)}")
    if "orchestrator" not in agents_seen:
        errors.append("'orchestrator' not found")

    starts = [r for r in tool_events if r.event_type == "TOOL_STARTING"]
    completes = [r for r in tool_events if r.event_type == "TOOL_COMPLETED"]
    tool_errs = [r for r in tool_events if r.event_type == "TOOL_ERROR"]
    if len(starts) != len(completes) + len(tool_errs):
        errors.append(f"Tool pairing mismatch")

    # Report
    print(f"\n  Agents: {agents_seen}")
    print(f"  Tool events: {len(tool_events)}")

    print("\n" + "=" * 70)
    if errors:
        print(f"ISSUES FOUND ({len(errors)}):")
        for e in errors:
            print(f"  * {e}")
    else:
        print("ALL MULTI-TURN AGENT ENGINE CHECKS PASSED")
        print(f"  {len(rows)} rows | {len(invocations)} turns | "
              f"{len(agents_seen)} agents | {len(tool_events)} tool events")
    print("=" * 70)

    # Cleanup
    print("\nDeleting Agent Engine...")
    try:
        remote_app.delete(force=True)
        print("Deleted.")
    except Exception as e:
        print(f"Delete failed (manual cleanup needed): {e}")


if __name__ == "__main__":
    asyncio.run(main())
