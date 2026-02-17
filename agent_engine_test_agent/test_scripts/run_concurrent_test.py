#!/usr/bin/env python3
"""Test concurrent users against Agent Engine deployed agent.

Exercises:
  - Multiple users hitting the deployed agent simultaneously
  - Multi-level agent delegation
  - BQ analytics plugin logging from Agent Engine
  - Verifies rows in BigQuery with correct user isolation

Prerequisites:
  Deploy the agent first (see run_multi_turn_test.py for instructions).
  Then update AGENT_ENGINE_ID below.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

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


def query_agent(
    remote_app, user_id: str, session_id: str, message: str
) -> dict:
    """Send a single message to the deployed agent and return the response."""
    print(f"  [{user_id}] Sending: {message[:60]}...")
    try:
        response = remote_app.stream_query(
            user_id=user_id,
            session_id=session_id,
            message=message,
        )
        events = list(response)
        print(f"  [{user_id}] Got {len(events)} events")
        return {"user_id": user_id, "events": events, "error": None}
    except Exception as e:
        print(f"  [{user_id}] ERROR: {e}")
        return {"user_id": user_id, "events": [], "error": str(e)}


async def run_user_sequence(
    remote_app, user_id: str, messages: list[str]
) -> list[dict]:
    """Run a sequence of messages for one user in a thread."""
    loop = asyncio.get_event_loop()
    session = await loop.run_in_executor(
        None,
        lambda: remote_app.create_session(user_id=user_id),
    )
    session_id = session["id"]
    print(f"  [{user_id}] Created session: {session_id}")

    results = []
    for msg in messages:
        result = await loop.run_in_executor(
            None,
            lambda m=msg: query_agent(remote_app, user_id, session_id, m),
        )
        results.append(result)
    return results


async def main():
    print("=" * 70)
    print("AGENT ENGINE CONCURRENT USERS BQ ANALYTICS PLUGIN TEST")
    print("=" * 70)

    vertexai.init(project=PROJECT_ID, location=REGION)

    resource_name = (
        f"projects/{PROJECT_ID}/locations/{REGION}"
        f"/reasoningEngines/{AGENT_ENGINE_ID}"
    )
    remote_app = agent_engines.AgentEngine(resource_name)
    print(f"Connected to Agent Engine: {resource_name}")

    test_start = time.time()

    # ---------- Define test messages per user ----------
    user_messages = {
        "concurrent_user_1": [
            f"What datasets are available in project {PROJECT_ID}?",
            f"What tables are in the {DATASET_ID} dataset?",
        ],
        "concurrent_user_2": [
            f"List all datasets in project {PROJECT_ID}",
            f"Show me the schema of {DATASET_ID}.{TABLE_ID}",
        ],
        "concurrent_user_3": [
            "Hello, tell me a joke about data engineering.",
        ],
        "concurrent_user_4": [
            f"What tables exist in {DATASET_ID} dataset of {PROJECT_ID}?",
        ],
    }

    # ---------- Run all users concurrently ----------
    print(f"\n--- Running {len(user_messages)} concurrent users ---")
    tasks = [
        run_user_sequence(remote_app, uid, msgs)
        for uid, msgs in user_messages.items()
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(all_results):
        if isinstance(result, Exception):
            print(f"  User task {i} failed: {result}")

    print("\nWaiting 20s for BQ write propagation...")
    await asyncio.sleep(20)

    # ---------- Verify rows in BigQuery ----------
    print("\n--- Querying BigQuery for logged events ---")
    bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
    SELECT *
    FROM `{FULL_TABLE}`
    WHERE timestamp >= TIMESTAMP_SECONDS({int(test_start)})
      AND JSON_VALUE(attributes, '$.custom_tags.test') = 'eventdata_refactor'
      AND JSON_VALUE(attributes, '$.custom_tags.env') = 'agent_engine'
    ORDER BY timestamp
    """

    rows = list(bq_client.query(query).result())
    print(f"\nTotal rows logged: {len(rows)}")

    if not rows:
        print("WARNING: No rows yet, retrying in 30s...")
        await asyncio.sleep(30)
        rows = list(bq_client.query(query).result())
        print(f"Total rows after retry: {len(rows)}")

    if not rows:
        print("ERROR: No rows found.")
        return

    # ---------- Analysis ----------
    event_types = set()
    agents_seen = set()
    users_seen = set()
    sessions_seen = set()

    for row in rows:
        event_types.add(row.event_type)
        agents_seen.add(row.agent)
        users_seen.add(row.user_id)
        sessions_seen.add(row.session_id)

    print(f"\nEvent types: {sorted(event_types)}")
    print(f"Agents seen: {sorted(agents_seen)}")
    print(f"Users seen:  {sorted(users_seen)}")
    print(f"Sessions:    {len(sessions_seen)} distinct sessions")

    # ---------- Validation ----------
    errors = []

    if "orchestrator" not in agents_seen:
        errors.append("Root agent 'orchestrator' not in logged agents")

    expected_users = set(user_messages.keys())
    missing_users = expected_users - users_seen
    if missing_users:
        errors.append(f"Missing expected users: {missing_users}")

    # User isolation check
    session_users = {}
    for row in rows:
        session_users.setdefault(row.session_id, set()).add(row.user_id)
    for sid, uids in session_users.items():
        if len(uids) > 1:
            errors.append(f"Session {sid} has events from multiple users: {uids}")

    # Report
    print("\n" + "=" * 70)
    if errors:
        print(f"FAILURES ({len(errors)}):")
        for e in errors:
            print(f"  * {e}")
    else:
        print("ALL CHECKS PASSED")
        print(f"  {len(rows)} rows | {len(users_seen)} users | "
              f"{len(sessions_seen)} sessions | {len(agents_seen)} agents")
        print("  User isolation: VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
