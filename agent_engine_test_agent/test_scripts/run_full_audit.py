#!/usr/bin/env python3
"""Full Agent Engine audit: concurrent users + multimodal + deep BQ row check.

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

TEST_IMAGE_URI = (
    "https://storage.googleapis.com/cloud-samples-data"
    "/generative-ai/image/scones.jpg"
)


def query_agent(remote_app, user_id, session_id, message):
    print(f"  [{user_id}] Sending: {message[:60]}...")
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


async def run_user_sequence(remote_app, user_id, messages):
    loop = asyncio.get_event_loop()
    session = await loop.run_in_executor(
        None, lambda: remote_app.create_session(user_id=user_id),
    )
    session_id = session["id"]
    print(f"  [{user_id}] Session: {session_id}")
    results = []
    for msg in messages:
        result = await loop.run_in_executor(
            None, lambda m=msg: query_agent(remote_app, user_id, session_id, m),
        )
        results.append(result)
    return results


async def main():
    print("=" * 70)
    print("AGENT ENGINE FULL AUDIT -- concurrent + multimodal + data quality")
    print("=" * 70)

    vertexai.init(project=PROJECT_ID, location=REGION)
    resource_name = (
        f"projects/{PROJECT_ID}/locations/{REGION}"
        f"/reasoningEngines/{AGENT_ENGINE_ID}"
    )
    remote_app = agent_engines.AgentEngine(resource_name)
    print(f"Connected: {resource_name}")

    test_start = time.time()

    # 5 concurrent users
    user_messages = {
        "ae_user_data1": [
            f"What datasets are available in project {PROJECT_ID}?",
            f"What tables are in the {DATASET_ID} dataset?",
        ],
        "ae_user_data2": [
            f"Show me the schema of {DATASET_ID}.{TABLE_ID} in {PROJECT_ID}",
        ],
        "ae_user_image1": [
            f"Please describe this image: {TEST_IMAGE_URI}",
        ],
        "ae_user_image2": [
            f"What is in this picture? {TEST_IMAGE_URI}",
        ],
        "ae_user_chat": [
            "Hello! Tell me a joke about databases.",
        ],
    }

    print(f"\n--- Running {len(user_messages)} concurrent users ---")
    tasks = [
        run_user_sequence(remote_app, uid, msgs)
        for uid, msgs in user_messages.items()
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(all_results):
        if isinstance(result, Exception):
            print(f"  User task {i} EXCEPTION: {result}")

    print("\nWaiting 25s for BQ write propagation...")
    await asyncio.sleep(25)

    # Query BQ
    print("\n--- Querying BQ ---")
    bq = bigquery.Client(project=PROJECT_ID)
    q = f"""
    SELECT *
    FROM `{FULL_TABLE}`
    WHERE timestamp >= TIMESTAMP_SECONDS({int(test_start)})
      AND JSON_VALUE(attributes, '$.custom_tags.test') = 'eventdata_refactor'
      AND JSON_VALUE(attributes, '$.custom_tags.env') = 'agent_engine'
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
        print("ERROR: No rows found. Aborting audit.")
        return

    # Comprehensive audit
    errors = []
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.event_type].append(r)

    def get_attrs(r):
        a = r.attributes
        if isinstance(a, str):
            try:
                return json.loads(a)
            except Exception:
                return {}
        return a if isinstance(a, dict) else {}

    def get_content(r):
        c = r.content
        if isinstance(c, str):
            try:
                return json.loads(c)
            except Exception:
                return c
        return c

    def get_latency(r):
        lat = r.latency_ms
        if isinstance(lat, str):
            try:
                return json.loads(lat)
            except Exception:
                return None
        return lat

    # CHECK 1: Required fields
    print("\n=== CHECK 1: Required fields ===")
    for r in rows:
        for fld in ("event_type", "agent", "user_id", "session_id",
                     "invocation_id", "status", "timestamp"):
            if getattr(r, fld) is None:
                errors.append(f"Row at {r.timestamp}: {fld} is null")
    c1_errs = [e for e in errors if "is null" in e]
    print(f"  {len(rows)} rows checked: {'PASS' if not c1_errs else f'{len(c1_errs)} nulls'}")

    # CHECK 2: Trace/span consistency
    print("\n=== CHECK 2: Trace/span consistency ===")
    for et in ("AGENT_COMPLETED", "LLM_RESPONSE", "TOOL_COMPLETED"):
        total = len(by_type.get(et, []))
        with_span = sum(1 for r in by_type.get(et, []) if r.span_id)
        with_lat = sum(1 for r in by_type.get(et, []) if get_latency(r))
        print(f"  {et}: span_id={with_span}/{total}  latency={with_lat}/{total}")
        if with_span < total:
            errors.append(f"{et}: {total - with_span} missing span_id")

    # CHECK 3: STATE_DELTA
    print("\n=== CHECK 3: STATE_DELTA ===")
    sd_total = len(by_type.get("STATE_DELTA", []))
    sd_ok = sum(1 for r in by_type.get("STATE_DELTA", []) if "state_delta" in get_attrs(r))
    print(f"  With state_delta attr: {sd_ok}/{sd_total}")

    # CHECK 4: Tool content
    print("\n=== CHECK 4: TOOL event content ===")
    tool_ok = tool_total = 0
    for et in ("TOOL_STARTING", "TOOL_COMPLETED"):
        for r in by_type.get(et, []):
            tool_total += 1
            content = get_content(r)
            if isinstance(content, dict) and "tool" in content:
                tool_ok += 1
    print(f"  With tool name: {tool_ok}/{tool_total}")

    # CHECK 5: User isolation
    print("\n=== CHECK 5: User isolation ===")
    session_users = defaultdict(set)
    for r in rows:
        session_users[r.session_id].add(r.user_id)
    cross = {sid: uids for sid, uids in session_users.items() if len(uids) > 1}
    if cross:
        errors.append(f"Cross-contaminated sessions: {cross}")
    else:
        print(f"  PASS: {len(session_users)} sessions, all single-user")

    # CHECK 6: Multimodal
    print("\n=== CHECK 6: Multimodal content_parts ===")
    image_parts = 0
    for r in rows:
        for p in (r.content_parts or []):
            if isinstance(p, dict) and "image" in p.get("mime_type", ""):
                image_parts += 1
    print(f"  Image parts: {image_parts}")

    # Report
    agents_seen = sorted({r.agent for r in rows})
    users_seen = sorted({r.user_id for r in rows})

    print("\n" + "=" * 70)
    if errors:
        print(f"ISSUES FOUND ({len(errors)}):")
        for e in errors:
            print(f"  * {e}")
    else:
        print("ALL AGENT ENGINE DATA QUALITY CHECKS PASSED")
        print(f"  {len(rows)} rows | {len(by_type)} event types "
              f"| {len(agents_seen)} agents | {len(users_seen)} users "
              f"| {len(session_users)} sessions")
        print(f"  User isolation: VERIFIED")
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
