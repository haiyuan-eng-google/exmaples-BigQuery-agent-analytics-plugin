# requirements:
#   pip install "google-cloud-aiplatform[agent_engines]>=1.126.0"
# auth:
#   gcloud auth application-default login
import os
import uuid
import vertexai
# ---- Fill these in (or set as env vars) ----
PROJECT_ID = "your project id"
LOCATION = "your location"
# --- UPDATE THIS ---
AGENT_ID = "your agent id after deployment"

# -------------------
AGENT_NAME = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{AGENT_ID}"
)


import argparse
import concurrent.futures
import time
import random


def stress_test_worker(agent, agent_id, worker_id):
  """Worker function to simulate a user session."""
  user_id = f"stress_user_{worker_id}_{uuid.uuid4().hex[:4]}"
  print(f"[Worker {worker_id}] Starting session {user_id}")

  # 1. Set State
  state_key = f"key_{worker_id}"
  state_val = f"val_{worker_id}"
  msg_set = (
      f"Call function set_state with key='{state_key}' and value='{state_val}'"
  )
  print(f"[Worker {worker_id}] Sending: {msg_set}")
  try:
    response_text = ""
    for chunk in agent.stream_query(message=msg_set, user_id=user_id):
      response_text += chunk.text if hasattr(chunk, "text") else str(chunk)
    print(
        f"[Worker {worker_id}] Response to set_state: {response_text.strip()}"
    )
  except Exception as e:
    print(f"[Worker {worker_id}] Error in set_state: {e}")

  # 2. Query State (Implicitly via tool or just checking robustness)
  msg_query = "What tools do you have?"
  print(f"[Worker {worker_id}] Sending: {msg_query}")
  try:
    response_text = ""
    for chunk in agent.stream_query(message=msg_query, user_id=user_id):
      response_text += chunk.text if hasattr(chunk, "text") else str(chunk)
    print(
        f"[Worker {worker_id}] Response to query:"
        f" {response_text.strip()[:50]}..."
    )
  except Exception as e:
    print(f"[Worker {worker_id}] Error in query: {e}")

  return f"[Worker {worker_id}] Finished"


def main():
  parser = argparse.ArgumentParser(description="Test Agent Engine Agent")
  parser.add_argument(
      "--stress-test",
      action="store_true",
      help="Run stress test with concurrent users",
  )
  parser.add_argument(
      "--concurrency", type=int, default=5, help="Number of concurrent threads"
  )
  parser.add_argument(
      "--agent-id", type=str, default=AGENT_ID, help="Agent Engine Agent ID"
  )
  args = parser.parse_args()

  # Update AGENT_NAME if ID is provided
  current_agent_id = args.agent_id
  agent_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{current_agent_id}"

  vertexai.init(project=PROJECT_ID, location=LOCATION)
  client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
  try:
    print(f"Fetching agent: {agent_name}")
    # This returns an 'AgentEngine' object in your SDK version
    agent = client.agent_engines.get(name=agent_name)

    if args.stress_test:
      print(
          f"Running stress test with {args.concurrency} threads on agent"
          f" {current_agent_id}..."
      )
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=args.concurrency
      ) as executor:
        futures = [
            executor.submit(stress_test_worker, agent, current_agent_id, i)
            for i in range(args.concurrency)
        ]
        for future in concurrent.futures.as_completed(futures):
          print(future.result())
      print("Stress test complete.")
      return

    user_id = f"interactive_user_{uuid.uuid4().hex[:8]}"
    print(f"Starting interactive session with user_id: {user_id}")
    print("Type 'exit' or 'quit' to end.\n")
    while True:
      user_message = input("You: ")
      if user_message.lower() in ("exit", "quit"):
        print("Exiting.")
        break
      print("Agent: ", end="", flush=True)
      try:
        # REVERTED to standard stream_query matching your original working code
        for chunk in agent.stream_query(message=user_message, user_id=user_id):
          print(chunk, end="", flush=True)
      except Exception as e:
        print(f"\nError during query: {e}")
      print("")  # Newline
  except Exception as e:
    print(f"An error occurred: {e}")


if __name__ == "__main__":
  main()