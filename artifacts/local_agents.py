import os
import json
import time
import threading
from datetime import datetime
from openai import OpenAI

# ================= CONFIGURATION =================
# Point OpenAI client to your local LM Studio server
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
MODEL_NAME = "openai/gpt-oss-20b" # Replace or match whatever 20b model name is active in LM Studio
MEMORY_FILE = "agent_memory.json"
# =================================================

client = OpenAI(base_url=LM_STUDIO_API_BASE, api_key="not-needed")

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(history):
    with open(MEMORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# --- Multi-Step Multi-Agent Engine ---
def run_multi_agent_pipeline(user_input, history):
    print(f"\n[Agent System] Processing request through local {MODEL_NAME}...")

    # Step 1: Planner Agent breaks down the goal
    planner_system = (
        "You are a strategic Planner Agent. Analyze the user request and break it down "
        "into 2 or 3 short, concrete execution steps for an Executor Agent."
    )
    planner_messages = [{"role": "system", "content": planner_system}] + history[-4:] + [{"role": "user", "content": user_input}]

    try:
        planner_res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=planner_messages,
            temperature=0.3
        )
        plan = planner_res.choices[0].message.content
        print(f"\n--- [Planner Agent Output] ---\n{plan}\n------------------------------")
    except Exception as e:
        print(f"[Error connecting to LM Studio]: {e}")
        print("Ensure LM Studio server is running on http://localhost:1234")
        return history

    # Step 2: Executor Agent executes the plan and generates the final output
    executor_system = (
        "You are a precise Executor Agent. Given the user's initial prompt and the Planner's steps, "
        "synthesize a thorough, direct, and final response."
    )
    executor_messages = [
        {"role": "system", "content": executor_system},
        {"role": "user", "content": f"User Goal: {user_input}\n\nPlan to follow:\n{plan}"}
    ]

    try:
        exec_res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=executor_messages,
            temperature=0.7
        )
        final_answer = exec_res.choices[0].message.content
        print(f"\n--- [Executor Agent Output] ---\n{final_answer}\n------------------------------")

        # Append to conversational memory
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": final_answer})
        save_memory(history)

    except Exception as e:
        print(f"[Error during execution step]: {e}")

    return history

# --- Background Task Scheduler (Runs if CLI is left open) ---
def background_scheduler():
    """Example background loop that triggers every hour or checks tasks."""
    while True:
        time.sleep(3600) # Checks every hour
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Background Scheduler @ {timestamp}] CLI is still open. Running background memory optimization check...")
        history = load_memory()
        if len(history) > 20:
            # Simple truncation/housekeeping logic for persistent memory
            print("[Scheduler] Trimming old memory blocks to keep context lean.")
            save_memory(history[-20:])

def main():
    print("====================================================")
    print("  Local Multi-Agent CLI (LM Studio + OpenAI API)  ")
    print("====================================================")
    print(f"Connecting to: {LM_STUDIO_API_BASE}")
    print(f"Target Model:  {MODEL_NAME}")
    print("Type 'exit' or 'quit' to close. Memory is persistent.\n")

    # Start background scheduler thread
    t = threading.Thread(target=background_scheduler, daemon=True)
    t.start()

    history = load_memory()
    print(f"[Loaded {len(history)} past messages from local memory storage.]")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting CLI. Stay safe out there!")
                break

            history = run_multi_agent_pipeline(user_input, history)

        except KeyboardInterrupt:
            print("\nExiting gracefully.")
            break

if __name__ == "__main__":
    main()
