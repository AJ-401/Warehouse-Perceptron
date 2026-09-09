"""
run.py

A simple interactive CLI so you can type questions at the assistant
directly, without writing a Python script each time. Lives at the
project root (not inside assistant/) since it's an entry point, not
a module other code imports.

Usage:
    python run_assistant.py

It will:
  1. Load data/warehouse_events.json via EventStore
  2. Use GroqLLMClient when GROQ_API_KEY is set, otherwise fall back
      to MockLLMClient (no network calls) so this always runs during setup.
  3. Show a small menu of the built-in query functions, plus free-form
     "ask anything" mode.
"""

from __future__ import annotations
import os
import sys

from assistant.event_loader import EventStore, EventLoadError
from assistant.llm_client import (
    get_llm_client,
    GeminiLLMClient,
    GroqLLMClient,
    MockLLMClient,
    LLMClientError,
)
from assistant import queries as q


def get_llm(username: str):
    memory_file = f"data/memory/memory_{username}.json" if username else None
    client = get_llm_client(memory_file=memory_file)
    model_name = getattr(client, "model", getattr(client, "current_model", type(client).__name__))
    print(f"Using {model_name} [1M Token Context Window].\n")
    return client




def print_result(result) -> None:
    print("\n--- Answer ---")
    print(result.answer)
    print(f"\n(grounded in {len(result.event_ids)} event(s): {', '.join(result.event_ids) or 'none'})")
    print("-" * 40 + "\n")


def main() -> None:
    username = input("Enter your username/ID to load your memory: ").strip().lower()
    
    try:
        store = EventStore.load()
    except EventLoadError as e:
        print(f"Could not load event data: {e}")
        sys.exit(1)

    print(f"Loaded {len(store.events)} events from data/warehouse_events.json.\n")
    llm = get_llm(username)

    while True:
        question = input("\nWhat do you want to ask? (or 'q' to quit)\n> ").strip()

        if question.lower() in ("q", "quit", "exit"):
            print("Bye.")
            break

        if question:
            try:
                print_result(q.ask(store, llm, question))
            except Exception as e:
                error_str = str(e)
                if any(x in error_str.lower() for x in ("limit exceeded", "429", "resource_exhausted", "rate limit")):
                    print("\n[!] Limit exceeded: API request limit reached. No waiting needed — please try again in a bit or try another question.\n")
                else:
                    print(f"\n[!] An error occurred while contacting the LLM: {e}\n")


if __name__ == "__main__":
    main()
