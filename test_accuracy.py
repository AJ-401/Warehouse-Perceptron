import os
from assistant.event_loader import EventStore
from assistant.llm_client import GroqLLMClient
from assistant.queries import ask

def test_accuracy():
    store = EventStore.load()
    llm = GroqLLMClient()
    
    questions = [
        "How many total events are there in the warehouse?",
        "Tell me about event EVT-20260908-0050",
        "Which events are Critical?",
        "How many near misses did we have today?",
    ]
    
    for q in questions:
        print(f"Q: {q}")
        res = ask(store, llm, q)
        print(f"A: {res.answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_accuracy()
