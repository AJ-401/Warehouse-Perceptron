"""
llm_client.py

Thin wrapper around the LLM API. Nothing fancy on purpose — queries.py
builds the (system, user) pair, this module just sends it and returns
text. Keeping the API call in one place means if the team ever swaps
providers or models, it's a one-file change.

The production backend is Groq. MockLLMClient provides the same interface
for tests without making network calls.

--- Groq (recommended, free) ---
    pip install groq
    export GROQ_API_KEY=your_key_here
    Get a key at: https://console.groq.com/keys (no card required)

"""

from __future__ import annotations
import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import groq
except ImportError:  # pragma: no cover
    groq = None

DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


class LLMClientError(Exception):
    """Raised when the LLM client is misconfigured or the API call fails."""


class GroqLLMClient:
    """
    Groq backend — free, and the most generous free daily request limits
    of the options here. Recommended default for this hackathon.

    Get a key at https://console.groq.com/keys (no card required), then:
        export GROQ_API_KEY=your_key_here

    Uses an OpenAI-compatible chat completions format under the hood.
    If the default model below ever gets retired, check
    https://console.groq.com/docs/models for the current free model list
    and set GROQ_MODEL instead of editing this file.
    """

    def __init__(self, model: str = DEFAULT_GROQ_MODEL, max_tokens: int = 1024, memory_file: str = None):
        if groq is None:
            raise LLMClientError(
                "The 'groq' package isn't installed. Run: pip install groq"
            )
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMClientError(
                "GROQ_API_KEY is not set. Export it before running the assistant."
            )
        self._client = groq.Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.memory_file = memory_file
        # Note: History memory accumulation is removed to ensure statelessness per query

    def ask(self, system: str, user_message: str, context: str = "") -> str:
        messages = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "user", "content": f"{user_message}\n\n{context}"})
        else:
            messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
            )
        except Exception as e:
            raise LLMClientError(
                f"Groq API call failed: {e}\n"
                f"(If this is a 404/decommissioned-model error, check "
                f"https://console.groq.com/docs/models for the current free "
                f"model list and set GROQ_MODEL to one of them.)"
            ) from e

        return (response.choices[0].message.content or "").strip()

    def ask_with_tools(self, system: str, user_message: str, tools: list[dict], tool_handler: callable) -> str:
        messages = []
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    messages = json.load(f)
            except Exception:
                pass
                
        if not messages:
            messages = [{"role": "system", "content": system}]
            
        messages.append({"role": "user", "content": user_message})

        try:
            # First pass: let the LLM decide which tools to call
            # We copy the messages array so we don't pollute the persistent history with tool outputs
            api_messages = list(messages)
            
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=api_messages,
                tools=tools,
                tool_choice="auto",
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            final_content = ""
            
            if tool_calls:
                api_messages.append(response_message)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    tool_result = tool_handler(function_name, function_args)
                    
                    api_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(tool_result),
                    })
                    
                # Second pass: let the LLM synthesize the final answer using tool outputs
                final_response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=api_messages,
                )
                final_content = (final_response.choices[0].message.content or "").strip()
            else:
                final_content = (response_message.content or "").strip()
                
            messages.append({"role": "assistant", "content": final_content})
            
            # Save ONLY the user questions and final assistant answers back to disk
            if self.memory_file:
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                with open(self.memory_file, 'w') as f:
                    json.dump(messages, f, indent=4)
                    
            return final_content
                
        except Exception as e:
            raise LLMClientError(f"Groq API call (with tools) failed: {e}") from e


class MockLLMClient:
    """
    Drop-in replacement for LLMClient that doesn't call the network —
    useful for testing queries.py and the dashboard integration before
    an API key is configured, or in CI. Just echoes back a summary of
    what it was asked, so you can verify the RIGHT events reached the
    prompt without spending API calls.
    """

    def __init__(self, memory_file: str = None, *_, **__):
        self.memory_file = memory_file

    def ask(self, system: str, user_message: str, context: str = "") -> str:
        return f"[MOCK] I would answer your question: '{user_message}' using {len(context)} bytes of context."

    def ask_with_tools(self, system: str, user_message: str, tools: list[dict], tool_handler: callable) -> str:
        return f"[MOCK] I would answer your question: '{user_message}' using tools."
