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
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import groq
except ImportError:  # pragma: no cover
    groq = None

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None

DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")

# Primary Gemini models (gemini-2.5-pro removed as requested):
GEMINI_MODEL_CHAIN = [
    os.environ.get("GEMINI_MODEL", "gemini-3.7-flash"),
    "gemini-3-flash-preview",
]

# Fallback Groq models with separate independent quotas:
GROQ_MODEL_CHAIN = [
    os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
]


class LLMClientError(Exception):
    """Raised when the LLM client is misconfigured or the API call fails."""


class GeminiLLMClient:
    """
    Google Gemini backend using the modern google-genai SDK.
    Provides a 1,000,000+ token context window, eliminating token limit issues.
    Supports native tool calling and persistent session memory.
    """

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, memory_file: str = None):
        if genai is None:
            raise LLMClientError(
                "The 'google-genai' package is not installed. Run: pip install google-genai"
            )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMClientError(
                "GEMINI_API_KEY is not set. Export it or set it in .env before running the assistant."
            )
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.memory_file = memory_file

    def ask(self, system: str, user_message: str, context: str = "") -> str:
        prompt = user_message
        if context:
            prompt = f"{user_message}\n\n{context}"

        config = types.GenerateContentConfig(
            system_instruction=system,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip()
        except Exception as e:
            raise LLMClientError(f"Gemini API call failed: {e}") from e

    def ask_with_tools(self, system: str, user_message: str, tools: list[dict], tool_handler: callable) -> str:
        # Build function declarations from OpenAI-style tool schemas
        function_declarations = []
        for t in tools:
            fn = t.get("function", t)
            function_declarations.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {})
            })
        gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Load session history if available
        history = []
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    saved = json.load(f)
                    for m in saved:
                        role = m.get("role")
                        if role in ("user", "assistant"):
                            gemini_role = "user" if role == "user" else "model"
                            history.append(
                                types.Content(
                                    role=gemini_role,
                                    parts=[types.Part.from_text(text=m.get("content", ""))]
                                )
                            )
            except Exception:
                pass

        def _execute_or_fail(call_fn):
            try:
                return call_fn()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    raise LLMClientError(f"Limit exceeded: Rate limit reached for '{self.model}'.") from e
                raise

        try:
            chat = self._client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=gemini_tools,
                ),
                history=history
            )

            response = _execute_or_fail(lambda: chat.send_message(user_message))

            # Handle multi-step function calls
            for _ in range(5):
                if not response.function_calls:
                    break
                tool_parts = []
                for call in response.function_calls:
                    fn_name = call.name
                    fn_args = dict(call.args) if call.args else {}
                    tool_result = tool_handler(fn_name, fn_args)
                    tool_parts.append(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": tool_result}
                        )
                    )
                response = _execute_or_fail(lambda: chat.send_message(tool_parts))

            final_text = (response.text or "").strip()

            # Save clean user & assistant turns to session memory
            saved_history = []
            if self.memory_file and os.path.exists(self.memory_file):
                try:
                    with open(self.memory_file, "r") as f:
                        saved_history = json.load(f)
                except Exception:
                    pass

            saved_history.append({"role": "user", "content": user_message})
            saved_history.append({"role": "assistant", "content": final_text})

            if self.memory_file:
                os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
                with open(self.memory_file, "w") as f:
                    json.dump(saved_history, f, indent=4)

            return final_text

        except LLMClientError:
            raise
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                raise LLMClientError(f"Limit exceeded: Rate limit reached for '{self.model}'.") from e
            raise LLMClientError(f"Gemini API call (with tools) failed: {e}") from e


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
            err_str = str(e)
            if any(x in err_str.lower() for x in ("rate_limit", "429", "quota", "limit exceeded")):
                raise LLMClientError(f"Limit exceeded: Rate limit reached for Groq model '{self.model}'.") from e
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
            err_str = str(e)
            if any(x in err_str.lower() for x in ("rate_limit", "429", "quota", "limit exceeded")):
                raise LLMClientError(f"Limit exceeded: Rate limit reached for Groq model '{self.model}'.") from e
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


class GeminiModelChainClient:
    """
    Cycles instantly through Gemini models when one hits a rate limit.
    Zero wait time — switches immediately since each model has an independent quota.
    If all models are exhausted, immediately prints 'Limit exceeded' without waiting.
    """

    def __init__(self, models: list[str] = None, memory_file: str = None):
        self._models = list(models or GEMINI_MODEL_CHAIN)
        self._index = 0
        self._clients: dict[str, GeminiLLMClient] = {}
        self.memory_file = memory_file
        # Validate first client
        self._get_client(self._models[0])

    def _get_client(self, model: str) -> GeminiLLMClient:
        if model not in self._clients:
            self._clients[model] = GeminiLLMClient(model=model, memory_file=self.memory_file)
        return self._clients[model]

    @property
    def current_model(self) -> str:
        return self._models[self._index]

    @property
    def model(self) -> str:
        return self.current_model

    def _execute_with_failover(self, method: str, *args, **kwargs) -> str:
        start_index = self._index
        while True:
            model = self._models[self._index]
            client = self._get_client(model)
            try:
                return getattr(client, method)(*args, **kwargs)
            except LLMClientError as e:
                err_lower = str(e).lower()
                if "limit exceeded" in err_lower or "429" in err_lower or "resource_exhausted" in err_lower:
                    next_index = (self._index + 1) % len(self._models)
                    if next_index == start_index:
                        # Full cycle finished - all models reached limit
                        print(f"\n[!] Limit exceeded: All available Gemini models reached their request limits.")
                        raise LLMClientError("Limit exceeded: All Gemini models reached their request limits.") from e
                    print(f"\n[Limit Exceeded on {model}] Switching immediately to {self._models[next_index]} (no wait)...")
                    self._index = next_index
                else:
                    raise
            except Exception as e:
                err_lower = str(e).lower()
                if "429" in err_lower or "resource_exhausted" in err_lower or "quota" in err_lower:
                    next_index = (self._index + 1) % len(self._models)
                    if next_index == start_index:
                        print(f"\n[!] Limit exceeded: All available Gemini models reached their request limits.")
                        raise LLMClientError("Limit exceeded: All Gemini models reached their request limits.") from e
                    print(f"\n[Limit Exceeded on {model}] Switching immediately to {self._models[next_index]} (no wait)...")
                    self._index = next_index
                else:
                    raise

    def ask(self, system: str, user_message: str, context: str = "") -> str:
        return self._execute_with_failover("ask", system, user_message, context)

    def ask_with_tools(self, system: str, user_message: str, tools: list[dict], tool_handler: callable) -> str:
        return self._execute_with_failover("ask_with_tools", system, user_message, tools, tool_handler)


class GroqModelChainClient:
    """
    Cycles through a chain of Groq models when rate limits hit.
    Zero wait time — immediately switches to the next model in the chain.
    """

    def __init__(self, models: list[str] = None, memory_file: str = None):
        self._models = list(models or GROQ_MODEL_CHAIN)
        self._index = 0
        self._clients: dict[str, GroqLLMClient] = {}
        self.memory_file = memory_file
        self._get_client(self._models[0])

    def _get_client(self, model: str) -> GroqLLMClient:
        if model not in self._clients:
            self._clients[model] = GroqLLMClient(model=model, memory_file=self.memory_file)
        return self._clients[model]

    @property
    def current_model(self) -> str:
        return self._models[self._index]

    @property
    def model(self) -> str:
        return self.current_model

    def _execute_with_failover(self, method: str, *args, **kwargs) -> str:
        start_index = self._index
        while True:
            model = self._models[self._index]
            client = self._get_client(model)
            try:
                return getattr(client, method)(*args, **kwargs)
            except Exception as e:
                err_lower = str(e).lower()
                if any(x in err_lower for x in ("rate_limit", "429", "resource_exhausted", "quota", "limit exceeded")):
                    next_index = (self._index + 1) % len(self._models)
                    if next_index == start_index:
                        print(f"\n[!] Limit exceeded: All available Groq models reached their request limits.")
                        raise LLMClientError("Limit exceeded: All Groq models reached their request limits.") from e
                    print(f"\n[Limit Exceeded on Groq/{model}] Switching immediately to {self._models[next_index]} (no wait)...")
                    self._index = next_index
                else:
                    raise

    def ask(self, system: str, user_message: str, context: str = "") -> str:
        return self._execute_with_failover("ask", system, user_message, context)

    def ask_with_tools(self, system: str, user_message: str, tools: list[dict], tool_handler: callable) -> str:
        return self._execute_with_failover("ask_with_tools", system, user_message, tools, tool_handler)


class HierarchicalFailoverClient:
    """
    Tier 1 (Primary): Gemini Model Chain (gemini-3.7-flash -> gemini-3-flash-preview)
    Tier 2 (Fallback): Groq Model Chain (qwen/qwen3.8-27b -> qwen/qwen3.6-27b -> openai/gpt-oss-120b)

    If all Gemini models exhaust their rate limits, instantly fails over to the Groq
    tier without waiting, and cycles through Groq models if needed.
    """

    def __init__(self, gemini_chain: GeminiModelChainClient, groq_chain: GroqModelChainClient):
        self.gemini_chain = gemini_chain
        self.groq_chain = groq_chain
        self._active_tier = "gemini"

    @property
    def model(self) -> str:
        if self._active_tier == "gemini":
            return f"Gemini ({self.gemini_chain.model})"
        return f"Groq ({self.groq_chain.model})"

    def _execute(self, method: str, *args, **kwargs) -> str:
        if self._active_tier == "gemini":
            try:
                return getattr(self.gemini_chain, method)(*args, **kwargs)
            except Exception as e:
                err_lower = str(e).lower()
                if any(x in err_lower for x in ("rate", "429", "resource_exhausted", "quota", "limit exceeded")):
                    print(f"\n[Failover Tier Triggered] All Gemini models exhausted. Switching instantly to Groq tier ({self.groq_chain.model})...")
                    self._active_tier = "groq"
                    return getattr(self.groq_chain, method)(*args, **kwargs)
                raise
        else:
            return getattr(self.groq_chain, method)(*args, **kwargs)

    def ask(self, system: str, user_message: str, context: str = "") -> str:
        return self._execute("ask", system, user_message, context)

    def ask_with_tools(self, system: str, user_message: str, tools: list[dict], tool_handler: callable) -> str:
        return self._execute("ask_with_tools", system, user_message, tools, tool_handler)


def get_llm_client(memory_file: str = None):
    """
    Factory function returning the best configured LLM client:
    1. HierarchicalFailoverClient (Gemini chain -> Groq chain) if both keys present
    2. GeminiModelChainClient if only GEMINI_API_KEY is present
    3. GroqModelChainClient if only GROQ_API_KEY is present
    4. MockLLMClient for testing without credentials
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    gemini_chain = None
    if gemini_key:
        try:
            gemini_chain = GeminiModelChainClient(memory_file=memory_file)
        except Exception as e:
            print(f"[LLM] Gemini chain initialization warning: {e}")

    groq_chain = None
    if groq_key:
        try:
            groq_chain = GroqModelChainClient(memory_file=memory_file)
        except Exception as e:
            print(f"[LLM] Groq chain initialization warning: {e}")

    if gemini_chain and groq_chain:
        print(f"[LLM] Multi-Tier Failover Active: Gemini {gemini_chain._models} -> Groq {groq_chain._models}")
        return HierarchicalFailoverClient(gemini_chain=gemini_chain, groq_chain=groq_chain)
    if gemini_chain:
        return gemini_chain
    if groq_chain:
        return groq_chain
    return MockLLMClient(memory_file=memory_file)

