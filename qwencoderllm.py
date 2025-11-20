# qwencoderllm.py
"""
Qwen <> Guardrails <> Pydantic integration (optimized).

- RAIL schema matches the user's requested plain JSON (no "root").
- Uses Guard.from_rail_string and guard.parse (defensive extraction).
- Pydantic final validation.
- Repair loop (LLM asked to fix JSON) up to MAX_VALIDATION_ATTEMPTS.
- Debug prints included to inspect raw and cleaned LLM outputs.
"""

import os
import re
import json
import asyncio
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ValidationError

import httpx

# Guardrails import - ensure guardrails is installed in your venv
from guardrails import Guard

# -----------------------
# Configuration
# -----------------------
SUGGESTION_OPENAI_BASE_URL = os.environ.get("SUGGESTION_OPENAI_BASE_URL", "http://10.0.131.69:8010/v1")
ANALYSIS_OPENAI_BASE_URL = os.environ.get("ANALYSIS_OPENAI_BASE_URL", "http://10.0.131.69:8011/v1")
DEFAULT_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "qwen-7b")
MAX_VALIDATION_ATTEMPTS = int(os.environ.get("MAX_VALIDATION_ATTEMPTS", "3"))
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "7200"))

# -----------------------
# RAIL (no 'root' wrapper) - matches user schema (Option B)
# -----------------------
MICROSERVICE_RAIL = r"""
<rail version="0.1">
  <output>
    <object>
      <list name="microservices">
        <object>
          <string name="name"/>
          <string name="description"/>
          <list name="classes">
            <string/>
          </list>
          <object name="dependencies">
            <list name="internal"><string/></list>
            <list name="external"><string/></list>
          </object>
        </object>
      </list>

      <list name="interconnections">
        <object>
          <string name="source"/>
          <string name="target"/>
          <string name="description"/>
        </object>
      </list>
    </object>
  </output>

  <prompt>
You MUST return ONLY JSON that matches the schema exactly.
No explanation, no markdown, no comments, no extra fields.
Use short 1-sentence placeholders if you can't infer a description.
  </prompt>
</rail>
"""

# -----------------------
# Pydantic models (final assurance)
# -----------------------
class DependencyModel(BaseModel):
    internal: List[str] = Field(default_factory=list)
    external: List[str] = Field(default_factory=list)

class MicroserviceModel(BaseModel):
    name: str
    description: str
    classes: List[str]
    dependencies: DependencyModel

class InterconnectionModel(BaseModel):
    source: str
    target: str
    description: str

class MicroserviceSuggestionModel(BaseModel):
    microservices: List[MicroserviceModel]
    interconnections: List[InterconnectionModel]

# -----------------------
# Utilities: prompt formatting + cleaning
# -----------------------
def format_chat_prompt(messages: List[dict]) -> str:
    prompt = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"
    return prompt

def clean_json_response(response_text: Optional[str]) -> str:
    """Extract JSON-like block, remove fences, fix trailing commas."""
    if not response_text:
        return ""
    text = response_text.strip()
    # Remove triple-backtick fences
    text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text, flags=re.IGNORECASE)
    # Extract first {...} block
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        text = m.group(1)
    # Fix common trailing commas
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text.strip()

# -----------------------
# LLM invocation helper
# -----------------------
async def generate_response(base_url: str, prompt_messages: List[dict], token_size: int = 512, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """
    Call local Qwen server via /completions.
    Returns the raw textual output (no parsing here).
    """
    prompt_text = format_chat_prompt(prompt_messages)
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "max_tokens": token_size,
        "temperature": 0.3,
        "top_p": 0.95,
    }

    last_exc = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT_SECONDS)) as client:
                url = base_url.rstrip("/") + "/completions"
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # Expected shape: {"choices":[{"text":"..."}], ...}
                choices = data.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    if isinstance(first, dict):
                        # many local servers return 'text'
                        text = first.get("text") or (first.get("message") or {}).get("content") or ""
                    else:
                        text = str(first)
                    return (text or "").strip()
                # fallback: return stringified body
                return json.dumps(data)
        except Exception as exc:
            last_exc = exc
            print(f"[generate_response] attempt {attempt + 1} failed: {exc}")
            await asyncio.sleep(1 + attempt * 2)
    raise RuntimeError(f"LLM call failed after retries: {last_exc}")

# -----------------------
# Analysis prompt & wrapper
# -----------------------
def get_prompt_for_analysis(context: str, file_content: str) -> List[dict]:
    system_message = "You are an assistant that migrates Java monolithic web API apps into microservices architecture. Provide a concise analysis (plain text)."
    user_prompt = f"""Context from related files:
{context}

File content:
{file_content}

Please extract (short bullets):
- Role (controller/service/repository)
- Class/interface name
- Short functional summary (1 sentence)
- Internal & external dependencies (class names or libraries)
"""
    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}]

async def analyse_file(context: str, file_content: str) -> str:
    prompt = get_prompt_for_analysis(context, file_content)
    return await generate_response(ANALYSIS_OPENAI_BASE_URL, prompt, token_size=512)

# -----------------------
# Microservice suggestion prompt (exact form required)
# -----------------------
def GetPromptForMicroserviceSuggestion(analysis_summary: str, previous_response: str = "") -> List[dict]:
    system_message = (
        "You are an expert assistant that migrates Java monolithic web API applications into microservices architecture. "
        "Analyze the classes and group them into suitable microservices using Domain-Driven Design (DDD) principles. "
        "Your output must always be a valid JSON object (no text outside JSON)."
    )

    if not previous_response:
        user_prompt = f"""
Below is the analysis of important files in a Java monolithic web API application.
Generate a strictly valid JSON suggestion for microservices. Don't miss any classes/interfaces.
Include test classes under the same related microservice (not a separate one).
If a class does not fit any domain, put it under "SharedServices".

{analysis_summary}

Follow this schema exactly:
{{
  "microservices": [
    {{
      "name": "ServiceName",
      "description": "Description of purpose.",
      "classes": ["package.ClassA", "package.ClassB"],
      "dependencies": {{
        "internal": [],
        "external": []
      }}
    }}
  ],
  "interconnections": [
    {{
      "source": "ServiceA",
      "target": "ServiceB",
      "description": "Interaction description."
    }}
  ]
}}
"""
    else:
        user_prompt = f"""
Here is the previous JSON suggestion:
{previous_response}

Now, merge it with new analysis:
{analysis_summary}

Ensure the final output remains a valid JSON and strictly follows the same schema.
"""
    # Enforce strict rules
    user_prompt += (
        "\n\nRULES:\n"
        "- RETURN ONLY the JSON object that matches the schema above. No markdown or commentary.\n"
        "- No trailing commas. Single-line string values only (no raw newlines inside values).\n"
        "- If a description is unknown, provide a short placeholder sentence.\n"
    )

    return [{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}]

# -----------------------
# Identify microservices: main function with Guardrails + Pydantic validation + repair loop
# -----------------------
async def identify_microservices(combined_summary: str, previous_response: str = "") -> str:
    """
    1. Send strict prompt to LLM asking for exact JSON (per schema).
    2. Clean the raw output and print debug info.
    3. Validate via Guardrails RAIL (no 'root').
    4. If failure, ask LLM to repair the JSON (RAIL-guided), up to MAX_VALIDATION_ATTEMPTS.
    5. Finalize with Pydantic model_validate and return canonical JSON string.
    """
    # Step 1: Ask model
    prompt = GetPromptForMicroserviceSuggestion(combined_summary, previous_response)
    raw = await generate_response(SUGGESTION_OPENAI_BASE_URL, prompt, token_size=10000)

    # Debug: show raw response so you can tweak prompt
    print("\n================ RAW LLM OUTPUT ================\n")
    print(raw)
    print("\n===============================================\n")

    cleaned = clean_json_response(raw)
    print("[identify_microservices] cleaned JSON-like output length:", len(cleaned))
    # quick early parse attempt for visibility
    try:
        preview = json.loads(cleaned)
        print("[identify_microservices] cleaned parsed as JSON (preview keys):", list(preview.keys()))
    except Exception:
        print("[identify_microservices] cleaned is not valid JSON yet (will attempt guard/repair).")

    # Step 2: Guardrails
    guard = Guard.from_rail_string(MICROSERVICE_RAIL)

    last_invalid = cleaned
    for attempt in range(MAX_VALIDATION_ATTEMPTS):
        print(f"[identify_microservices] Guard attempt {attempt + 1}/{MAX_VALIDATION_ATTEMPTS}")
        try:
            validated = guard.parse(cleaned)  # may return dict-like or object
            # Defensive extraction of the actual JSON-like structure from validated
            extracted = None

            if isinstance(validated, dict):
                # Guard may return 'validated_output' or 'output' or direct shape
                extracted = validated.get("validated_output") or validated.get("output") or validated
            else:
                # object-like - try common attributes
                if hasattr(validated, "validated_output"):
                    extracted = getattr(validated, "validated_output")
                elif hasattr(validated, "output"):
                    extracted = getattr(validated, "output")
                else:
                    # last resort: try converting to dict
                    try:
                        extracted = dict(validated)
                    except Exception:
                        extracted = validated

            # After extraction, normalized_json should be a dict or JSON-string
            normalized = extracted

            # If guard returned a wrapper object (some versions), try unwrap once more
            if isinstance(normalized, dict) and "root" in normalized:
                normalized = normalized["root"]

            # If the guard output contains nested structure with same top-level keys, accept it
            # If normalized is a pydantic ValidationOutcome-like object (seen in some guard versions),
            # try to inspect common fields
            if not isinstance(normalized, (dict, list, str)):
                # try to get attribute 'v' or 'value' or 'validated_output'
                for attr in ("v", "value", "validated_output", "output"):
                    if hasattr(normalized, attr):
                        normalized = getattr(normalized, attr)
                        break

            # If still not dict and is iterable, try to convert
            if hasattr(normalized, "__iter__") and not isinstance(normalized, (dict, list, str)):
                try:
                    normalized = dict(normalized)
                except Exception:
                    pass

            # If it's a JSON string, parse it
            if isinstance(normalized, str):
                normalized = json.loads(normalized)

            # Final Pydantic validation (ensures right keys and types)
            MicroserviceSuggestionModel.model_validate(normalized)

            # Canonical JSON (pretty)
            final_json = json.dumps(normalized, indent=2)
            print("[identify_microservices] Validation successful. Returning final JSON.")
            return final_json

        except Exception as e:
            print(f"[identify_microservices] Validation/guard failed on attempt {attempt + 1}: {e}")
            last_invalid = cleaned
            # If last attempt, raise with helpful message
            if attempt >= MAX_VALIDATION_ATTEMPTS - 1:
                raise ValueError(
                    f"Failed to generate valid JSON after {MAX_VALIDATION_ATTEMPTS} attempts.\n"
                    f"Last invalid JSON (truncated):\n{last_invalid[:2000]}\n\nError: {e}"
                )

            # Prepare repair prompt (include RAIL and last invalid JSON)
            repair_prompt = [
                {"role": "system", "content": "You are a JSON fixer assistant. Repair the JSON so it exactly matches the required schema. ONLY return the JSON."},
                {"role": "user", "content": "REQUIRED SCHEMA (RAIL):\n" + MICROSERVICE_RAIL + "\n\nINVALID JSON (fix this):\n" + cleaned},
            ]

            repaired_raw = await generate_response(SUGGESTION_OPENAI_BASE_URL, repair_prompt, token_size=8000)
            print(f"[identify_microservices] Repair attempt {attempt + 1} returned (raw length {len(repaired_raw or '')})")
            # If repair returns the same content, try a stricter repair instruction next iteration
            repaired_cleaned = clean_json_response(repaired_raw)
            if not repaired_cleaned:
                # fallback: produce simple minimal valid empty result to let pipeline continue (optional)
                repaired_cleaned = '{"microservices": [], "interconnections": []}'

            # show repaired preview
            print("[identify_microservices] Repaired cleaned preview (first 500 chars):")
            print(repaired_cleaned[:500])
            # set cleaned to repaired_cleaned to retry validation
            cleaned = repaired_cleaned
            # loop again to validate repaired JSON

    # unreachable
    raise RuntimeError("identify_microservices failed unexpectedly.")


