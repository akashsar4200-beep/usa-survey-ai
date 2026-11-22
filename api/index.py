import os
import json
import base64
import httpx
from flask import Flask, request, jsonify, send_from_directory

from google import genai

app = Flask(__name__, static_folder="../public", static_url_path="")

# Defaults (mirror the desktop app's defaults)
DEFAULT_GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

DEFAULT_OPENROUTER_MODELS = [
    "nvidia/nemotron-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-opus",
]

DEFAULT_PERSONAS = {
    "US Male - Tech (Texas)": (
        "Name: Michael Carter\n"
        "Age / DOB: 34 (Born: May 14, 1992)\n"
        "Gender: Male\n"
        "Location: Austin, TX 78701\n"
        "Education: Bachelor's Degree in Business Administration\n"
        "Employment: Full-time, IT Project Manager at a mid-sized software company (500-1000 employees)\n"
        "Household Income: $115,000/year | Personal Income: $95,000/year\n"
        "Marital Status: Married, 1 child (Age 4)\n"
        "Car: 2022 Honda Accord\n"
        "Tech & Subscriptions: iPhone 15 Pro, Windows PC, Netflix, Amazon Prime, Spotify, Chase Bank"
    )
}

STRICT_SYSTEM_TEMPLATE = """You are strictly acting as the persona defined below to answer USA market research surveys.

[PERSONA PROFILE]
{persona_details}

[STRICT RULES]
1. CONSISTENCY: Never contradict any detail from the profile above or any previous answer given in this conversation. If a question asks something not explicitly listed in the profile, deduce a logical, realistic detail fitting a US resident of this demographic, and stick to it for the rest of the chat.
2. NO HALLUCINATION & WEB SEARCH: If a survey asks about a specific real-world US brand, recent event, TV show, product, or price you are not 100% certain about, ALWAYS use Web Search to verify facts before answering. Never guess.
3. TRAP / ATTENTION CHECK QUESTIONS: Carefully analyze questions (and screenshots) for attention checks or fake/non-existent brands. Verify via search if unsure, and explicitly warn me if an option is a fake trap.
4. OUTPUT FORMAT:
   - Start with **[EXACT ANSWER TO SELECT]** clearly in bold.
   - Below it, provide a 1-sentence brief reason or any newly invented persona fact I should remember for this survey.
"""


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(default, dict):
                merged = dict(default)
                merged.update(data)
                return merged
            return data
        except Exception:
            pass
    return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except OSError:
        # Vercel read-only filesystem-এ ক্র্যাশ এড়াতে নিরাপদে ইগনোর করা হলো
        pass


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
PERSONAS_FILE = os.path.join(BASE_DIR, "personas.json")


@app.route("/")
def index():
    index_path = os.path.join(BASE_DIR, "public", "index.html")
    if not os.path.exists(index_path):
        return f"File not found at: {index_path}", 404
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/config", methods=["GET"])
def get_config():
    config = load_json(CONFIG_FILE, {
        "provider": "OpenRouter",
        "api_key": "",
        "openrouter_api_key": "",
        "model": DEFAULT_GEMINI_MODELS[0],
        "openrouter_model": DEFAULT_OPENROUTER_MODELS[0],
        "gemini_models": list(DEFAULT_GEMINI_MODELS),
        "openrouter_models": list(DEFAULT_OPENROUTER_MODELS),
        "thinking_level": "medium",
    })
    return jsonify(config)


@app.route("/api/config", methods=["POST"])
def save_config():
    data = request.get_json(force=True) or {}
    config = load_json(CONFIG_FILE, {})
    config.update(data)
    save_json(CONFIG_FILE, config)
    return jsonify({"status": "ok"})


@app.route("/api/personas", methods=["GET"])
def get_personas():
    personas = load_json(PERSONAS_FILE, DEFAULT_PERSONAS)
    return jsonify(personas)


@app.route("/api/personas", methods=["POST"])
def add_persona():
    data = request.get_json(force=True) or {}
    name = data.get("name", "").strip()
    details = data.get("details", "").strip()
    if not name or not details:
        return jsonify({"error": "name and details required"}), 400
    personas = load_json(PERSONAS_FILE, DEFAULT_PERSONAS)
    personas[name] = details
    save_json(PERSONAS_FILE, personas)
    return jsonify({"status": "ok", "personas": personas})


@app.route("/api/personas/<name>", methods=["PUT"])
def edit_persona(name):
    data = request.get_json(force=True) or {}
    new_name = data.get("new_name", name).strip()
    new_details = data.get("details", "").strip()
    if not new_name or not new_details:
        return jsonify({"error": "new_name and details required"}), 400
    personas = load_json(PERSONAS_FILE, DEFAULT_PERSONAS)
    if new_name != name:
        del personas[name]
    personas[new_name] = new_details
    save_json(PERSONAS_FILE, personas)
    return jsonify({"status": "ok", "personas": personas})


@app.route("/api/personas/<name>", methods=["DELETE"])
def delete_persona(name):
    personas = load_json(PERSONAS_FILE, DEFAULT_PERSONAS)
    if len(personas) <= 1:
        return jsonify({"error": "at least one persona must remain"}), 400
    if name in personas:
        del personas[name]
    save_json(PERSONAS_FILE, personas)
    return jsonify({"status": "ok", "personas": personas})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}

    provider = data.get("provider", "OpenRouter")
    api_key = data.get("api_key", "").strip()
    model_name = data.get("model", "").strip()
    system_instruction = data.get("system_instruction", "")
    history = data.get("history", [])
    user_prompt = data.get("user_prompt", "")
    use_search = data.get("use_search", False)
    thinking_level = data.get("thinking_level", "medium")
    image_b64 = data.get("image_b64")
    image_mime = data.get("image_mime", "image/png")

    if not api_key:
        return jsonify({"error": f"Please enter your {provider} API Key first."}), 400
    if not user_prompt and not image_b64:
        return jsonify({"error": "No message or image provided."}), 400

    try:
        # ================= OPENROUTER =================
        if provider == "OpenRouter":
            model_id = model_name.strip()
            if use_search and not model_id.endswith(":online"):
                model_id += ":online"

            messages = [{"role": "system", "content": system_instruction}]
            messages.extend(history)

            if image_b64:
                user_content = [
                    {"type": "text", "text": user_prompt or "Analyze this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                    },
                ]
            else:
                user_content = user_prompt

            messages.append({"role": "user", "content": user_content})

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            if "nemotron" in model_id.lower():
                reasoning_cfg = {"enabled": True}
            else:
                reasoning_cfg = {"effort": thinking_level}

            payload = {
                "model": model_id,
                "messages": messages,
                "reasoning": reasoning_cfg,
            }

            with httpx.Client(timeout=60.0) as http:
                resp = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code != 200:
                    return jsonify({"error": f"OpenRouter Error {resp.status_code}: {resp.text}"}), 502
                resp_data = resp.json()

            msg = resp_data["choices"][0]["message"]
            response_text = msg.get("content") or ""
            reasoning_details = msg.get("reasoning_details")
            used_search = bool(use_search or msg.get("annotations"))

            return jsonify({
                "response_text": response_text,
                "interaction_id": "",
                "used_search": used_search,
                "reasoning_details": reasoning_details,
            })

        # ================= GEMINI =================
        from google.genai import types

        client = genai.Client(api_key=api_key)

        contents = []
        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            contents.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image_mime or "image/png",
                )
            )
            contents.append(user_prompt or "Analyze this image.")
        else:
            contents.append(user_prompt or "Analyze this survey question.")

        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if use_search:
            config_kwargs["tools"] = [{"google_search": {}}]

        if thinking_level:
            try:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=str(thinking_level).upper()
                )
            except Exception:
                pass

        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=model_name.strip(),
            contents=contents,
            config=config,
        )

        response_text = getattr(response, "text", None) or ""
        used_search = False

        for cand in getattr(response, "candidates", None) or []:
            meta = getattr(cand, "grounding_metadata", None)
            if meta and (
                getattr(meta, "web_search_queries", None)
                or getattr(meta, "grounding_chunks", None)
            ):
                used_search = True
                break

        return jsonify({
            "response_text": response_text,
            "interaction_id": "",
            "used_search": used_search,
            "reasoning_details": None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Vercel serverless entry point
handler = app
