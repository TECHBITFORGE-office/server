from Provider import *
from flask import request, jsonify, Flask, Response
from functools import wraps
import threading
import time
import uuid
import json
import requests

app = Flask(__name__)

# =======================
# SIMPLE API KEY STORE
# =======================
VALID_API_KEYS = {
    "sk-apinow-tbfgenrated1": {"user": "demo"},
    "sk-apinow-tbfgenratedpro": {"user": "pro"}
}

def verify_api_key(key: str):
    return VALID_API_KEYS.get(key)

# =======================
# AUTH DECORATOR
# =======================
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", "")

        if not auth.startswith("Bearer "):
            return jsonify({
                "error": {
                    "message": "Missing API key",
                    "type": "authentication_error"
                }
            }), 401

        api_key = auth.replace("Bearer ", "").strip()
        key_data = verify_api_key(api_key)

        if not key_data:
            return jsonify({
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error"
                }
            }), 401

        request.api_user = key_data["user"]
        return f(*args, **kwargs)
    return decorated

# =======================
# MODELS ENDPOINT
# =======================
@app.route('/models', methods=['GET'])
def get_models():
    all_models = []
    for _, models in provider_and_models.items():
        all_models.extend(models)
    return jsonify(all_models)

# =======================
# CHAT COMPLETIONS
# =======================
@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    data = request.get_json(silent=True) or {}

    model_name = data.get("model")
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    max_tokens = data.get("max_tokens", 2048)

    try:
        provider = make_workable(model_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    SUPPORTED_PROVIDERS = (Qwen3VL, Qwen3Omni, Coherelabs, gpt_oss_120b)

    if not isinstance(provider, SUPPORTED_PROVIDERS):
        return jsonify({"error": "Provider not supported yet"}), 400

    response = provider.create(
        message=messages,
        model=model_name,
        stream=stream,
        max_tokens=max_tokens
    )

    # =======================
    # STREAM RESPONSE (SSE)
    # =======================
    if stream:
        completion_id = f"gen-{int(time.time())}-{uuid.uuid4().hex[:20]}"
        created = int(time.time())

        def generate():
            # Initial role chunk
            yield f"data: {json.dumps({ 
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model_name,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant'},
                    'finish_reason': None
                }]
            })}\n\n"

            for token in response:
                yield f"data: {json.dumps({
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'model': model_name,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': token},
                        'finish_reason': None
                    }]
                })}\n\n"

            yield f"data: {json.dumps({
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': model_name,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            })}\n\n"

            yield "data: [DONE]\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    # =======================
    # NON-STREAM RESPONSE
    # =======================
    assistant_text = response if isinstance(response, str) else str(response)

    return jsonify({
        "id": f"gen-{int(time.time())}-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
            "completion_tokens": len(assistant_text.split()),
            "total_tokens": sum(len(m.get("content", "").split()) for m in messages) + len(assistant_text.split()),
            "cost": 0
        }
    })

# =======================
# HEALTH CHECK
# =======================
@app.route("/")
def home():
    return "Server is alive"

# =======================
# HF KEEP-ALIVE WORKER
# =======================
SERVERS = ["https://techbitforge-m.hf.space/"]
PING_INTERVAL = 60  # seconds
HEADERS = {"User-Agent": "HF-KeepAlive"}

def background_worker():
    while True:
        print("🔄 Pinging servers...")
        for url in SERVERS:
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                print(f"{url} → {r.status_code}")
            except Exception as e:
                print(f"{url} → ERROR: {e}")
        print("✅ Cycle complete\n")
        time.sleep(PING_INTERVAL)

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    threading.Thread(target=background_worker, daemon=True).start()
    app.run(host="0.0.0.0", port=7860)
