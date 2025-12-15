from Provider import *
from flask import request, jsonify, Flask
from functools import wraps
import time
import uuid
import json

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

        # attach user info (optional)
        request.api_user = key_data["user"]

        return f(*args, **kwargs)
    return decorated


# =======================
# MODELS ENDPOINT
# =======================
@app.route('/models', methods=['GET'])
def get_models():
    all_models = []
    for provider_name, models in provider_and_models.items():
        all_models.extend(models)
    return jsonify(all_models)


# =======================
# CHAT COMPLETIONS
# =======================
@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    data = request.json or {}

    model_name = data.get('model')
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    max_tokens = data.get('max_tokens', 2048)

    try:
        provider = make_workable(model_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not isinstance(provider, (Qwen3VL,Qwen3Omni, Coherelabs, gpt_oss_120b)):
        return jsonify({"error": "Provider not supported yet."}), 400

    response = provider.create(
        message=messages,
        model=model_name,
        stream=stream,
        max_tokens=max_tokens
    )

    # =======================
    # STREAM RESPONSE
    # =======================
    if stream:
        completion_id = f"gen-{int(time.time())}-{uuid.uuid4().hex[:20]}"
        created = int(time.time())

        def generate():
            # role chunk
            yield f"data: {json.dumps({
                'id': completion_id,
                'model': model_name,
                'object': 'chat.completion.chunk',
                'created': created,
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant'},
                    'finish_reason': None,
                    'native_finish_reason': None,
                    'logprobs': None
                }]
            })}\n\n"

            # token chunks
            for token in response:
                yield f"data: {json.dumps({
                    'id': completion_id,
                    'model': model_name,
                    'object': 'chat.completion.chunk',
                    'created': created,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': token},
                        'finish_reason': None,
                        'native_finish_reason': None,
                        'logprobs': None
                    }]
                })}\n\n"

            # end chunk
            yield f"data: {json.dumps({
                'id': completion_id,
                'model': model_name,
                'object': 'chat.completion.chunk',
                'created': created,
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop',
                    'native_finish_reason': 'stop',
                    'logprobs': None
                }]
            })}\n\n"

            yield "data: [DONE]\n\n"

        return app.response_class(generate(), mimetype="text/event-stream")


    # =======================
    # NON-STREAM RESPONSE
    # =======================
    assistant_text = response if isinstance(response, str) else str(response)
    completion_id = f"gen-{int(time.time())}-{uuid.uuid4().hex[:20]}"
    created = int(time.time())

    return jsonify({
        "id": completion_id,
        "model": model_name,
        "object": "chat.completion",
        "created": created,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_text,
                "refusal": None,
                "reasoning": None
            },
            "logprobs": None,
            "finish_reason": "stop",
            "native_finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": sum(len(m.get("content","").split()) for m in messages),
            "completion_tokens": len(assistant_text.split()),
            "total_tokens": sum(len(m.get("content","").split()) for m in messages) + len(assistant_text.split()),
            "cost": 0,
            "is_byok": False
        }
    })


# =======================
# RUN SERVER
# =======================
if __name__ == "__main__":
    app.run(host="localhost", port=7860, debug=False)

