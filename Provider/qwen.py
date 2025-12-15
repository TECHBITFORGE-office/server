import requests
import uuid
import json

class Qwen3Omni:
    def __init__(self):
        self.url_base = "https://qwen-qwen3-omni-demo.hf.space"
        self.model_aliases = ["Qwen3Omni", "Qwen3Omni-think"]
        self.messages = []
        self.default_model = "Qwen3Omni"
        self.session_hash = str(uuid.uuid4()).replace('-', '')[:10]
        self.model_list = self.model_aliases

        self.prompt = None
        self.system_prompt = None
        self.temperature = 0.6
        self.top_p = 0.95
        self.top_k = 20
        self.maxtoken = 2048

        self.system_template = """
        You must strictly ensure your response never exceeds {MAXTOKENS} tokens.
        If a user's request requires more than {MAXTOKENS} tokens, you must compress,
        summarize, or shorten the output so the entire response fits within {MAXTOKENS} tokens.
        Never exceed this limit, never continue the answer in another message,
        and never ignore this constraint for any reason.
        """

    # -----------------------------------------------------
    def __add_system_prompt__(self):
        """Extract system prompt and remove it from messages."""
        self.system_prompt = self.system_template.replace("{MAXTOKENS}", str(self.maxtoken))

        for i, msg in enumerate(self.messages):
            if msg.get("role") == "system":
                self.system_prompt += "\n" + msg.get("content")
                del self.messages[i]
                break

    # -----------------------------------------------------
    def __prompt_and_messages_gen__(self):
        converted_messages = []

        for msg in self.messages:
            converted_messages.append({
                "role": "assistant" if msg.get("role") == "assistant" else "user",
                "metadata": None,
                "content": msg.get("content"),
                "options": None
            })

        # Extract last user prompt
        last_user_prompt = None
        for i in range(len(converted_messages) - 1, -1, -1):
            if converted_messages[i]["role"] == "user":
                last_user_prompt = converted_messages[i]["content"]
                del converted_messages[i]
                break

        self.prompt = last_user_prompt
        self.messages = converted_messages

    # -----------------------------------------------------
    def __model_alias__(self):
        if self.default_model == "Qwen3Omni":
            self.thinking = False
        elif self.default_model == "Qwen3Omni-think":
            self.thinking = True
        else:
            print("Invalid model. Using base model.")
            self.thinking = False

    # -----------------------------------------------------
    def __join_queue__(self):
        url = f"{self.url_base}/gradio_api/queue/join?"

        payload = {
            "data": [
                self.prompt,
                None,
                None,
                None,
                self.messages,
                self.system_prompt,
                "Cherry / 芊悦",
                self.temperature,
                self.top_p,
                self.top_k,
                False,
                self.thinking
            ],
            "event_data": None,
            "fn_index": 4,
            "trigger_id": 37,
            "session_hash": self.session_hash
        }
        res = requests.post(url, json=payload)
        res.raise_for_status()
        return res.json()

    # -----------------------------------------------------
    def __get_response__(self):
        url = f"{self.url_base}/gradio_api/queue/data?session_hash={self.session_hash}"
        res = requests.get(url, stream=True)
        res.raise_for_status()

        for line in res.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue

            data = json.loads(line[6:])
            msg = data.get("msg")

            if msg == "process_generating":
                outputs = data.get("output", {}).get("data", [])
                if not outputs:
                    continue

                delta_list = outputs[4]   # <<< TOKEN STREAM HERE

                for patch in delta_list:
                    if len(patch) == 3:
                        op, path, value = patch
                        if op == "append" and isinstance(value, str):
                            yield value

                            

    # -----------------------------------------------------
    def create(self, message, model="Qwen3Omni",max_tokens=2000,stream:bool=True):
        self.messages = message
        self.default_model = model
        self.maxtoken = max_tokens

        self.__add_system_prompt__()
        self.__prompt_and_messages_gen__()
        self.__model_alias__()
        self.__join_queue__()

        if stream:
            return self.__get_response__()
        else:
            text=''
            for chunk in self.__get_response__():
                text += chunk
            
            return text



class Qwen3VL:
    def __init__(self):
        self.url_base = "https://qwen-qwen3-vl-demo.hf.space"
        self.join_url = f"{self.url_base}/gradio_api/queue/join?__theme=dark"
        self.stream_url = f"{self.url_base}/gradio_api/queue/data"

        self.model_aliases = ["Qwen3VL"]
        self.model_list = self.model_aliases
        self.default_model = "Qwen3VL"

        self.messages = []
        self.prompt = None
        self.max_tokens = 2048
        self.system_prompt = """
        You must strictly ensure your response never exceeds {MAXTOKENS} tokens.
        If a user's request requires more than {MAXTOKENS} tokens, you must compress,
        summarize, or shorten the output so the entire response fits within {MAXTOKENS} tokens.
        Never exceed this limit, never continue the answer in another message,
        and never ignore this constraint for any reason.
        """

        self.session_hash = str(uuid.uuid4()).replace("-", "")[:10]
        self.session = requests.Session()

    # -----------------------------------------------------
    def __gen_prompt__(self):
        """Generate prompt from messages."""
        prompt = ""
        self.system_prompt = self.system_prompt.replace("{MAXTOKENS}", str(self.max_tokens))

        for msg in self.messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"<|System|>:{content}\n" + self.system_prompt + "\n"
            elif role == "user":
                prompt += f"<|User|>:{content}\n"
            elif role == "assistant":
                prompt += f"<|Assistant|>:{content}\n"

        self.prompt = prompt.strip()

    def __build_payload__(self):
        return {
            "data": [
                {"files": None, "text": self.prompt},
                None,
                None
            ],
            "event_data": None,
            "fn_index": 11,
            "trigger_id": 31,
            "session_hash": self.session_hash
        }

    def __join_queue__(self):
        payload = self.__build_payload__()
        res = self.session.post(self.join_url, json=payload)
        res.raise_for_status()
        return res.json().get("event_id")

    def __listen_stream__(self):
        url = f"{self.stream_url}?session_hash={self.session_hash}"
        res = self.session.get(url, stream=True)
        res.raise_for_status()

        for raw_line in res.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue

            data = raw_line[5:].strip()
            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

    

            if event.get("msg") != "process_generating":
                continue

            # Gradio diff ops
            updates = event["output"]["data"][5]

            for op in updates:
                # ['append', ['value', 1, 'content', 0, 'content'], 'text']
                if op[0] == "append" and isinstance(op[2], str):
                    yield op[2]

    # -----------------------------------------------------
    def create(self, message, model="Qwen3VL", max_tokens=10000000000, stream=True):
        self.default_model = model
        self.messages = message
        self.max_tokens = max_tokens

        self.__gen_prompt__()
        self.__join_queue__()

        if stream:
            return self.__listen_stream__()
        else:
            output = ""
            for chunk in self.__listen_stream__():
                output += chunk
            return output