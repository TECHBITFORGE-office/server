import requests
import uuid
import json

class gpt_oss_120b():
    def __init__(self):
        self.BASE = "https://amd-gpt-oss-120b-chatbot.hf.space"
        self.session_hash = str(uuid.uuid4()).replace("-", "")[:10]
        self.session = requests.Session()
        self.model_aliases = ["gpt-oss-120b","gpt-oss-20b"]
        self.default_model = "gpt-oss-120b"
        self.messages = []
        self.max_tokens = 2048
        self.prompt = None
        self.system_prompt = """
        You must strictly ensure your response never exceeds {MAXTOKENS} tokens.
        If a user's request requires more than {MAXTOKENS} tokens, you must compress,
        summarize, or shorten the output so the entire response fits within {MAXTOKENS} tokens.
        Never exceed this limit, never continue the answer in another message,
        and never ignore this constraint for any reason.
        """
    # -----------------------------------------------------
    def __gen_prompt__(self):
        """Generate prompt from messages."""
        prompt = ""
        system_prompt = ""
        self.system_prompt = self.system_prompt.replace("{MAXTOKENS}", str(self.max_tokens))
        for msg in self.messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_prompt += f"System: {content}\n" + self.system_prompt + "\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt = self.system_prompt + "\n" + prompt + "Assistant: "
        self.prompt = prompt
        self.system_prompt = system_prompt
    # -----------------------------------------------------
    def __send_user_message__(self):
        """Send user message to the model."""
        payload = {
            "data": [self.prompt],
            "event_data": None,
            "fn_index": 2,
            "trigger_id": 13,
            "session_hash": self.session_hash
        }
        self.session.post(
            f"{self.BASE}/gradio_api/run/predict?__theme=dark",
            json=payload
        )
    # -----------------------------------------------------
    def __push_chat_state__(self):
        """Push chat state to the model."""
       
        self.session.post(
            f"{self.BASE}/gradio_api/run/predict?__theme=dark",
            json={
                "data": [None, []],
                "event_data": None,
                "fn_index": 3,
                "trigger_id": 13,
                "session_hash": self.session_hash
            }
)
    # -----------------------------------------------------
    def __join_queue__(self):
        """Join the inference queue."""
        payload = {
            "data": [None, None, self.system_prompt, 0.7],
            "event_data": None,
            "fn_index": 4,
            "trigger_id": 13,
            "session_hash": self.session_hash
        }
        res = self.session.post(
            f"{self.BASE}/gradio_api/queue/join?__theme=dark",
            json=payload
        )
        res.raise_for_status()
        return res.json().get("event_id")
    
    # -----------------------------------------------------
    def __stream_response__(self):
        """Stream response from the model."""
        url = f"{self.BASE}/gradio_api/queue/data?session_hash={self.session_hash}"
        res = self.session.get(url, stream=True)
        res.raise_for_status()
        
        for line in res.iter_lines():
            if not line:
                continue

            if not line.startswith(b"data:"):
                continue

            payload = json.loads(line[5:])

            # stop condition
            if payload.get("msg") == "process_completed":
                break

            if payload.get("msg") != "process_generating":
                continue

            output = payload.get("output", {})
            data = output.get("data", [])

            if not data or not data[1]:
                continue

            # Gradio diff format:
            # ['append', [1, 'content'], 'text']
            for diff in data[1]:
                if (
                    isinstance(diff, list)
                    and diff[0] == "append"
                    and diff[1] == [1, "content"]
                ):
                    token = diff[2]
                    yield token
    # -----------------------------------------------------
    def create(
            self,
            message: list,
            max_tokens: int = 2048,
            model: str = "gpt-oss-120b",
            stream: bool = True
        ):
        self.default_model = model
        self.messages = message
        self.max_tokens = max_tokens

        self.__gen_prompt__()
        self.__send_user_message__()
        self.__push_chat_state__()
        self.__join_queue__()

        if stream:
            return self.__stream_response__()
        
        # NON-STREAMING RESPONSE
        output = ""
        for token in self.__stream_response__():
            output += token
        return output