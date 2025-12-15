import requests
import json

class c4ai:
    def __init__(self):
        self.default_model = "command-a"

        self.model_aliases = {
            "command-a": "command-a-03-2025",
            "command-r-plus": "command-r-plus-08-2024",
            "command-r": "command-r-08-2024",
            "command-r7b": "command-r7b-12-2024",
        }

        self.maxtoken = 2048
        self.messages = []
        self.prompt = None
        self.msgid = ""
        self.con_id = None

        self.url = "https://coherelabs-c4ai-command.hf.space"
        self.CONV_URL = f"{self.url}/conversation"
        self.session = requests.Session()

        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Origin": self.url,
            "Referer": self.url + "/",
        }

        # Base system prompt
        self.system_template = """
You must strictly ensure your response never exceeds {MAXTOKENS} tokens.
If a user's request requires more than {MAXTOKENS} tokens, you must compress,
summarize, or shorten the output so the entire response fits within {MAXTOKENS} tokens.
Never exceed this limit, never continue the answer in another message, 
and never ignore this constraint for any reason.
"""

    # ------------------ SYSTEM PROMPT ------------------
    def __add_system_prompt__(self):
        """Extract system prompt from messages list."""
        self.system = self.system_template.replace("{MAXTOKENS}", str(self.maxtoken))

        for i, msg in enumerate(self.messages):
            if msg["role"] == "system":
                self.system += "\nAdditional system rule:\n" + msg["content"]
                del self.messages[i]
                break

    # ------------------ TRANSCRIPT MAKER ------------------
    def __custom_prompt_maker__(self):
        last_messages = self.messages[-10:]

        for msg in last_messages:
            if msg["role"] == "assistent":
                msg["role"] = "assistant"

        self.prompt = "\n".join(f"{m['role']}: {m['content']}" for m in last_messages)

    # ------------------ PAYLOAD BUILDER ------------------
    def __payloads__(self, mode=None):
        if mode == "CONV":
            return {
                "model": self.model_aliases[self.default_model],
                "preprompt": self.system
            }

        return {
            "data": (
                None,
                json.dumps({
                    "inputs": self.prompt,
                    "id": self.msgid,
                    "is_retry": False,
                    "is_continue": False,
                    "web_search": False,
                    "tools": []
                }),
                "application/json"
            )
        }

    # ------------------ GET CONVERSATION ID ------------------
    def __get_conversationId__(self):
        try:
            payload = self.__payloads__("CONV")

            res = self.session.post(
                self.CONV_URL,
                json=payload,
                headers=self.headers
            )
            res.raise_for_status()
            self.con_id = res.json()["conversationId"]

        except Exception as e:
            print("CONV ERROR:", e)

    # ------------------ FETCH FIRST DATA.JSON ------------------
    def __data_json__(self):
        try:
            res = self.session.get(
                f"{self.CONV_URL}/{self.con_id}/__data.json",
                headers=self.headers
            )
            res.raise_for_status()
            first_line = res.text.split("\n")[0]
            data = json.loads(first_line)

            self.msgid = data["nodes"][1]["data"][3]

        except Exception as e:
            print("DATA.JSON ERROR:", e)

    # ------------------ MAIN CHAT STREAM ------------------
    def __chat__(self):
        payload = self.__payloads__()

        try:
            res = self.session.post(
                f"{self.CONV_URL}/{self.con_id}",
                files=payload,
                headers=self.headers,
                stream=True
            )
            res.raise_for_status()

            for line in res.iter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.decode())
                except:
                    continue

                if obj.get("type") == "stream":
                    yield obj["token"].replace("\x00", "")

        except Exception as e:
            print("CHAT ERROR:", e)

    # ------------------ PUBLIC CREATE FUNCTION ------------------
    def create(
            self,
            message: list,
            max_tokens: int = 2048,
            model:str=  "command-a",
            stream: bool = True
            ):
        self.default_model = model
        self.messages = message
        self.maxtoken = max_tokens

        self.__add_system_prompt__()
        self.__custom_prompt_maker__()
        self.__get_conversationId__()
        self.__data_json__()

        if stream:
            return self.__chat__()
        else:
            output = ""
            for chunk in self.__chat__():
                output += chunk
            return output