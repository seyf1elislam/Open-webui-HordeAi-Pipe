"""
title: HordeAI-pipe
author: seyf1elislam
author_url: https://github.com/seyf1elislam
version: 0.2.1
"""

from pydantic import BaseModel, Field
import requests
import time
from typing import List, Dict


class HordeFunctions:
    def __init__(self, api_key="0000000000"):
        self.base_url = "https://aihorde.net/api/v2"
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9,ar-DZ;q=0.8,ar;q=0.7,fr-DZ;q=0.6,fr;q=0.5,en-US;q=0.4",
            "priority": "u=1, i",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "referrer": "https://lite.koboldai.net/",
            "referrerPolicy": "strict-origin-when-cross-origin",
            "apikey": api_key,
        }

    def get_workers(self):
        url = f"{self.base_url}/workers?type=text"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def get_models_list(self):
        data = self.get_workers()
        # models_list = [worker['models'] for worker in data]
        # models_list = [
        #     f"{worker['models']} {worker['max_context_length']}" for worker in data
        # ]
        models_list = [
            f"{worker['models'][0]}-{worker['max_context_length']}" for worker in data
        ]

        return models_list

    def generate_text(self, prompt, model,params=None):
        url = f"{self.base_url}/generate/text/async"
        default_params = {
        "n": 1,
        "max_context_length": 1800,
        "max_length": 200,
        "rep_pen": 1.07,
        "temperature": 0.75,
        "top_p": 0.92,
        "top_k": 100,
        "top_a": 0,
        "typical": 1,
        "tfs": 1,
        "rep_pen_range": 360,
        "rep_pen_slope": 0.7,
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
        "use_default_badwordsids": False,
        "stop_sequence": ["### Instruction:", "### Response:"],
        "min_p": 0,
        "dynatemp_range": 0,
        "dynatemp_exponent": 1,
        "smoothing_factor": 0,
    }
        if params:
            default_params.update(params)
            
        body = {
            "prompt": prompt,
            "params": default_params,
            "models": [model],
            "workers": [],
        }
        response = requests.post(url, headers=self.headers, json=body)
        return response.json()
        

    def get_status(self, id):
        status_url = f"{self.base_url}/generate/text/status/{id}"
        response = requests.get(status_url, headers=self.headers)
        return response.json()


templates = [
     {
        "name": "chatml",
        "user_prefix":"<|im_start|> user",
        "user_suffix":"<|im_end|>",
        "assistant_prefix":"<|im_start|> assistant",
        "assistant_suffix":"<|im_end|>",
        "system_prefix":"<|im_start|> system",
        "system_suffix":"<|im_end|>",
    },
    {
        "name": "alpaca",
        "user_prefix":"## Instruction",
        "user_suffix":"",
        "assistant_prefix":"## Response",
        "assistant_suffix":"",
        "system_prefix":"## System",
        "system_suffix":"",
    },
    {
        "name": "mistral",
        "user_prefix":"[INST]",
        "user_suffix":"",
        "assistant_prefix":"[/INST]",
        "assistant_suffix":"<s>",
        "system_prefix":"[INST]",
        "system_suffix":"[/INST]</s>",
    },
   
]

def format_messages_to_markdown(messages: List[Dict[str, str]],format :str ="chatml" ) -> str:
    result = []
    template = templates[0]
    for t in templates:
        if t["name"] == format:
            template = t
            break

    for message in messages:
        if message['role'].lower() == "user":
            result.append(template["user_prefix"] + "\n" + message["content"] + "\n" + template["user_suffix"])
        elif message['role'].lower() == "system":
            result.append(template["system_prefix"] + "\n" + message["content"] + "\n" + template["system_suffix"])
        elif message['role'].lower() == "assistant":
            result.append(template["assistant_prefix"] + "\n" + message["content"] + "\n" + template["assistant_suffix"])

    return "\n".join(result) #.strip()


class Pipe:
    class Valves(BaseModel):
        # TODO add  constrains for the values
        # MODEL_ID: str = Field(default="")
        HORDE_KEY: str = Field(default="0000000000")
        chat_template: str = Field(default="chatml")
        max_context_length: int = Field(default=4096)
        max_length: int = Field(default=200)
        temperature: float = Field(default=0.75, ge=0, le=2)
        top_p: float = Field(default=0.92)
        top_k: int = Field(default=100)
        rep_pen: float = Field(default=1.07)
        rep_pen_range: int = Field(default=360)
        rep_pen_slope: float = Field(default=0.7)
        stop_sequence: List[str] = Field(default=["### Instruction:", "### Response:","<|im_start|>","<|im_end|>","[INST]","[/INST]"])




    def __init__(self):
        self.valves = self.Valves()
        self.horde = HordeFunctions(api_key=self.valves.HORDE_KEY)
        self.models_list = self.horde.get_models_list()

    def pipes(self):
        return [{"id": model, "name": model} for model in self.models_list]

    def pipe(self, body: dict, __user__: dict):
        print("-" * 50)
        print(f"pipe:{__name__}")
        print("-" * 50)
        # Extract model id from the model name
        model_id = body["model"][body["model"].find(".") + 1 :]
        model_id = "-".join(model_id.rsplit("-", 1)[:-1])
        print(f"Model id : {model_id}")
        

        # Get parameters from valves
        params = {
            "max_context_length": self.valves.max_context_length,
            "max_length": self.valves.max_length,
            "temperature": self.valves.temperature,
            "top_p": self.valves.top_p,
            "top_k": self.valves.top_k,
            "rep_pen": self.valves.rep_pen,
            "rep_pen_range": self.valves.rep_pen_range,
            "rep_pen_slope": self.valves.rep_pen_slope,
            "stop_sequence": self.valves.stop_sequence,
        }

        try:
            response = self.horde.generate_text(
                format_messages_to_markdown(body["messages"],
                                            self.valves.chat_template), 
                model_id,
                params
            )
            task_id = response["id"]

            while True:
                status = self.horde.get_status(task_id)
                if not status['is_possible']:
                    return "Error: Model not available"
                elif status["done"] or status['finished']:
                    result = status["generations"][0]["text"]
                    return result
                    break
                else:
                    print("Waiting for generation to complete...")
                    time.sleep(5)  # Wait for 5 seconds before checking the status again

        except Exception as e:
            return f"Error: {e} with : {model_id}"
