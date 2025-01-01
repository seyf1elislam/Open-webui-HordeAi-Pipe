"""
title: HodeAI-pipe
author: seyf1elislam
author_url: https://github.com/seyf1elislam
version: 0.1
"""

from pydantic import BaseModel, Field
import requests
from pprint import pp
import time
from typing import List, Dict


class HordeFunctions:
    def __init__(self):
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
            "apikey": "0000000000",
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

    def generate_text(self, prompt, model):
        url = f"{self.base_url}/generate/text/async"
        body = {
            "prompt": prompt,
            "params": {
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
            },
            "models": [model],
            "workers": [],
        }
        response = requests.post(url, headers=self.headers, json=body)
        return response.json()

    def get_status(self, id):
        status_url = f"{self.base_url}/generate/text/status/{id}"
        response = requests.get(status_url, headers=self.headers)
        return response.json()


def format_messages_to_markdown(messages: List[Dict[str, str]]) -> str:
    result = []

    for message in messages:
        result.append(f"## {message['role'].capitalize()}")
        result.append(message["content"])
        result.append("")  # Empty line for spacing

    return "\n".join(result).strip()


class Pipe:
    class Valves(BaseModel):
        # MODEL_ID: str = Field(default="")
        HORDE_KEY: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()
        self.horde = HordeFunctions()
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
        

        try:
            response = self.horde.generate_text(
                format_messages_to_markdown(body["messages"]), model_id
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
