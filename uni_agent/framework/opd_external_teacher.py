"""Use an independently deployed vLLM OpenAI server for k1 OPD scoring."""

import os

import aiohttp

from verl.workers.rollout.replica import TokenOutput


class ExternalTeacherClient:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/") + "/v1/completions"
        self.model = model

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        if sampling_params.get("prompt_logprobs") != 0:
            raise NotImplementedError("External OPD teacher currently supports k1 loss only")
        if any(kwargs.get(key) for key in ("image_data", "video_data", "audio_data")):
            raise NotImplementedError("External OPD teacher currently supports text prompts only")
        payload = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
        }
        timeout = aiohttp.ClientTimeout(total=3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=payload) as response:
                body = await response.json()
                if response.status != 200:
                    raise RuntimeError(f"Teacher request failed: HTTP {response.status}: {body}")
        token_logprobs = body["choices"][0]["logprobs"]["token_logprobs"]
        if len(token_logprobs) < len(prompt_ids):
            raise ValueError(f"Teacher returned {len(token_logprobs)} logprobs for {len(prompt_ids)} prompt tokens")
        # VERL scores each prompt token except the first, then pads the last.
        scores = [[float(value)] for value in token_logprobs[1 : len(prompt_ids)]] + [[0.0]]
        ids = [[int(value)] for value in prompt_ids[1:]] + [[0]]
        return TokenOutput(token_ids=[], extra_fields={"prompt_ids": ids, "prompt_logprobs": scores})


class ExternalTeacherModelManager:
    def __init__(self, config):
        self.url = os.environ["OPD_TEACHER_URL"]
        self.config = config

    def get_client(self):
        return {
            "default": ExternalTeacherClient(
                self.url,
                self.config.distillation.teacher_models.teacher_model.model_path,
            )
        }
