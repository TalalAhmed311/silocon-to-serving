"""rayserve/app.py — a Ray Serve deployment that wraps vLLM's AsyncLLMEngine behind an OpenAI-style completions route.

Why Ray Serve: Python-first composition (pre/post-processing, multi-model graphs, autoscaling by ongoing requests) in
one framework. The vLLM API used here (AsyncEngineArgs / AsyncLLMEngine / SamplingParams) is the long-standing one;
at vLLM 0.31.0 check `vllm/engine/` or `vllm/v1/engine/` for the current async entry point — UNVERIFIED.
Deploy with the RayService manifest (rayservice.yaml) via KubeRay, or locally: `serve run rayserve.app:app` (T2).
"""
from fastapi import FastAPI, Request
from ray import serve

api = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 1}, autoscaling_config={"min_replicas": 1, "max_replicas": 2,
                                                                           "target_ongoing_requests": 32})
@serve.ingress(api)
class LLM:
    def __init__(self, model_path: str = "/models/model"):
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=model_path, max_model_len=8192))

    @api.post("/v1/completions")
    async def complete(self, req: Request):
        import uuid

        from vllm import SamplingParams
        body = await req.json()
        sp = SamplingParams(max_tokens=int(body.get("max_tokens", 64)), temperature=float(body.get("temperature", 0.7)))
        final = None
        async for out in self.engine.generate(body["prompt"], sp, request_id=uuid.uuid4().hex):
            final = out
        text = final.outputs[0].text
        return {"object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": final.outputs[0].finish_reason}],
                "usage": {"prompt_tokens": len(final.prompt_token_ids), "completion_tokens": len(final.outputs[0].token_ids)}}


app = LLM.bind()
