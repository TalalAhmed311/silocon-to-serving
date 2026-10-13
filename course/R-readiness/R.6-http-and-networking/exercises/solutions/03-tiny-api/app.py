"""Exercise 3 (solution): a tiny JSON API with validation."""
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

app = FastAPI()


class EchoRequest(BaseModel):
    text: str
    repeat: int = Field(default=1, ge=1, le=5)


@app.post("/v1/echo")
async def echo(request: Request):
    try:
        payload = json.loads(await request.body())
    except ValueError:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    try:
        req = EchoRequest.model_validate(payload)
    except ValidationError as e:
        return JSONResponse({"error": "invalid body", "detail": json.loads(e.json())}, status_code=422)
    if not req.text.strip():
        return JSONResponse({"error": "text must not be empty"}, status_code=400)
    return {"text": " ".join([req.text] * req.repeat)}
