"""Exercise 3: a tiny JSON API with validation."""
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

app = FastAPI()


class EchoRequest(BaseModel):
    text: str
    repeat: int = 1


@app.post("/v1/echo")
async def echo(request: Request):
    """POST /v1/echo with {"text": "hi", "repeat": 2} -> 200 {"text": "hi hi"}.

    Rules:
      - body is not valid JSON                      -> 400 {"error": "invalid JSON"}
      - JSON but wrong shape/type (pydantic fails)  -> 422 {"error": "invalid body", "detail": [...]}
      - text is empty or only spaces                -> 400 {"error": "text must not be empty"}
      - repeat is not between 1 and 5               -> 422 (use the same 422 shape)
      - otherwise                                   -> 200 {"text": " ".join([text] * repeat)}
    """
    # TODO: read the raw body with `await request.body()`, parse it with json.loads (catch ValueError),
    # validate with EchoRequest.model_validate (catch ValidationError), then apply the rules above.
    raise NotImplementedError
