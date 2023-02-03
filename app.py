from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
import base64
import logging
import os
from tempfile import NamedTemporaryFile
from pydantic import BaseModel
import whisper
import torch
# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Item(BaseModel):
    audio_data: str


app = FastAPI()
model = whisper.load_model("base", device=DEVICE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

@app.get("/")
async def root():
    logging.basicConfig(level=logging.INFO)
    logging.info('Doing something')
    return {"message": "Hello GWT!"}

@app.post("/whisper")
async def whisper(item: Item):
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler()])

    logging.info(item.audio_data)
    audio_data = item.audio_data
    if not audio_data:
        return JSONResponse(status_code=400, content={"message": "Bad Request"})

    temp = NamedTemporaryFile()
    temp.write(base64.b64decode(audio_data.encode()))
    temp.seek(0)

    result = model.transcribe(temp.name)
    return {'transcript': result['text']}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")
