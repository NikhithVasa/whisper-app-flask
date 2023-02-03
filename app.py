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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Item(BaseModel):
    audio_data: str
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[logging.StreamHandler()])


app = FastAPI()
model = whisper.load_model("base.en", device=DEVICE)

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

@app.get('/analyzeTranscript')
def fetchAnswer(question: str):
    logging.info('question is ', question)
    prediction = email_classifier(question)
    logging.info('prediction is ', prediction)
    if prediction == "LABEL_1":
        print("This is a Question")
        return "This is a Question"
    else:
        print("This is not a Question")
        return "This is not a Question"

@app.post("/whisper")
async def whisper(item: Item):

    logging.info(item.audio_data)
    audio_data = item.audio_data
    if not audio_data:
        return JSONResponse(status_code=400, content={"message": "Bad Request"})

    temp = NamedTemporaryFile()
    temp.write(base64.b64decode(audio_data.encode()))
    temp.seek(0)

    result = model.transcribe(temp.name)
    return {'transcript': result['text']}

def email_classifier(text):
    """
    Tokenizes a given sentence and returns the predicted class. 
    
    Returns:
    LABEL_0 --> sentence is predicted as a statement
    LABEL_1 --> sentence is predicted as a question
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")
    model = AutoModelForSequenceClassification.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")

    inputs = tokenizer(f"{text}", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")
