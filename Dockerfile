FROM python:3.10-slim

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install git -y
RUN pip3 install -r requirements.txt
RUN pip3 install "git+https://github.com/openai/whisper.git" 
RUN apt-get install -y ffmpeg
RUN pip install flask-cors
RUN pip install fastapi
RUN pip install starlette
RUN pip install uvicorn
RUN pip install pydantic

COPY . .

EXPOSE 8000

CMD [ "uvicorn", "app:app", "--host=0.0.0.0", "--port=8000" ]
