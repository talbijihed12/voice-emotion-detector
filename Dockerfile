FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y build-essential portaudio19-dev
RUN apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev alsa-utils
RUN apt-get update && apt-get install -y alsa-utils
RUN pip install -r requirements.txt
EXPOSE 8094
CMD ["python3","-u", "scream.py"]
