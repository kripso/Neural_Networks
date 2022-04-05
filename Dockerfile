FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt 
ENTRYPOINT ["/bin/bash"]
# CMD ["/bin/bash"]