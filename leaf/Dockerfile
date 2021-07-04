FROM python:3.6

COPY . /workspace
WORKDIR /workspace

RUN apt-get update -y

RUN pip install pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PYTHONPATH="/workspace:$PYTHONPATH"
ENV PATH="/workspace:$PATH"

CMD ["/bin/bash"]
