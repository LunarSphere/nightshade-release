#use ubuntu lastest version as base image
FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

#set working directory
WORKDIR /app

#copy all files from current directory to working directory in container
COPY . /app

RUN mkdir -p /Data

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python3.10 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir pip setuptools

RUN pip install --upgrade pip

#install dependencies
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

RUN apt-get update && apt-get install -y \
    curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

#Download raw data from S3 bucket wont work because credentials are run stage
# RUN aws s3 sync s3://memoryscapes-media-dev/uploads/raw/ ./repos/Data --no-sign-request

