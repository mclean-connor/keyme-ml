# docker file for training the model with nvidia gpu
# use tensorflow 2.14
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# install python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# add poetry to the container
RUN pip3 install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# make all gpu available
ENV NVIDIA_VISIBLE_DEVICES all

# setup prefect
ENV PREFECT_API_KEY=$(PREFECT_API_KEY)
RUN prefect cloud login

# Set the working directory
WORKDIR /app

# files to app
COPY . /app/

# Install the dependencies. use the cache dir to speed up the build
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

# set working directory
WORKDIR /app/keyme_ml

# initialize dvc
RUN dvc init
