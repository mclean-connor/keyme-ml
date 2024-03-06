# docker file for training the model with nvidia gpu
FROM tensorflow/tensorflow:2.14.0-gpu

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set the timezone to UTC
ENV TZ=UTC

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    pipx \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# install poetry with pipx
# Ensure pipx bin directory is in PATH
ENV PATH="/root/.local/bin:$PATH"
RUN pipx ensurepath
RUN pipx install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory
WORKDIR /app

# files to app
COPY . /app/

# Install the dependencies. use the cache dir to speed up the build
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

# add /app to python path
ENV PYTHONPATH "${PYTHONPATH}:/app"

# initialize dvc
# RUN run dvc init
