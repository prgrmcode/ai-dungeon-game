FROM python:3.10-slim

# Add build argument
# ARG HUGGINGFACE_API_KEY
# ENV HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'


# Copy project files
COPY . .

# # Environment variables
ENV PYTHONUNBUFFERED=1
# ENV HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}

# HuggingFace login at container startup
RUN --mount=type=secret,id=HUGGINGFACE_API_KEY \
    HUGGINGFACE_API_KEY=$(cat /run/secrets/HUGGINGFACE_API_KEY) && \
    huggingface-cli login --token $HUGGINGFACE_API_KEY

# RUN mkdir -p ~/.huggingface && \
#     echo "${HUGGINGFACE_API_KEY}" > ~/.huggingface/token && \
#     huggingface-cli login --token ${HUGGINGFACE_API_KEY}

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "main.py"]
