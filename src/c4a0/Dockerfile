FROM python:3.11-slim

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    libssl-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install maturin  # Add this line

# Copy source code
COPY . .

# Build Rust extension
WORKDIR /app/rust
RUN maturin develop --release

# Move back to main working directory
WORKDIR /app

# Expose Render port
EXPOSE $PORT

# Start the server
CMD uvicorn src.c4a0.api:app --host 0.0.0.0 --port $PORT
