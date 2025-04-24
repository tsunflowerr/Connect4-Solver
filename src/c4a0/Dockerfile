FROM python:3.11-slim

# Cài toolchain + các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    libssl-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài rustup (công cụ cài Rust)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Cài pip, maturin và dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy toàn bộ source
COPY . .

# Build Rust extension (giả sử rust nằm trong thư mục rust/)
WORKDIR /app/rust
RUN maturin develop --release

# Quay về thư mục chính để run app
WORKDIR /app

# Port cho Render
EXPOSE $PORT

CMD uvicorn src.api:app --host 0.0.0.0 --port $PORT
