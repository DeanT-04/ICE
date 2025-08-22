# Multi-stage Dockerfile for Ultra-Fast AI Model
# Supports Rust, Zig, and Go compilation with optimized distroless runtime

# =============================================================================
# Stage 1: Rust Builder
# =============================================================================
FROM rust:1.80-slim as rust-builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Rust project files
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# Create cache layer for dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --bin ultra-fast-ai
RUN rm -rf src/

# Copy actual source and build
COPY src/ src/
RUN cargo build --release --bin ultra-fast-ai --bin train

# =============================================================================
# Stage 2: Zig Builder (for performance kernels)
# =============================================================================
FROM alpine:3.18 as zig-builder

# Install Zig compiler
RUN apk add --no-cache \
    curl \
    xz \
    && curl -L https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz \
    | tar -xJ -C /opt/ \
    && ln -s /opt/zig-linux-x86_64-0.13.0/zig /usr/local/bin/zig

WORKDIR /app

# Copy Zig build configuration
COPY build.zig ./
COPY src/kernels/ src/kernels/

# Build Zig kernels (when implemented)
# RUN zig build lib -Doptimize=ReleaseFast

# =============================================================================
# Stage 3: Go Builder (for MCP interfaces)
# =============================================================================
FROM golang:1.23-alpine as go-builder

# Install dependencies
RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy Go module files
COPY go.mod go.sum ./
RUN go mod download

# Copy Go source (when implemented)
COPY src/mcp/ src/mcp/

# Build Go MCP services (when implemented)
# RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o mcp-server src/mcp/main.go

# =============================================================================
# Stage 4: Model Assets and Data
# =============================================================================
FROM alpine:3.18 as assets

WORKDIR /app

# Create model and cache directories
RUN mkdir -p \
    models/ \
    .cache/mcp/ \
    __fixtures__/datasets/ \
    __fixtures__/mcp/

# Copy configuration files
COPY config.toml ./

# Create default model directory structure
RUN touch models/.gitkeep \
    && touch .cache/mcp/.gitkeep \
    && touch __fixtures__/datasets/.gitkeep \
    && touch __fixtures__/mcp/.gitkeep

# =============================================================================
# Stage 5: Final Runtime (Distroless)
# =============================================================================
FROM gcr.io/distroless/cc-debian12:latest

# Metadata
LABEL maintainer=\"AI Model Team\" \
      version=\"0.1.0\" \
      description=\"Ultra-Fast AI Model with SNN-SSM-Liquid NN architecture\" \
      org.opencontainers.image.source=\"https://github.com/user/ultra-fast-ai\" \
      org.opencontainers.image.title=\"Ultra-Fast AI Model\" \
      org.opencontainers.image.description=\"Hyper-efficient AI model targeting <100M params, <100ms inference, <50W power\"

# Create non-root user (UID 1000 as specified in requirements)
USER 1000:1000

WORKDIR /app

# Copy compiled binaries from Rust builder
COPY --from=rust-builder --chown=1000:1000 /app/target/release/ultra-fast-ai /usr/local/bin/ultra-fast-ai
COPY --from=rust-builder --chown=1000:1000 /app/target/release/train /usr/local/bin/train

# Copy Zig libraries (when available)
# COPY --from=zig-builder --chown=1000:1000 /app/zig-out/lib/ /usr/local/lib/

# Copy Go MCP services (when available)
# COPY --from=go-builder --chown=1000:1000 /app/mcp-server /usr/local/bin/mcp-server

# Copy model assets and configuration
COPY --from=assets --chown=1000:1000 /app/ ./

# Set environment variables
ENV RUST_LOG=info \
    RUST_BACKTRACE=1 \
    MODEL_PATH=/app/models \
    CACHE_PATH=/app/.cache \
    CONFIG_PATH=/app/config.toml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [\"/usr/local/bin/ultra-fast-ai\", \"--help\"] || exit 1

# Expose ports for MCP servers
EXPOSE 8001 8002 8003 8004

# Default command
ENTRYPOINT [\"/usr/local/bin/ultra-fast-ai\"]
CMD [\"--help\"]

# =============================================================================
# Build Instructions:
# 
# Development build:
#   docker build -t ultra-fast-ai:dev .
#
# Production build with multi-platform:
#   docker buildx build --platform linux/amd64,linux/arm64 -t ultra-fast-ai:latest .
#
# Usage:
#   # Run inference
#   docker run --rm ultra-fast-ai:latest infer --input \"Hello world\"
#   
#   # Run training (with GPU support)
#   docker run --rm --gpus all -v $(pwd)/models:/app/models ultra-fast-ai:latest train
#   
#   # Interactive shell for debugging
#   docker run --rm -it --entrypoint /bin/sh ultra-fast-ai:latest
#
# Performance considerations:
#   - Uses distroless base image for minimal attack surface and size
#   - Multi-stage build reduces final image size by ~90%
#   - Non-root user (1000:1000) for security
#   - Cached dependency layers for faster rebuilds
#   - Optimized Rust release builds with LTO enabled
# =============================================================================