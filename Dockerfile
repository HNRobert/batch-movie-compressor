# Use Python Alpine base with GPU acceleration support
FROM python:3.11-alpine

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies including FFmpeg with GPU support
RUN apk add --no-cache \
    ffmpeg \
    ffmpeg-dev \
    # GPU acceleration libraries for Intel/AMD
    libva \
    libva-utils \
    libdrm \
    libdrm-dev \
    mesa-va-gallium \
    mesa-vdpau-gallium \
    # Additional libraries that may be available
    mesa-dri-gallium 

# Create working directory
WORKDIR /app

# Create data directories
RUN mkdir -p /data/in /data/out

# Create a non-root user
RUN adduser -D -s /bin/sh compressor && \
    chown -R compressor:compressor /data /app

# Copy the main script
COPY main.py /app/

# Set non-root user
USER compressor

# Set the working directory
WORKDIR /app
