# Use Python Alpine base and install FFmpeg
FROM python:3.11-alpine

# Install FFmpeg and dependencies
RUN apk add --no-cache \
    ffmpeg \
    ffmpeg-dev

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
