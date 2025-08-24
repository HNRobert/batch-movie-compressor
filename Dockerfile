# Use lightweight Alpine image with FFmpeg
FROM jrottenberg/ffmpeg:6.1-alpine

# Install Python
RUN apk add --no-cache python3 py3-pip

# Create directories
WORKDIR /app
RUN mkdir -p /data/in /data/out

# Copy application
COPY main.py /app/

# Set permissions
RUN chmod +x /app/main.py

# Create non-root user
RUN adduser -D -u 1000 compressor && \
    chown -R compressor:compressor /app /data
USER compressor

# Default command
ENTRYPOINT ["python3", "main.py"]
CMD ["/data/in", "/data/out"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ffmpeg -version > /dev/null 2>&1 || exit 1

# Labels
LABEL maintainer="Movie Compressor" \
      description="Lightweight batch movie compressor with AV1 encoding" \
      version="2.0"