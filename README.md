# Batch Movie Compressor

A lightweight AV1-based 4K HDR movie batch compression tool that compresses ~30GB movies to ~12GB while maintaining excellent visual quality.

## Features

- 🎬 **Batch Processing**: Automatically process all movie files in a folder
- 🚀 **AV1 Encoding**: Uses latest AV1 encoders (SVT-AV1 or libaom-av1)
- 🌈 **HDR Support**: Complete HDR10 metadata and color information preservation
- 📏 **Precise Size Control**: Smart bitrate calculation to achieve target file size
- 🔊 **Audio Optimization**: Automatic best audio encoding selection (EAC-3 for 5.1+, AAC for stereo)
- 🐳 **Containerized**: Docker ensures environment consistency
- 📊 **Progress Monitoring**: Real-time compression progress and statistics
- ⚡ **Lightweight**: Uses Alpine-based FFmpeg image for minimal footprint
- 📁 **Separate I/O**: Clean separation of input and output directories

## Supported Formats

**Input formats**: MKV, MP4, AVI, MOV, M4V, TS, M2TS  
**Output format**: MKV (recommended for HDR content)

## Directory Structure

The container uses separate input and output directories:

```text
/data/in/        # Input directory (mounted from host INPUT_DIR)
├── Movie1.mkv
├── Movie2.mp4
└── ...

/data/out/       # Output directory (mounted from host OUTPUT_DIR)
├── Movie1_AV1_4K_HDR10.mkv
├── Movie2_AV1_4K_HDR10.mkv
├── compression.log
└── ...
```

## Usage

### Command Line Options

```bash
python3 main.py <input_dir> <output_dir> [options]

Options:
  --target-size FLOAT  Target file size in GB (default: 12.0)
  --dry-run           Dry run mode - only show files to be processed
  -h, --help          Show help information
```

### Docker Usage

```bash
# Build image
docker-compose build

# Run compression task
docker-compose up -d

# View logs
docker-compose logs -f batch-movie-compressor
```

## Compression Settings

### AV1 Encoder Selection

The script automatically detects available AV1 encoders:

1. **SVT-AV1** (Recommended)

   - Faster encoding speed
   - Good quality/speed balance
   - Suitable for batch processing

2. **libaom-av1**
   - Higher compression efficiency
   - Slower encoding speed
   - Slightly better quality

### Quality Settings

- **Target size**: Default 12GB, adjustable
- **Video encoding**: AV1, 10-bit HDR
- **Audio encoding**:
  - Multi-channel (5.1+): EAC-3 @ 640kbps
  - Stereo: AAC @ 192kbps

### HDR Settings

- **Color space**: BT.2020
- **Transfer function**: PQ (HDR10)
- **Pixel format**: YUV420P10LE (10-bit)

## Performance Optimization

### Hardware Requirements

- **CPU**: 8+ cores recommended (AV1 encoding requires significant CPU resources)
- **Memory**: 16GB+ recommended
- **Storage**: Sufficient space for original and compressed files

### Docker Resource Configuration

Adjust resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0' # Adjust based on your CPU cores
      memory: 16G # Adjust based on your memory
```

### Encoding Speed Adjustment

Adjust encoding presets in `config.py`:

```python
# SVT-AV1: 0-12 (0=slowest best quality, 12=fastest lowest quality)
'preset': '6'  # Default balanced setting

# libaom-av1: 0-8 (0=slowest best quality, 8=fastest lowest quality)
'cpu_used': '4'  # Default balanced setting
```

## Example Output

```text
2024-08-25 15:30:00 - INFO - Starting batch movie compression
2024-08-25 15:30:00 - INFO - Found 8 video files
2024-08-25 15:30:01 - INFO - Processing 1/8: Mission.Impossible.The.Final.Reckoning.2025.mkv
2024-08-25 15:30:01 - INFO - Target bitrate: 8500kbps, duration: 8820.50s
2024-08-25 18:45:23 - INFO - Compression completed: Mission.Impossible.The.Final.Reckoning.2025.mkv
2024-08-25 18:45:23 - INFO - Output size: 11.85GB
2024-08-25 18:45:23 - INFO - Compression ratio: 2.54:1
2024-08-25 18:45:23 - INFO - Time taken: 3.26h
```

## Troubleshooting

### Common Issues

1. **AV1 encoder not available**

   ```bash
   # Check encoder support
   ffmpeg -encoders | grep av1
   ```

2. **Permission errors**

   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER /path/to/directories
   ```

3. **Insufficient memory**

   - Reduce concurrency or decrease Docker memory limits
   - Use faster encoding presets

4. **Insufficient disk space**
   - Ensure output directory has enough space
   - Consider compressing and deleting originals gradually (use caution)

### Log Files

- Container logs: `docker-compose logs batch-movie-compressor`
- Application logs: `/data/out/compression.log`
- FFmpeg detailed logs: Set `FFMPEG_LOG_LEVEL=debug`

## Contributing

Feel free to submit Issues and Pull Requests to improve this project.

## License

This project is licensed under the MIT License.

## Technical Details

### Compression Strategy

1. **Bitrate calculation**: Dynamic calculation based on target file size and video duration
2. **Constrained bitrate**: Uses constrained bitrate mode to ensure file size
3. **HDR preservation**: Complete HDR metadata passthrough
4. **Audio processing**: Select best encoding based on channel count

### File Structure

```text
batch-movie-compressor/
├── main.py              # Main program
├── config.py           # Configuration file
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose configuration
├── .env.example        # Environment variables example
├── README.md           # Documentation
├── LICENSE             # License
└── data/               # Example data directories
    ├── in/             # Input directory (put original movies here)
    ├── out/            # Output directory (compressed movies)
    └── README.md       # Data directory documentation
```

## Changelog

### v2.0.0

- Simplified Docker image using Alpine FFmpeg
- Separated input and output directories (/data/in and /data/out)
- English documentation and comments
- Lightweight container design
- Improved user experience

### v1.0.0

- Initial release
- AV1 encoding support
- HDR10 support
- Batch processing functionality
- Docker containerization
- HDR10 支持
- 批量处理功能
- Docker 容器化
