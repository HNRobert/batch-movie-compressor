#!/usr/bin/env python3
"""
Movie Compression Configuration
"""

# Compression settings
DEFAULT_TARGET_SIZE_GB = 12.0  # Target file size (GB)
DEFAULT_AUDIO_BITRATE = 640    # Audio bitrate (kbps) for 7.1 audio

# AV1 encoding settings
AV1_ENCODERS = {
    'libsvtav1': {
        'preset': '6',  # 0-12, lower is better quality but slower
        'params': 'tune=0:film-grain=8:enable-overlays=1:scd=1:scm=0'
    },
    'libaom-av1': {
        'cpu_used': '4',  # 0-8, lower is better quality but slower
        'params': 'tune=ssim:enable-fwd-kf=1:kf-max-dist=240'
    }
}

# HDR settings
HDR_SETTINGS = {
    'color_primaries': '9',     # BT.2020
    'color_trc': '16',          # PQ (Perceptual Quantizer)
    'colorspace': '9',          # BT.2020nc
    'pixel_format': 'yuv420p10le'  # 10-bit
}

# Supported video formats
SUPPORTED_FORMATS = {'.mkv', '.mp4', '.avi', '.mov', '.m4v', '.ts', '.m2ts'}

# Audio encoding settings
AUDIO_ENCODING = {
    'multichannel': {  # 5.1 and above
        'codec': 'eac3',
        'bitrate': '640k'
    },
    'stereo': {  # Stereo
        'codec': 'aac',
        'bitrate': '192k'
    }
}

# FFmpeg general parameters
FFMPEG_GENERAL_PARAMS = {
    'gop_size': 240,           # GOP size, suitable for 4K
    'max_rate_multiplier': 1.5,  # Maximum bitrate multiplier
    'buffer_multiplier': 2,     # Buffer multiplier
}

# File naming template
OUTPUT_FILENAME_TEMPLATE = "{stem}_AV1_4K_HDR10.mkv"

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
