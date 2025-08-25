#!/usr/bin/env python3
"""
Batch Movie Compressor - AV1 encoding for 4K HDR movies
Compress ~30GB movies to ~12GB while maintaining 4K HDR10 quality
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/out/compression.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MovieCompressor:
    def __init__(self, input_dir: str, output_dir: str, target_size_gb: float = 12.0, recursive: bool = True, enable_gpu: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size_gb = target_size_gb
        self.target_size_mb = target_size_gb * 1024
        self.recursive = recursive
        self.enable_gpu = enable_gpu

        # GPU acceleration capabilities
        self.gpu_info = {
            'nvidia': False,
            'intel': False,
            'available_encoders': [],
            'preferred_encoder': None
        }

        # Supported video formats
        self.supported_formats = {'.mkv', '.mp4',
                                  '.avi', '.mov', '.m4v', '.ts', '.m2ts'}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if FFmpeg is available and detect GPU capabilities
        self._check_ffmpeg()
        if self.enable_gpu:
            self._detect_gpu_capabilities()

    def _check_ffmpeg(self):
        """Check if FFmpeg is installed and supports AV1"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("FFmpeg not found")

            # Check AV1 encoder support
            encoders_result = subprocess.run(
                ['ffmpeg', '-encoders'], capture_output=True, text=True)
            if 'libsvtav1' not in encoders_result.stdout and 'libaom-av1' not in encoders_result.stdout:
                logger.warning(
                    "AV1 encoders might not be available, will try to use available encoders")

            logger.info("FFmpeg check passed")
        except Exception as e:
            logger.error(f"FFmpeg check failed: {e}")
            sys.exit(1)

    def _detect_gpu_capabilities(self):
        """Detect available GPU acceleration capabilities"""
        logger.info("Detecting GPU acceleration capabilities...")

        # Check for NVIDIA GPU support
        try:
            # Test NVIDIA encoders
            nvidia_encoders = ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc']
            for encoder in nvidia_encoders:
                result = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'testsrc2=duration=0.1:size=64x64:rate=1',
                     '-c:v', encoder, '-f', 'null', '-'],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    self.gpu_info['nvidia'] = True
                    self.gpu_info['available_encoders'].append(encoder)
                    logger.info(f"NVIDIA encoder available: {encoder}")
        except Exception as e:
            logger.debug(f"NVIDIA GPU test failed: {e}")

        # Check for Intel GPU support (VAAPI)
        try:
            # Test Intel/VAAPI encoders
            intel_encoders = ['h264_vaapi', 'hevc_vaapi', 'av1_vaapi']
            for encoder in intel_encoders:
                result = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-vaapi_device', '/dev/dri/renderD128',
                     '-f', 'lavfi', '-i', 'testsrc2=duration=0.1:size=64x64:rate=1',
                     '-vf', 'format=nv12,hwupload', '-c:v', encoder, '-f', 'null', '-'],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    self.gpu_info['intel'] = True
                    self.gpu_info['available_encoders'].append(encoder)
                    logger.info(f"Intel VAAPI encoder available: {encoder}")
        except Exception as e:
            logger.debug(f"Intel VAAPI test failed: {e}")

        # Determine preferred encoder
        if 'av1_nvenc' in self.gpu_info['available_encoders']:
            self.gpu_info['preferred_encoder'] = 'av1_nvenc'
            logger.info("Using NVIDIA AV1 hardware encoder")
        elif 'av1_vaapi' in self.gpu_info['available_encoders']:
            self.gpu_info['preferred_encoder'] = 'av1_vaapi'
            logger.info("Using Intel VAAPI AV1 hardware encoder")
        elif 'hevc_nvenc' in self.gpu_info['available_encoders']:
            self.gpu_info['preferred_encoder'] = 'hevc_nvenc'
            logger.info(
                "Using NVIDIA HEVC hardware encoder (AV1 not available)")
        elif 'hevc_vaapi' in self.gpu_info['available_encoders']:
            self.gpu_info['preferred_encoder'] = 'hevc_vaapi'
            logger.info(
                "Using Intel VAAPI HEVC hardware encoder (AV1 not available)")
        else:
            logger.info(
                "No hardware encoders available, will use software encoding")

    def get_video_info(self, video_path: Path) -> Dict:
        """Get video information using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get video info: {video_path}, error: {e}")
            return {}

    def calculate_bitrate(self, duration: float, target_size_mb: float, audio_bitrate_kbps: int = 640) -> int:
        """Calculate target video bitrate"""
        # Reserve space for audio (640kbps for 7.1 audio)
        audio_size_mb = (audio_bitrate_kbps * duration) / (8 * 1024)
        video_size_mb = target_size_mb - audio_size_mb - 100  # Reserve 100MB for safety

        if video_size_mb <= 0:
            logger.warning("Target size too small, using minimum bitrate")
            return 1000

        video_bitrate_kbps = int((video_size_mb * 8 * 1024) / duration)
        return max(video_bitrate_kbps, 1000)  # Minimum 1Mbps

    def get_best_encoder(self) -> tuple[str, dict]:
        """Get the best available encoder (GPU preferred, then CPU)"""
        encoder_config = {
            'encoder': None,
            'hw_device': None,
            'input_filters': [],
            'encoder_params': {},
            'output_format': 'mkv'
        }

        # Try GPU encoders first if enabled
        if self.enable_gpu and self.gpu_info['preferred_encoder']:
            preferred = self.gpu_info['preferred_encoder']

            if preferred == 'av1_nvenc':
                encoder_config.update({
                    'encoder': 'av1_nvenc',
                    'encoder_params': {
                        'preset': 'p4',  # Medium preset for balance
                        'tune': 'hq',    # High quality
                        'rc': 'vbr',     # Variable bitrate
                    }
                })
                logger.info("Using NVIDIA AV1 hardware encoder")
                return preferred, encoder_config

            elif preferred == 'av1_vaapi':
                encoder_config.update({
                    'encoder': 'av1_vaapi',
                    'hw_device': '/dev/dri/renderD128',
                    'input_filters': ['format=nv12', 'hwupload'],
                    'encoder_params': {
                        'quality': '25',  # Good quality
                    }
                })
                logger.info("Using Intel VAAPI AV1 hardware encoder")
                return preferred, encoder_config

            elif preferred == 'hevc_nvenc':
                encoder_config.update({
                    'encoder': 'hevc_nvenc',
                    'encoder_params': {
                        'preset': 'medium',
                        'tune': 'hq',
                        'rc': 'vbr',
                    },
                    'output_format': 'mp4'  # HEVC works better with MP4
                })
                logger.info("Using NVIDIA HEVC hardware encoder (fallback)")
                return preferred, encoder_config

            elif preferred == 'hevc_vaapi':
                encoder_config.update({
                    'encoder': 'hevc_vaapi',
                    'hw_device': '/dev/dri/renderD128',
                    'input_filters': ['format=nv12', 'hwupload'],
                    'encoder_params': {
                        'quality': '25',
                    },
                    'output_format': 'mp4'
                })
                logger.info(
                    "Using Intel VAAPI HEVC hardware encoder (fallback)")
                return preferred, encoder_config

        # Fallback to software AV1 encoders
        software_encoders = ['libsvtav1', 'libaom-av1']
        for encoder in software_encoders:
            try:
                result = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'testsrc2=duration=1:size=320x240:rate=1', '-c:v', encoder, '-f', 'null', '-'],
                                        capture_output=True, timeout=10)
                if result.returncode == 0:
                    encoder_config.update({
                        'encoder': encoder,
                        'encoder_params': {
                            'preset': '6' if encoder == 'libsvtav1' else None,
                            'cpu-used': '4' if encoder == 'libaom-av1' else None,
                        }
                    })
                    logger.info(f"Using software AV1 encoder: {encoder}")
                    return encoder, encoder_config
            except:
                continue

        logger.error("No suitable encoder available")
        raise Exception("No encoder available")

    def compress_video(self, input_path: Path, output_path: Path) -> bool:
        """Compress a single video file"""
        logger.info(f"Starting compression: {input_path.name}")

        # Get video information
        video_info = self.get_video_info(input_path)
        if not video_info:
            logger.error(f"Cannot get video info: {input_path}")
            return False

        # Find video and audio streams
        video_stream = None
        audio_streams = []

        for stream in video_info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_streams.append(stream)

        if not video_stream:
            logger.error(f"No video stream found: {input_path}")
            return False

        # Get video parameters
        duration = float(video_info.get('format', {}).get('duration', 0))
        if duration == 0:
            logger.error(f"Cannot get video duration: {input_path}")
            return False

        # Calculate target bitrate
        target_bitrate = self.calculate_bitrate(duration, self.target_size_mb)
        logger.info(
            f"Target bitrate: {target_bitrate}kbps, duration: {duration:.2f}s")

        # Get best available encoder
        try:
            encoder_name, encoder_config = self.get_best_encoder()
        except Exception as e:
            logger.error(f"Encoder selection error: {e}")
            return False

        # Build FFmpeg command
        cmd = ['ffmpeg']

        # Add hardware device if needed
        if encoder_config.get('hw_device'):
            cmd.extend(['-vaapi_device', encoder_config['hw_device']])

        cmd.extend(['-i', str(input_path)])

        # Add input filters for hardware encoding
        if encoder_config.get('input_filters'):
            cmd.extend(['-vf', ','.join(encoder_config['input_filters'])])

        # Video encoding settings
        cmd.extend(['-c:v', encoder_config['encoder']])

        # Hardware encoders use different bitrate control
        if 'nvenc' in encoder_config['encoder']:
            # NVIDIA encoder settings
            cmd.extend(['-b:v', f'{target_bitrate}k'])
            if encoder_config['encoder_params'].get('preset'):
                cmd.extend(
                    ['-preset', encoder_config['encoder_params']['preset']])
            if encoder_config['encoder_params'].get('tune'):
                cmd.extend(['-tune', encoder_config['encoder_params']['tune']])
            if encoder_config['encoder_params'].get('rc'):
                cmd.extend(['-rc', encoder_config['encoder_params']['rc']])
        elif 'vaapi' in encoder_config['encoder']:
            # Intel VAAPI encoder settings
            cmd.extend(['-b:v', f'{target_bitrate}k'])
            if encoder_config['encoder_params'].get('quality'):
                cmd.extend(
                    ['-global_quality', encoder_config['encoder_params']['quality']])
        else:
            # Software encoder settings
            cmd.extend([
                '-b:v', f'{target_bitrate}k',
                '-g', '240',  # GOP size for 4K
                '-pix_fmt', 'yuv420p10le',  # 10-bit for HDR
            ])

            # Software encoder specific parameters
            if encoder_config['encoder'] == 'libsvtav1':
                if encoder_config['encoder_params'].get('preset'):
                    cmd.extend(
                        ['-preset', encoder_config['encoder_params']['preset']])
            elif encoder_config['encoder'] == 'libaom-av1':
                if encoder_config['encoder_params'].get('cpu-used'):
                    cmd.extend(
                        ['-cpu-used', encoder_config['encoder_params']['cpu-used']])

        # HDR handling
        if 'color_primaries' in video_stream:
            cmd.extend(
                ['-color_primaries', str(video_stream.get('color_primaries', '9'))])
        if 'color_transfer' in video_stream:
            cmd.extend(
                ['-color_trc', str(video_stream.get('color_transfer', '16'))])
        if 'color_space' in video_stream:
            cmd.extend(
                ['-colorspace', str(video_stream.get('color_space', '9'))])

        # Audio processing - maintain original quality or compress appropriately
        if audio_streams:
            # Select best audio stream
            best_audio = max(audio_streams, key=lambda x: x.get('channels', 0))
            channels = best_audio.get('channels', 2)

            if channels >= 6:  # 5.1 or more channels
                cmd.extend(['-c:a', 'eac3', '-b:a', '640k'])
            else:  # Stereo
                cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

        # Output options
        cmd.extend([
            '-map', '0:v:0',  # First video stream
        ])

        # Only map audio if audio streams exist
        if audio_streams:
            cmd.extend(['-map', '0:a'])  # All audio streams

        cmd.extend([
            '-sn',            # Skip subtitles
            '-movflags', '+faststart',
            '-y',  # Overwrite output file
            str(output_path)
        ])

        logger.info(f"FFmpeg command: {' '.join(cmd)}")

        # Execute compression
        start_time = time.time()
        last_log_time = 0
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, universal_newlines=True) as process:

                # Monitor progress
                if process.stderr:
                    for line in process.stderr:
                        if 'time=' in line and 'bitrate=' in line:
                            # Parse progress information
                            parts = line.split()
                            for part in parts:
                                if part.startswith('time='):
                                    current_time = part.split('=')[1]
                                    current_elapsed = time.time() - start_time

                                    # Determine logging interval based on elapsed time
                                    if current_elapsed <= 10:
                                        # First 10 seconds: log every 2 seconds
                                        log_interval = 2
                                    else:
                                        # After 10 seconds: log every minute
                                        log_interval = 60

                                    # Only log if enough time has passed since last log
                                    if current_elapsed - last_log_time >= log_interval:
                                        logger.info(
                                            f"Progress: {current_time}/{duration:.2f}s")
                                        last_log_time = current_elapsed
                                    break

                return_code = process.wait()

                if return_code == 0:
                    end_time = time.time()
                    compression_time = end_time - start_time

                    # Check output file size
                    output_size_gb = output_path.stat().st_size / (1024**3)
                    compression_ratio = (
                        input_path.stat().st_size / output_path.stat().st_size)

                    logger.info(f"Compression completed: {input_path.name}")
                    logger.info(f"Output size: {output_size_gb:.2f}GB")
                    logger.info(
                        f"Compression ratio: {compression_ratio:.2f}:1")
                    logger.info(f"Time taken: {compression_time/3600:.2f}h")

                    return True
                else:
                    logger.error(
                        f"FFmpeg compression failed, return code: {return_code}")
                    return False

        except Exception as e:
            logger.error(f"Error during compression: {e}")
            return False

    def find_video_files(self) -> List[Path]:
        """Find all video files in input directory"""
        video_files = []

        if self.recursive:
            # Recursive search in all subdirectories
            logger.info("Searching recursively in all subdirectories...")
            for file_path in self.input_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    video_files.append(file_path)
        else:
            # Search only in the top-level directory
            logger.info("Searching only in the top-level directory...")
            for file_path in self.input_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    video_files.append(file_path)

        logger.info(
            f"Found {len(video_files)} video files ({'recursive' if self.recursive else 'non-recursive'} search)")

        # Log directory structure if recursive and files found
        if self.recursive and video_files:
            directories = set(file.parent for file in video_files)
            logger.info(f"Files found in {len(directories)} directories:")
            for directory in sorted(directories):
                rel_dir = directory.relative_to(self.input_dir)
                count = len([f for f in video_files if f.parent == directory])
                logger.info(f"  {rel_dir or '.'}: {count} files")

        return sorted(video_files)

    def _generate_output_filename(self, input_file: Path) -> str:
        """Generate appropriate output filename based on encoder capabilities"""
        try:
            encoder_name, encoder_config = self.get_best_encoder()

            # Determine codec and suffix
            if 'av1' in encoder_config['encoder']:
                codec_suffix = "AV1"
            elif 'hevc' in encoder_config['encoder'] or 'h265' in encoder_config['encoder']:
                codec_suffix = "HEVC"
            elif 'h264' in encoder_config['encoder']:
                codec_suffix = "H264"
            else:
                codec_suffix = "COMPRESSED"

            # Determine acceleration type
            if 'nvenc' in encoder_config['encoder']:
                accel_suffix = "_GPU_NVIDIA"
            elif 'vaapi' in encoder_config['encoder']:
                accel_suffix = "_GPU_INTEL"
            else:
                accel_suffix = ""

            # Get file extension
            ext = encoder_config.get('output_format', 'mkv')
            if not ext.startswith('.'):
                ext = f'.{ext}'

            return f"{input_file.stem}_{codec_suffix}_4K_HDR10{accel_suffix}{ext}"

        except:
            # Fallback filename
            return f"{input_file.stem}_COMPRESSED.mkv"

    def process_all_videos(self):
        """Process all video files"""
        video_files = self.find_video_files()

        if not video_files:
            logger.warning("No video files found")
            return

        success_count = 0
        total_count = len(video_files)

        for i, video_file in enumerate(video_files, 1):
            # Calculate relative path from input directory
            rel_path = video_file.relative_to(self.input_dir)
            logger.info(f"Processing {i}/{total_count}: {rel_path}")

            # Generate output path maintaining directory structure
            if self.recursive and video_file.parent != self.input_dir:
                # Maintain subdirectory structure
                output_subdir = self.output_dir / \
                    video_file.parent.relative_to(self.input_dir)
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_filename = self._generate_output_filename(video_file)
                output_path = output_subdir / output_filename
            else:
                # Place in root output directory
                output_filename = self._generate_output_filename(video_file)
                output_path = self.output_dir / output_filename

            # Skip existing files
            if output_path.exists():
                logger.info(
                    f"Skipping existing file: {output_path.relative_to(self.output_dir)}")
                continue

            # Compress video
            if self.compress_video(video_file, output_path):
                success_count += 1
            else:
                logger.error(f"Compression failed: {rel_path}")
                # Remove failed output file
                if output_path.exists():
                    output_path.unlink()

        logger.info(
            f"Batch compression completed: {success_count}/{total_count} successful")


def main():
    # Get default values from environment variables
    default_target_size = float(os.environ.get('TARGET_SIZE', 12.0))
    default_recursive = os.environ.get(
        'RECURSIVE', 'true').lower() in ('true', '1', 'yes', 'on')
    default_gpu = os.environ.get(
        'ENABLE_GPU', 'true').lower() in ('true', '1', 'yes', 'on')

    parser = argparse.ArgumentParser(
        description='Batch Movie Compressor - AV1 encoding with GPU acceleration support')
    parser.add_argument('input_dir', help='Input directory path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument('--target-size', type=float, default=default_target_size,
                        help=f'Target file size in GB (default: {default_target_size}GB)')
    parser.add_argument('--recursive', action='store_true', default=default_recursive,
                        help='Search recursively in subdirectories (default: %(default)s)')
    parser.add_argument('--no-recursive', dest='recursive', action='store_false',
                        help='Search only in the top-level directory')
    parser.add_argument('--gpu', action='store_true', default=default_gpu,
                        help='Enable GPU acceleration (default: %(default)s)')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                        help='Disable GPU acceleration, use CPU only')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run mode - only show files to be processed')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if args.dry_run:
        # Dry run mode
        compressor = MovieCompressor(
            args.input_dir, args.output_dir, args.target_size, args.recursive, args.gpu)
        video_files = compressor.find_video_files()

        print("\n=== DRY RUN MODE ===")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Target size: {args.target_size}GB")
        print(f"Recursive search: {args.recursive}")
        print(f"GPU acceleration: {args.gpu}")

        # Show GPU capabilities if enabled
        if args.gpu:
            print(f"GPU capabilities: {compressor.gpu_info}")

        print(f"Video files found: {len(video_files)}")

        if video_files:
            print("\nFiles to be processed:")
            for video_file in video_files:
                file_size_gb = video_file.stat().st_size / (1024**3)
                rel_path = video_file.relative_to(Path(args.input_dir))
                # Show what filename would be generated
                output_filename = compressor._generate_output_filename(
                    video_file)
                print(
                    f"  - {rel_path} ({file_size_gb:.2f}GB) -> {output_filename}")

        return

    # Normal execution mode
    logger.info("Starting batch movie compression")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target size: {args.target_size}GB")
    logger.info(f"Recursive search: {args.recursive}")
    logger.info(f"GPU acceleration: {args.gpu}")

    compressor = MovieCompressor(
        args.input_dir, args.output_dir, args.target_size, args.recursive, args.gpu)
    compressor.process_all_videos()


if __name__ == "__main__":
    main()
