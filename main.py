#!/usr/bin/env python3
"""
Batch Movie Compressor - AV1 encoding for 4K HDR movies
Compress ~30GB movies to ~12GB while maintaining 4K HDR10 quality
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Optional[Path]) -> None:
    """Configure logging for both console and file.

    Priority for log file location:
    1) Environment variable LOG_PATH if set
    2) <output_dir>/compression.log if output_dir provided
    If neither works, fall back to console-only logging.
    Log level can be controlled via LOG_LEVEL or FFMPEG_LOG_LEVEL env vars.
    """
    # Determine log level from env; default INFO
    level_name = os.environ.get('LOG_LEVEL') or os.environ.get(
        'FFMPEG_LOG_LEVEL') or 'INFO'
    level = getattr(logging, str(level_name).upper(), logging.INFO)

    # Clear existing handlers to avoid duplicate logs on reruns (e.g., in VS Code)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handlers: List[logging.Handler] = [logging.StreamHandler()]

    log_path_env = os.environ.get('LOG_PATH')
    log_path: Optional[Path] = None
    if log_path_env:
        log_path = Path(log_path_env)
    elif output_dir is not None:
        log_path = Path(output_dir) / 'compression.log'

    # Try to create file handler if possible
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.insert(0, logging.FileHandler(log_path))
        except Exception:
            # If file handler fails, continue with console only
            pass

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


class MovieCompressor:
    def __init__(self, input_dir: str, output_dir: str, target_size_gb: float = 12.0, recursive: bool = True, enable_gpu: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size_gb = target_size_gb
        self.target_size_mb = target_size_gb * 1024
        self.recursive = recursive
        self.enable_gpu = enable_gpu

        # Environment-driven tunables
        self.audio_bitrate_kbps_default = int(
            os.environ.get('AUDIO_BITRATE_KBPS', '640'))
        self.safety_reserve_mb = float(
            os.environ.get('SAFETY_RESERVE_MB', '100'))
        self.min_video_bitrate_kbps = int(
            os.environ.get('MIN_VIDEO_BITRATE_KBPS', '1000'))
        self.vaapi_device = os.environ.get(
            'VAAPI_DEVICE', '/dev/dri/renderD128')
        self.preferred_encoder_env = os.environ.get(
            'PREFERRED_ENCODER')  # e.g., av1_nvenc, av1_vaapi, libsvtav1
        self.preferred_decoder_env = os.environ.get(
            'PREFERRED_DECODER')  # e.g., hevc_cuvid, h264_vaapi
        self.force_gpu_decode = os.environ.get(
            'FORCE_GPU_DECODE', 'false').strip().lower() in ('true', '1', 'yes', 'on')  # Force GPU decoding even if not optimal
        self.output_format_env = os.environ.get(
            'OUTPUT_FORMAT')  # e.g., mkv, mp4
        self.eac3_bitrate_kbps = int(
            os.environ.get('EAC3_BITRATE_KBPS', '640'))
        self.aac_bitrate_kbps = int(os.environ.get('AAC_BITRATE_KBPS', '192'))
        self.progress_log_interval_start = int(
            os.environ.get('PROGRESS_LOG_INTERVAL_START_SEC', '2'))
        self.progress_log_interval = int(
            os.environ.get('PROGRESS_LOG_INTERVAL_SEC', '60'))

        # GPU acceleration capabilities
        self.gpu_info = {
            'nvidia': False,
            'intel': False,
            'available_encoders': [],
            'available_decoders': [],
            'preferred_encoder': None,
            'preferred_decoder': None
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

    def _libaom_cpu_used(self) -> str:
        """Compute libaom-av1 -cpu-used as (CPU count - 1), clamped to 0..8"""
        try:
            cpu_cnt = os.cpu_count() or 1
        except Exception:
            cpu_cnt = 1
        val = max(0, min(8, cpu_cnt - 1))
        return str(val)

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

            # Test NVIDIA decoders
            nvidia_decoders = ['h264_cuvid',
                               'hevc_cuvid', 'av1_cuvid', 'vp9_cuvid']
            for decoder in nvidia_decoders:
                try:
                    # Test with a simple probe command
                    result = subprocess.run(
                        ['ffmpeg', '-hide_banner', '-f', 'lavfi', '-i', 'testsrc2=duration=0.1:size=64x64:rate=1',
                         '-c:v', 'libx264', '-t', '0.1', '-f', 'h264', 'pipe:1'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5
                    )
                    if result.returncode == 0 and result.stdout:
                        # Test if decoder can handle the stream
                        test_decode = subprocess.run(
                            ['ffmpeg', '-hide_banner', '-c:v', decoder,
                                '-i', 'pipe:0', '-f', 'null', '-'],
                            input=result.stdout, capture_output=True, timeout=5
                        )
                        if test_decode.returncode == 0:
                            self.gpu_info['available_decoders'].append(decoder)
                            logger.info(f"NVIDIA decoder available: {decoder}")
                except:
                    continue

        except Exception as e:
            logger.debug(f"NVIDIA GPU test failed: {e}")

        # Check for Intel GPU support (VAAPI)
        try:
            # Test Intel/VAAPI encoders
            intel_encoders = ['h264_vaapi', 'hevc_vaapi', 'av1_vaapi']
            for encoder in intel_encoders:
                result = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-vaapi_device', self.vaapi_device,
                     '-f', 'lavfi', '-i', 'testsrc2=duration=0.1:size=64x64:rate=1',
                     '-vf', 'format=nv12,hwupload', '-c:v', encoder, '-f', 'null', '-'],
                    capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    self.gpu_info['intel'] = True
                    self.gpu_info['available_encoders'].append(encoder)
                    logger.info(f"Intel VAAPI encoder available: {encoder}")

            # Test Intel/VAAPI decoders
            if self.gpu_info['intel']:  # Only test if Intel GPU is detected
                intel_decoders = ['h264_vaapi',
                                  'hevc_vaapi', 'av1_vaapi', 'vp9_vaapi']
                for decoder in intel_decoders:
                    try:
                        # Simple test with VAAPI device
                        result = subprocess.run(
                            ['ffmpeg', '-hide_banner', '-vaapi_device', self.vaapi_device,
                             '-f', 'lavfi', '-i', 'testsrc2=duration=0.1:size=64x64:rate=1',
                             '-c:v', 'libx264', '-t', '0.1', '-f', 'h264', '/tmp/test_vaapi_decode.h264'],
                            capture_output=True, timeout=5
                        )
                        if result.returncode == 0:
                            # Test decode
                            test_result = subprocess.run(
                                ['ffmpeg', '-hide_banner', '-vaapi_device', self.vaapi_device,
                                 '-hwaccel', 'vaapi', '-i', '/tmp/test_vaapi_decode.h264', '-f', 'null', '-'],
                                capture_output=True, timeout=5
                            )
                            if test_result.returncode == 0:
                                self.gpu_info['available_decoders'].append(
                                    decoder)
                                logger.info(
                                    f"Intel VAAPI decoder available: {decoder}")
                            # Clean up test file
                            try:
                                os.unlink('/tmp/test_vaapi_decode.h264')
                            except:
                                pass
                    except:
                        continue

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

        # Determine preferred decoder
        if 'hevc_cuvid' in self.gpu_info['available_decoders']:
            self.gpu_info['preferred_decoder'] = 'hevc_cuvid'
            logger.info("Using NVIDIA CUVID hardware decoder")
        elif 'h264_cuvid' in self.gpu_info['available_decoders']:
            self.gpu_info['preferred_decoder'] = 'h264_cuvid'
            logger.info("Using NVIDIA H.264 CUVID hardware decoder")
        elif 'av1_cuvid' in self.gpu_info['available_decoders']:
            self.gpu_info['preferred_decoder'] = 'av1_cuvid'
            logger.info("Using NVIDIA AV1 CUVID hardware decoder")
        elif 'hevc_vaapi' in self.gpu_info['available_decoders']:
            self.gpu_info['preferred_decoder'] = 'hevc_vaapi'
            logger.info("Using Intel VAAPI HEVC hardware decoder")
        elif 'h264_vaapi' in self.gpu_info['available_decoders']:
            self.gpu_info['preferred_decoder'] = 'h264_vaapi'
            logger.info("Using Intel VAAPI H.264 hardware decoder")
        else:
            logger.info(
                "No hardware decoders available, will use software decoding")

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
        """Calculate target video bitrate using env-configured safety and minimums"""
        # Use env-configured audio bitrate if not explicitly provided
        effective_audio_kbps = int(os.environ.get(
            'AUDIO_BITRATE_KBPS', str(audio_bitrate_kbps)))
        audio_size_mb = (effective_audio_kbps * duration) / (8 * 1024)
        video_size_mb = target_size_mb - audio_size_mb - self.safety_reserve_mb

        if video_size_mb <= 0:
            logger.warning(
                "Target size too small after safety reserve, using minimum bitrate")
            return self.min_video_bitrate_kbps

        video_bitrate_kbps = int((video_size_mb * 8 * 1024) / duration)
        return max(video_bitrate_kbps, self.min_video_bitrate_kbps)

    def min_video_bit_rate(self) -> int:
        return self.min_video_bitrate_kbps

    def get_best_decoder(self, video_codec: str) -> Optional[str]:
        """Get the best available hardware decoder for the given codec"""
        if not self.enable_gpu:
            return None

        # If a preferred decoder is forced via env, use that directly
        if self.preferred_decoder_env:
            forced_decoder = self.preferred_decoder_env.strip()
            if forced_decoder in self.gpu_info['available_decoders']:
                logger.info(
                    f"Using forced decoder: {forced_decoder} for codec: {video_codec}")
                return forced_decoder
            else:
                logger.warning(
                    f"Forced decoder {forced_decoder} not available, falling back to auto selection")

        # Map video codec to corresponding hardware decoders
        codec_to_decoders = {
            'h264': ['h264_cuvid', 'h264_vaapi'],
            'avc': ['h264_cuvid', 'h264_vaapi'],  # H.264 alias
            'hevc': ['hevc_cuvid', 'hevc_vaapi'],
            'h265': ['hevc_cuvid', 'hevc_vaapi'],  # HEVC alias
            'av1': ['av1_cuvid', 'av1_vaapi'],
            'vp9': ['vp9_cuvid', 'vp9_vaapi'],  # VP9 support
            'mpeg2': ['mpeg2_cuvid'],  # MPEG-2 support
            'mpeg4': ['mpeg4_cuvid'],  # MPEG-4 support
            'vc1': ['vc1_cuvid'],  # VC-1 support
        }

        # Get potential decoders for this codec
        potential_decoders = codec_to_decoders.get(video_codec.lower(), [])

        # If no specific decoders for this codec but force GPU decode is enabled,
        # try generic hardware decoders
        if not potential_decoders and self.force_gpu_decode:
            potential_decoders = [d for d in self.gpu_info['available_decoders']
                                  if 'cuvid' in d or 'vaapi' in d]

        # Prioritize NVIDIA CUVID if available (generally better performance)
        nvidia_decoders = [d for d in potential_decoders if 'cuvid' in d]
        intel_decoders = [d for d in potential_decoders if 'vaapi' in d]

        # Try NVIDIA first, then Intel
        for decoder_list in [nvidia_decoders, intel_decoders]:
            for decoder in decoder_list:
                if decoder in self.gpu_info['available_decoders']:
                    logger.info(
                        f"Using hardware decoder: {decoder} for codec: {video_codec}")
                    return decoder

        logger.info(
            f"No hardware decoder available for codec: {video_codec}, using software decoding")
        return None

    def get_best_encoder(self) -> tuple[str, dict]:
        """Get the best available encoder (GPU preferred, then CPU)"""
        encoder_config = {
            'encoder': None,
            'hw_device': None,
            'input_filters': [],
            'encoder_params': {},
            'output_format': (self.output_format_env or 'mkv')
        }

        # If a preferred encoder is forced via env, use that directly
        if self.preferred_encoder_env:
            forced = self.preferred_encoder_env.strip()
            if forced in ('av1_nvenc', 'hevc_nvenc', 'h264_nvenc'):
                encoder_config.update({
                    'encoder': forced,
                    'encoder_params': {
                        'preset': 'p4' if forced == 'av1_nvenc' else 'medium',
                        'tune': 'hq',
                        'rc': 'vbr',
                    }
                })
                return forced, encoder_config
            if forced in ('av1_vaapi', 'hevc_vaapi', 'h264_vaapi'):
                encoder_config.update({
                    'encoder': forced,
                    'hw_device': self.vaapi_device,
                    'input_filters': ['format=nv12', 'hwupload'],
                    'encoder_params': {
                        'quality': '25',
                    }
                })
                return forced, encoder_config
            if forced in ('libsvtav1', 'libaom-av1'):
                encoder_config.update({
                    'encoder': forced,
                    'encoder_params': {
                        'preset': '6' if forced == 'libsvtav1' else None,
                        'cpu-used': self._libaom_cpu_used() if forced == 'libaom-av1' else None,
                    }
                })
                return forced, encoder_config

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
                    'hw_device': self.vaapi_device,
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
                    # HEVC works better with MP4
                    'output_format': (self.output_format_env or 'mp4')
                })
                logger.info("Using NVIDIA HEVC hardware encoder (fallback)")
                return preferred, encoder_config

            elif preferred == 'hevc_vaapi':
                encoder_config.update({
                    'encoder': 'hevc_vaapi',
                    'hw_device': self.vaapi_device,
                    'input_filters': ['format=nv12', 'hwupload'],
                    'encoder_params': {
                        'quality': '25',
                    },
                    'output_format': (self.output_format_env or 'mp4')
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
                            'cpu-used': self._libaom_cpu_used() if encoder == 'libaom-av1' else None,
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

        # Get best available decoder for the input video codec
        input_codec = video_stream.get('codec_name', '')
        hardware_decoder = self.get_best_decoder(input_codec)

        # Build FFmpeg command
        cmd = ['ffmpeg']

        # Add hardware device if needed (for encoding or decoding)
        if encoder_config.get('hw_device') or hardware_decoder:
            if encoder_config.get('hw_device'):
                cmd.extend(['-vaapi_device', encoder_config['hw_device']])

        # Add hardware decoder if available
        if hardware_decoder:
            if 'cuvid' in hardware_decoder:
                # NVIDIA CUVID decoder
                cmd.extend(['-hwaccel', 'cuvid', '-c:v', hardware_decoder])
            elif 'vaapi' in hardware_decoder:
                # Intel VAAPI decoder
                cmd.extend(
                    ['-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi'])

        cmd.extend(['-i', str(input_path)])

        # Build video filter chain
        video_filters = []

        # Handle hardware decoding to encoding pipeline
        if hardware_decoder and encoder_config.get('encoder'):
            if 'cuvid' in hardware_decoder and 'nvenc' in encoder_config['encoder']:
                # NVIDIA GPU pipeline: CUVID -> NVENC (no format conversion needed)
                logger.info(
                    "Using NVIDIA GPU-to-GPU pipeline (CUVID -> NVENC)")
            elif 'cuvid' in hardware_decoder and 'vaapi' in encoder_config['encoder']:
                # NVIDIA -> Intel pipeline: need to download and upload
                video_filters.extend(['hwdownload', 'format=nv12', 'hwupload'])
                logger.info("Using NVIDIA to Intel GPU pipeline")
            elif 'vaapi' in hardware_decoder and 'nvenc' in encoder_config['encoder']:
                # Intel -> NVIDIA pipeline: need to download and upload
                video_filters.extend(['hwdownload', 'format=nv12'])
                logger.info("Using Intel to NVIDIA GPU pipeline")
            elif 'vaapi' in hardware_decoder and 'vaapi' in encoder_config['encoder']:
                # Intel GPU pipeline: VAAPI -> VAAPI (no format conversion needed)
                logger.info("Using Intel GPU-to-GPU pipeline (VAAPI -> VAAPI)")
            else:
                # Hardware decode to software encode
                if 'cuvid' in hardware_decoder:
                    video_filters.append('hwdownload')
                elif 'vaapi' in hardware_decoder:
                    video_filters.append('hwdownload')
                logger.info(
                    "Using hardware decode to software encode pipeline")
        elif encoder_config.get('input_filters'):
            # Software decode to hardware encode
            video_filters.extend(encoder_config['input_filters'])
            logger.info("Using software decode to hardware encode pipeline")

        # Add video filters if any
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])

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

            # Log CPU core count when using software encoding
            try:
                cpu_cores = os.cpu_count() or 1
            except Exception:
                cpu_cores = 1
            logger.info(
                f"Software encoding selected; CPU cores available: {cpu_cores}")

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
                cmd.extend(['-c:a', 'eac3', '-b:a',
                           f'{self.eac3_bitrate_kbps}k'])
            else:  # Stereo
                cmd.extend(
                    ['-c:a', 'aac', '-b:a', f'{self.aac_bitrate_kbps}k'])

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
                                    if current_elapsed <= 60:
                                        # First seconds: log at configured fast interval
                                        log_interval = self.progress_log_interval_start
                                    else:
                                        # After threshold: log at configured normal interval
                                        log_interval = self.progress_log_interval

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


def _getenv_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ('true', '1', 'yes', 'on')


def main():
    # Read configuration from environment variables only
    input_dir = os.environ.get('INPUT_DIR', '/data/in')
    output_dir = os.environ.get('OUTPUT_DIR', '/data/out')
    target_size_gb = float(os.environ.get('TARGET_SIZE', '12.0'))
    recursive = _getenv_bool('RECURSIVE', True)
    enable_gpu = _getenv_bool('ENABLE_GPU', True)
    dry_run = _getenv_bool('DRY_RUN', False)

    # Configure logging using output_dir (or LOG_PATH)
    try:
        setup_logging(Path(output_dir))
    except Exception:
        pass

    # Validate input directory
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if dry_run:
        compressor = MovieCompressor(
            input_dir, output_dir, target_size_gb, recursive, enable_gpu)
        video_files = compressor.find_video_files()

        print("\n=== DRY RUN MODE ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Target size: {target_size_gb}GB")
        print(f"Recursive search: {recursive}")
        print(f"GPU acceleration: {enable_gpu}")

        if enable_gpu:
            print(f"GPU capabilities:")
            print(f"  NVIDIA: {compressor.gpu_info['nvidia']}")
            print(f"  Intel: {compressor.gpu_info['intel']}")
            print(
                f"  Available encoders: {compressor.gpu_info['available_encoders']}")
            print(
                f"  Available decoders: {compressor.gpu_info['available_decoders']}")
            print(
                f"  Preferred encoder: {compressor.gpu_info['preferred_encoder']}")
            print(
                f"  Preferred decoder: {compressor.gpu_info['preferred_decoder']}")

        print(f"Video files found: {len(video_files)}")

        if video_files:
            print("\nFiles to be processed:")
            for video_file in video_files:
                file_size_gb = video_file.stat().st_size / (1024**3)
                rel_path = video_file.relative_to(Path(input_dir))
                output_filename = compressor._generate_output_filename(
                    video_file)
                print(
                    f"  - {rel_path} ({file_size_gb:.2f}GB) -> {output_filename}")
        return

    # Normal execution mode
    logger.info("Starting batch movie compression")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target size: {target_size_gb}GB")
    logger.info(f"Recursive search: {recursive}")
    logger.info(f"GPU acceleration: {enable_gpu}")

    compressor = MovieCompressor(
        input_dir, output_dir, target_size_gb, recursive, enable_gpu)
    compressor.process_all_videos()


if __name__ == "__main__":
    main()
