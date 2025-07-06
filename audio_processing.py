"""
audio_processing.py

Handles audio preprocessing to improve transcription quality while preserving speaker detection.
"""

import numpy as np
import logging
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import os

# Set up logging
logger = logging.getLogger(__name__)

def preprocess_audio_for_transcription(audio_path, output_path=None):
    """
    Preprocess audio specifically for transcription to improve text accuracy.
    Uses gentle processing to avoid affecting speaker detection.
    
    Args:
        audio_path (str): Path to input audio file
        output_path (str): Path to save processed audio (optional)
    Returns:
        str: Path to processed audio file
    """
    try:
        logger.info(f"Preprocessing audio for transcription: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # 1. Gentle normalization (preserves relative volumes)
        y = librosa.util.normalize(y, norm=np.inf)
        
        # 2. Mild high-pass filter (only remove very low noise, keep deep voices)
        y = apply_gentle_highpass_filter(y, sr, cutoff=60)
        
        # 3. Mild low-pass filter (keep most speech frequencies)
        y = apply_gentle_lowpass_filter(y, sr, cutoff=10000)
        
        # 4. Gentle noise reduction (preserves quiet speakers)
        y = apply_gentle_noise_reduction(y, sr)
        
        # 5. Final normalization
        y = librosa.util.normalize(y)
        
        # Save processed audio
        if output_path is None:
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_transcription_processed.wav"
        
        sf.write(output_path, y, sr)
        logger.info(f"Audio preprocessing for transcription completed: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error preprocessing audio for transcription: {e}")
        return audio_path  # Return original if preprocessing fails

def apply_gentle_highpass_filter(y, sr, cutoff=60):
    """Apply gentle high-pass filter to remove very low-frequency noise only."""
    try:
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        # Use lower order filter for gentler effect
        result = butter(2, normal_cutoff, btype='high', analog=False)
        if result is None:
            return y
        b, a = result[:2]  # Take only first two elements
        filtered = filtfilt(b, a, y)
        return filtered
    except Exception as e:
        logger.warning(f"Gentle high-pass filter failed: {e}")
        return y

def apply_gentle_lowpass_filter(y, sr, cutoff=10000):
    """Apply gentle low-pass filter to remove high-frequency noise only."""
    try:
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        # Use lower order filter for gentler effect
        result = butter(2, normal_cutoff, btype='low', analog=False)
        if result is None:
            return y
        b, a = result[:2]  # Take only first two elements
        filtered = filtfilt(b, a, y)
        return filtered
    except Exception as e:
        logger.warning(f"Gentle low-pass filter failed: {e}")
        return y

def apply_gentle_noise_reduction(y, sr, noise_reduce_level=0.05):
    """Apply gentle noise reduction that preserves quiet speakers."""
    try:
        # Calculate noise profile from first 0.3 seconds (assuming it's mostly noise)
        noise_samples = int(0.3 * sr)
        if len(y) > noise_samples:
            noise_profile = y[:noise_samples]
            
            # Apply very gentle spectral gating
            stft = librosa.stft(y)
            noise_stft = librosa.stft(noise_profile)
            
            # Calculate noise spectrum
            noise_spectrum = np.mean(np.abs(noise_stft), axis=1)
            
            # Apply very gentle gate (lower threshold)
            gate_threshold = noise_spectrum * noise_reduce_level
            gate_threshold = gate_threshold.reshape(-1, 1)
            
            # Apply gate with soft transition
            magnitude = np.abs(stft)
            gate_factor = np.maximum(0.8, magnitude / (magnitude + gate_threshold))
            stft_gated = stft * gate_factor
            
            # Convert back to time domain
            y_denoised = librosa.istft(stft_gated)
            return y_denoised
        else:
            return y
    except Exception as e:
        logger.warning(f"Gentle noise reduction failed: {e}")
        return y

def enhance_speech_clarity(y, sr):
    """Enhance speech clarity using gentle spectral enhancement."""
    try:
        # Apply gentle spectral enhancement for speech
        stft = librosa.stft(y)
        
        # Calculate spectral magnitude
        magnitude = np.abs(stft)
        
        # Apply gentle enhancement (small boost)
        enhanced_magnitude = magnitude * 1.1  # Very gentle boost
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * np.angle(stft))
        y_enhanced = librosa.istft(enhanced_stft)
        
        return y_enhanced
    except Exception as e:
        logger.warning(f"Speech enhancement failed: {e}")
        return y

def get_audio_info(audio_path):
    """Get basic audio information for debugging."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        logger.info(f"Audio info - Duration: {duration:.2f}s, Sample rate: {sr}Hz")
        return {
            'duration': duration,
            'sample_rate': sr,
            'samples': len(y)
        }
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        return None

def cleanup_processed_files(processed_path):
    """Clean up processed audio files to save disk space."""
    try:
        if os.path.exists(processed_path) and processed_path != 'audio_chunk.wav':
            os.remove(processed_path)
            logger.info(f"Cleaned up processed file: {processed_path}")
    except Exception as e:
        logger.warning(f"Could not cleanup processed file: {e}") 