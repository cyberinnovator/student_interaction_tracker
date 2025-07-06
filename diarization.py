"""
diarization.py

Handles speaker diarization and segment extraction using pyannote.audio 3.1.
"""

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face token
HUGGINGFACE_TOKEN = ""

# Load the diarization pipeline
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    logger.info("Successfully loaded pyannote.audio 3.1 pipeline")
except Exception as e:
    logger.error(f"Failed to load pipeline: {e}")
    raise

def diarize_audio(wav_path):
    """
    Diarize the given .wav file and return a list of speaker segments.
    Args:
        wav_path (str): Path to the input .wav file.
        rttm_path (str): Path to save the RTTM file.
    Returns:
        List[dict]: [{'speaker': 'SPEAKER_00', 'start': float, 'end': float}, ...]
    """
    try:
        # Run diarization with progress hook for better feedback
        logger.info(f"Starting diarization for {wav_path}")
        with ProgressHook() as hook:
            diarization = pipeline(wav_path, hook=hook)
        
        # Save RTTM file
        with open("output.rttm", "w") as rttm:
            diarization.write_rttm(rttm)
        
        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": float(turn.start),
                "end": float(turn.end)
            })
        
        logger.info(f"Diarization completed. Found {len(segments)} segments.")
        return segments
        
    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        raise