"""
unknown_speaker.py

Handles unknown speaker processing: transcription, roll number extraction, and student registration.
"""

import re
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("faster-whisper imported successfully")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.error("faster-whisper not installed. Please install with: pip install faster-whisper")
    raise

# CPU mode configuration
def setup_cpu_mode():
    """Setup CPU mode for transcription."""
    print(f"\n🔧 SYSTEM CONFIGURATION:")
    print(f"   Device: CPU")
    print(f"   Compute Type: float32")
    print(f"   Status: ✅ CPU MODE ENABLED")
    return []

# Transcription configuration - CPU mode
DEVICE = "cpu"  # Use CPU for transcription
COMPUTE_TYPE = "float32"  # Use float32 for compatibility

# Whisper model options - optimized for CPU
WHISPER_MODELS = {
    "tiny": "tiny",      # Fastest, smallest (39MB), perfect for CPU
    "base": "base",      # Good balance (74MB), works well on CPU
    "small": "small",    # Better accuracy (244MB), slower on CPU
}

# Choose your preferred model here - tiny is fastest and safest for CPU
SELECTED_MODEL = "tiny"  # Recommended for CPU

# Setup CPU mode
supported_compute_types = setup_cpu_mode()

# Load faster-whisper model with CPU
whisper_model = None
try:
    # Use faster-whisper with CPU
    whisper_model = WhisperModel(
        SELECTED_MODEL, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE,
        cpu_threads=4,  # CPU threads for processing
        num_workers=1   # Single worker
    )
    logger.info(f"Loaded faster-whisper model: {SELECTED_MODEL} on {DEVICE} with {COMPUTE_TYPE}")
    print(f"\n🎯 WHISPER MODEL LOADED:")
    print(f"   Model: {SELECTED_MODEL}")
    print(f"   Device: {DEVICE}")
    print(f"   Compute Type: {COMPUTE_TYPE}")
    print(f"   Status: ✅ CPU MODE ENABLED\n")
except Exception as e:
    logger.error(f"Failed to load faster-whisper model: {e}")
    # Try with simpler settings
    try:
        whisper_model = WhisperModel(
            "tiny", 
            device="cpu", 
            compute_type="float32",
            cpu_threads=2,
            num_workers=1
        )
        logger.info("Fallback to tiny model with simpler settings")
        print(f"\n🎯 WHISPER MODEL LOADED (FALLBACK):")
        print(f"   Model: tiny")
        print(f"   Device: cpu")
        print(f"   Compute Type: float32")
        print(f"   Status: ✅ CPU MODE (FALLBACK)\n")
    except Exception as e2:
        logger.error(f"Failed to load fallback model: {e2}")
        raise

def extract_roll_number_from_text(text):
    """
    Extract roll number from transcribed text using regex patterns.
    Args:
        text (str): Transcribed text.
    Returns:
        str or None: Roll number if found, else None.
    """
    try:
        # Strictly look for "roll no. is" pattern as requested
        patterns = [
            r'roll\s+no\.?\s+is\s+(\d+)',  # "roll no. is 123" or "roll no is 123"
            r'roll\s+number\s+is\s+(\d+)',  # "roll number is 123"
            r'role\s+number\s+is\s+(\d+)',  # "role number is 123" (common transcription)
            r'my\s+roll\s+no\.?\s+is\s+(\d+)',  # "my roll no. is 123"
            r'my\s+role\s+number\s+is\s+(\d+)',  # "my role number is 123"
            r'i\s+am\s+(\d+)',  # "i am 123" (fallback)
        ]
        
        text_lower = text.lower()
        
        # Print roll number extraction process prominently
        print("\n" + "🔍"*20)
        print("🔍 ROLL NUMBER EXTRACTION:")
        print("🔍"*20)
        print(f"Text to analyze: '{text}'")
        print(f"Text (lowercase): '{text_lower}'")
        
        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, text_lower)
            print(f"Pattern {i}: {pattern}")
            if match:
                roll_no = match.group(1)
                print(f"✅ MATCH FOUND! Roll number: {roll_no}")
                logger.info(f"Extracted roll number: {roll_no} from text: '{text}' using pattern: {pattern}")
                print("🔍"*20 + "\n")
                return roll_no
            else:
                print(f"❌ No match")
        
        print("❌ NO ROLL NUMBER FOUND IN ANY PATTERN")
        logger.warning(f"No roll number found in text: '{text}'")
        print("🔍"*20 + "\n")
        return None
    except Exception as e:
        logger.error(f"Error extracting roll number: {e}")
        return None

def transcribe_full_audio(audio_path):
    """
    Transcribe the entire audio file for better accuracy using faster-whisper.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        str: Transcribed text.
    """
    try:
        logger.info(f"Transcribing full audio file: {audio_path}")
        
        # Preprocess audio to improve transcription quality
        processed_audio_path = audio_path  # Default to original
        if AUDIO_PROCESSING_AVAILABLE:
            print("🔧 Preprocessing audio for better transcription...")
            processed_audio_path = preprocess_audio_for_transcription(audio_path)
            print(f"✅ Audio preprocessing completed: {processed_audio_path}")
        else:
            print("⚠️ Using original audio (audio processing not available)")
        
        if whisper_model is not None:
            try:
                # Use CPU transcription with preprocessed audio
                print("🎤 Starting transcription with preprocessed audio...")
                segments, info = whisper_model.transcribe(
                    processed_audio_path, 
                    beam_size=5,
                    language="en",  # Specify English for better accuracy
                    vad_filter=True  # Use voice activity detection
                )
                transcription = " ".join([segment.text for segment in segments])
                print("✅ Transcription successful!")
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                return ""
        else:
            logger.error("No transcription model available")
            return ""
        
        logger.info(f"Full transcription: '{transcription}'")
        
        # Print transcription prominently to terminal for debugging
        print("\n" + "="*60)
        print("🎤 TRANSCRIBED TEXT:")
        print("="*60)
        print(f"'{transcription}'")
        print("="*60 + "\n")
        
        # Clean up processed audio file
        if AUDIO_PROCESSING_AVAILABLE and processed_audio_path != audio_path:
            cleanup_processed_files(processed_audio_path)
        
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing full audio: {e}")
        return ""

def map_transcription_to_speakers(audio_path, segments, full_transcription):
    """
    Map the full transcription to individual speaker segments.
    Args:
        audio_path (str): Path to the audio file.
        segments (List[dict]): List of speaker segments.
        full_transcription (str): Full audio transcription.
    Returns:
        dict: Mapping of speaker to transcription.
    """
    try:
        if not full_transcription:
            logger.warning("No transcription available to map to speakers")
            return {}
        
        # Ensure transcription is a string and not empty
        if not isinstance(full_transcription, str):
            full_transcription = str(full_transcription)
        
        # Check if transcription is empty or just whitespace
        if not full_transcription or full_transcription.isspace():
            logger.warning("No transcription available to map to speakers")
            return {}
        
        # For now, we'll use a simple approach: assign the full transcription
        # to the longest speaking segment, or distribute evenly
        speaker_transcriptions = {}
        
        if len(segments) == 1:
            # Single speaker, assign full transcription
            speaker_transcriptions[segments[0]['speaker']] = full_transcription
        else:
            # Multiple speakers - assign to the longest segment
            longest_segment = max(segments, key=lambda x: x['end'] - x['start'])
            speaker_transcriptions[longest_segment['speaker']] = full_transcription
            
            # For other speakers, assign empty or partial transcription
            for segment in segments:
                if segment['speaker'] != longest_segment['speaker']:
                    speaker_transcriptions[segment['speaker']] = ""
        
        logger.info(f"Mapped transcription to speakers: {speaker_transcriptions}")
        return speaker_transcriptions
        
    except Exception as e:
        logger.error(f"Error mapping transcription to speakers: {e}")
        return {}

def process_unknown_speaker(embedding, audio_path, segments, full_transcription, save_embedding_func, add_student_func, get_student_func):
    """
    Process an unknown speaker by transcribing and extracting roll number.
    Args:
        embedding (np.ndarray): Speaker embedding.
        audio_path (str): Path to the audio file.
        segments (List[dict]): List of speaker segments.
        full_transcription (str): Full audio transcription.
        save_embedding_func (callable): Function to save embedding.
        add_student_func (callable): Function to add student to database.
        get_student_func (callable): Function to check if student exists.
    Returns:
        str or None: Roll number if registered, else None.
    """
    try:
        logger.info("Processing unknown speaker...")
        
        # Map transcription to speakers
        speaker_transcriptions = map_transcription_to_speakers(audio_path, segments, full_transcription)
        
        # Try to find roll number in any speaker's transcription
        roll_no = None
        for speaker, transcription in speaker_transcriptions.items():
            if transcription.strip():
                roll_no = extract_roll_number_from_text(transcription)
                if roll_no:
                    logger.info(f"Found roll number {roll_no} for speaker {speaker}")
                    break
        
        if roll_no:
            # Check if this roll number already exists
            existing_student = get_student_func(roll_no)
            if existing_student:
                logger.info(f"Student with roll number {roll_no} already exists")
                return roll_no
            
            # Save embedding for new student
            embedding_path = save_embedding_func(embedding, f"{roll_no}.npy")
            
            # Add to database
            add_student_func(roll_no, embedding_path)
            logger.info(f"Added new student with roll number {roll_no}")
            return roll_no
        else:
            logger.warning("Could not extract roll number from transcription")
            return None
            
    except Exception as e:
        logger.error(f"Error processing unknown speaker: {e}")
        return None

# Import audio processing module
try:
    from audio_processing import preprocess_audio_for_transcription, cleanup_processed_files
    AUDIO_PROCESSING_AVAILABLE = True
    logger.info("Audio processing module imported successfully")
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.warning("Audio processing module not available. Using original audio for transcription.") 