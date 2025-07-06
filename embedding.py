"""
embedding.py

Handles speaker embedding extraction and similarity calculations using Resemblyzer.
"""

from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io import wavfile
import numpy as np
import os
from scipy.io.wavfile import write
import logging

# Set up logging
logger = logging.getLogger(__name__)

EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

encoder = VoiceEncoder()

def extract_embedding_from_audio(audio_path):
    """
    Extract speaker embedding from a full audio file.
    Args:
        audio_path (str): Path to the audio file.
    Returns:
        np.ndarray: The embedding vector.
    """
    try:
        wav = preprocess_wav(audio_path)
        embedding = encoder.embed_utterance(wav)
        return embedding
    except Exception as e:
        logger.error(f"Error extracting embedding from audio: {e}")
        raise

def extract_embedding_from_segment(audio_path, start_time, end_time):
    """
    Extract embedding from a specific time segment of audio.
    Args:
        audio_path (str): Path to the audio file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    Returns:
        np.ndarray: The embedding vector.
    """
    try:
        # Read the audio file
        rate, data = wavfile.read(audio_path)
        if data.ndim > 1:  # Convert stereo to mono
            data = data[:, 0]
        
        # Extract the specific segment
        start_sample = int(start_time * rate)
        end_sample = int(end_time * rate)
        segment_data = data[start_sample:end_sample]
        
        # Save segment to temporary file
        temp_path = f"temp_segment_{start_time}_{end_time}.wav"
        write(temp_path, rate, segment_data.astype(data.dtype))
        
        # Extract embedding from the segment
        wav = preprocess_wav(temp_path)
        embedding = encoder.embed_utterance(wav)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting embedding from segment: {e}")
        raise

def concatenate_speaker_segments(wav_path, intervals):
    """
    Concatenate all audio segments for a speaker using scipy.
    Args:
        wav_path (str): Path to the original audio file.
        intervals (List[Tuple[float, float]]): List of (start, end) times in seconds.
    Returns:
        str: Path to the temporary concatenated audio file.
    """
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:  # If stereo, take only one channel
        data = data[:, 0]
    segments = []
    for start, end in intervals:
        start_sample = int(start * rate)
        end_sample = int(end * rate)
        segments.append(data[start_sample:end_sample])
    if segments:
        concatenated = np.concatenate(segments)
    else:
        concatenated = np.array([], dtype=data.dtype)
    temp_path = "temp_speaker.wav"
    write(temp_path, rate, concatenated.astype(data.dtype))
    return temp_path

def extract_embedding_from_intervals(wav_path, intervals):
    """
    Extract a speaker embedding from concatenated segments.
    Args:
        wav_path (str): Path to the original audio file.
        intervals (List[Tuple[float, float]]): List of (start, end) times in seconds.
    Returns:
        np.ndarray: The embedding vector.
    """
    temp_path = concatenate_speaker_segments(wav_path, intervals)
    wav = preprocess_wav(temp_path)
    embedding = encoder.embed_utterance(wav)
    os.remove(temp_path)
    return embedding

def save_embedding(embedding, filename):
    """
    Save the embedding to disk as a .npy file in the embeddings folder.
    Args:
        embedding (np.ndarray): The embedding vector.
        filename (str): The filename (not path) to save the .npy file as.
    Returns:
        str: The full path to the saved .npy file.
    """
    save_path = os.path.join(EMBEDDING_DIR, filename)
    np.save(save_path, embedding)
    return save_path

def load_embedding(embedding_path):
    """
    Load an embedding from a .npy file.
    Args:
        embedding_path (str): Path to the .npy file.
    Returns:
        np.ndarray: The embedding vector.
    """
    return np.load(embedding_path)

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two embeddings.
    Args:
        a (np.ndarray): First embedding.
        b (np.ndarray): Second embedding.
    Returns:
        float: Cosine similarity.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_best_match(embedding, known_embeddings):
    """
    Find the best matching speaker from known embeddings using cosine similarity.
    Args:
        embedding (np.ndarray): The embedding to match.
        known_embeddings (List[Tuple]): List of (id, embedding_path, time) tuples.
    Returns:
        Tuple[str, float]: (best_match_id, similarity_score)
    """
    try:
        if not known_embeddings:
            return None, 0.0
        
        similarities = []
        for speaker_id, emb_path, _ in known_embeddings:
            known_emb = load_embedding(emb_path)
            similarity = cosine_similarity(embedding, known_emb)
            similarities.append((speaker_id, similarity))
        
        # Find the best match
        best_match = max(similarities, key=lambda x: x[1])
        return best_match[0], best_match[1]
    except Exception as e:
        logger.error(f"Error finding best match: {e}")
        return None, 0.0

def extract_embedding_from_speaker_segments(audio_path, speaker_segments):
    """
    Extract embedding from concatenated segments of a specific speaker.
    Args:
        audio_path (str): Path to the audio file.
        speaker_segments (List[dict]): List of segments for one speaker.
    Returns:
        np.ndarray: The embedding vector.
    """
    try:
        # Concatenate all segments for this speaker
        temp_path = concatenate_speaker_segments(audio_path, [(seg['start'], seg['end']) for seg in speaker_segments])
        
        # Extract embedding from concatenated segments
        wav = preprocess_wav(temp_path)
        embedding = encoder.embed_utterance(wav)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting embedding from speaker segments: {e}")
        raise 