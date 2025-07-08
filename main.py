import time
import io
import sounddevice as sd
from scipy.io.wavfile import write
from collections import defaultdict
from diarization import diarize_audio
# Import embedding functions from embedding.py
from embedding import (
    extract_embedding_from_audio, 
    extract_embedding_from_segment, 
    extract_embedding_from_speaker_segments,
    save_embedding, 
    find_best_match
)
# Import unknown speaker functions from unknown_speaker.py
from unknown_speaker import (
    process_unknown_speaker,
    transcribe_full_audio
)
from db import get_all_student_embeddings, get_all_teacher_embeddings, update_student_time, add_teacher, add_student, get_student_by_roll_no, get_teacher_by_teacher_id
import math
import numpy as np
import logging
from pathlib import Path
import sqlite3
import re
import os

# Audio settings
SAMPLE_RATE = 16000  # 16 kHz, suitable for speech
CHANNELS = 1         # Mono

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Placeholder for audio capture (replace with actual audio capture logic)
def capture_audio_chunk(duration_sec=10, output_path="audio_chunk.wav"):
    try:
        print(f"[INFO] Recording {duration_sec}s of audio from microphone...")
        audio = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
        sd.wait()
        write(output_path, SAMPLE_RATE, audio)
        print(f"[INFO] Audio chunk captured: {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] Audio capture failed: {e}")
        return None

def register_teachers():
    try:
        logger.info("Starting teacher registration...")
        
        # Check if teachers already exist
        existing_teachers = get_all_teacher_embeddings()
        if existing_teachers:
            logger.info(f"Found {len(existing_teachers)} existing teachers")
            # Still ask if user wants to add more teachers
            add_more = input("Would you like to add more teachers? (y/n): ").lower().strip()
            if add_more != 'y':
                return
        
        # Interactive teacher registration
        while True:
            add = input("Would you like to add a teacher? (y/n): ").lower().strip()
            
            if add == 'n':
                logger.info("Teacher registration skipped")
                break
            elif add == 'y':
                teacher_id = input("Enter teacher ID: ").strip()
                
                if teacher_id:
                    logger.info(f"Recording teacher {teacher_id} voice...")
                    
                    # Record teacher's voice (10 seconds for better quality)
                    teacher_audio_path = record_audio_chunk(duration=10, sample_rate=16000)
                    
                    # Extract embedding
                    embedding = extract_embedding_from_audio(teacher_audio_path)
                    
                    # Save embedding
                    embedding_path = save_embedding(embedding, f"{teacher_id}.npy")
                    
                    # Add to database
                    add_teacher(teacher_id, embedding_path)
                    logger.info(f"Registered teacher: {teacher_id}")
                else:
                    logger.warning("Please enter a valid teacher ID")
                    continue
            else:
                logger.warning("Please enter 'y' or 'n'")
                continue
        
        logger.info("Teacher registration completed")
        
    except Exception as e:
        logger.error(f"Error registering teachers: {e}")

def print_leaderboard():
    try:
        logger.info("📊 Generating leaderboard...")
        students = get_all_student_embeddings()  # [(roll_no, embedding_path, time), ...]
        leaderboard = []
        for roll_no, _, total_time in students:
            points = math.floor(total_time / 5.0)
            leaderboard.append((roll_no, total_time, points))
        leaderboard.sort(key=lambda x: x[2], reverse=True)  # Sort by points descending
        
        logger.info("\n" + "=" * 60)
        logger.info("🏆 PARTICIPATION LEADERBOARD 🏆")
        logger.info("=" * 60)
        logger.info(f"{'Rank':<4} {'Roll No':<10} {'Time (s)':<12} {'Points':<8}")
        logger.info("-" * 60)
        for i, (roll_no, total_time, points) in enumerate(leaderboard[:5], 1):
            logger.info(f"{i:<4} {roll_no:<10} {total_time:<12.2f} {points:<8}")
        logger.info("=" * 60)
        logger.info(f"📈 Total students: {len(leaderboard)}")
        
    except Exception as e:
        logger.error(f"❌ Leaderboard calculation failed: {e}")

def record_audio_chunk(duration=10, sample_rate=16000):
    """Record audio for a specified duration using sounddevice."""
    try:
        logger.info(f"Recording {duration} seconds of audio...")
        
        # Record audio using sounddevice
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        
        # Save to WAV file
        write("audio_chunk.wav", sample_rate, audio)
        
        logger.info("Audio recording completed and saved to audio_chunk.wav")
        return "audio_chunk.wav"
        
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        raise



def get_leaderboard():
    """Get the participation leaderboard."""
    try:
        with sqlite3.connect('student_voice_track.db') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT roll_no, time FROM students ORDER BY time DESC')
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        return []

def main():
    """Main function to run the student voice tracking system."""
    try:
        logger.info("Starting Student Voice Tracking System")
        
        # Register teachers at startup
        register_teachers()
        
        while True:
            try:
                # Step 1: Audio input to chunks
                logger.info("🎤 Step 1: Recording audio chunk...")
                logger.info("  📝 Calling record_audio_chunk(duration=10)...")
                audio_path = record_audio_chunk(duration=10)
                logger.info(f"  ✅ Audio recorded successfully: {audio_path}")
                
                # Step 2: Speaker diarization
                logger.info("🎭 Step 2: Performing speaker diarization...")
                logger.info("  📝 Calling diarize_audio(audio_path)...")
                segments = diarize_audio(audio_path)
                logger.info(f"  ✅ Diarization completed: {len(segments)} segments found")
                
                if not segments:
                    logger.warning("No speech segments detected")
                    continue
                
                logger.info(f"Found {len(segments)} speech segments")
                
                # Step 3: Concatenate all segments of respective speakers
                logger.info("🔗 Step 3: Grouping segments by speaker...")
                speaker_groups = {}
                for segment in segments:
                    speaker = segment['speaker']
                    if speaker not in speaker_groups:
                        speaker_groups[speaker] = []
                    speaker_groups[speaker].append(segment)
                
                logger.info(f"Grouped into {len(speaker_groups)} unique speakers: {list(speaker_groups.keys())}")
                
                # Step 4: Take embedding of each speaker (from concatenated segments)
                logger.info("🎵 Step 4: Extracting embeddings from complete speaker audio...")
                speaker_embeddings = {}
                
                for speaker, speaker_segments in speaker_groups.items():
                    try:
                        # Calculate total duration for this speaker
                        total_duration = sum(seg['end'] - seg['start'] for seg in speaker_segments)
                        
                        # Skip speakers with very short total duration
                        if total_duration < 1.0:
                            logger.info(f"⏭️ Skipping speaker {speaker} (total duration: {total_duration:.2f}s < 1.0s)")
                            continue
                        
                        logger.info(f"Processing speaker {speaker} with {len(speaker_segments)} segments (total duration: {total_duration:.2f}s)")
                        
                        # Extract embedding from concatenated segments
                        logger.info(f"    📝 Calling extract_embedding_from_speaker_segments() for {speaker}...")
                        embedding = extract_embedding_from_speaker_segments(audio_path, speaker_segments)
                        logger.info(f"    ✅ Embedding extracted successfully for {speaker}")
                        speaker_embeddings[speaker] = {
                            'embedding': embedding,
                            'segments': speaker_segments,
                            'total_duration': total_duration
                        }
                        
                    except Exception as e:
                        logger.error(f"Error extracting embedding for speaker {speaker}: {e}")
                        continue
                
                # Step 5: Process each speaker
                logger.info("👥 Step 5: Processing each speaker...")
                
                for speaker, speaker_data in speaker_embeddings.items():
                    try:
                        embedding = speaker_data['embedding']
                        segments = speaker_data['segments']
                        total_duration = speaker_data['total_duration']
                        
                        logger.info(f"🔍 Analyzing speaker {speaker} (duration: {total_duration:.2f}s)")
                        
                        # Step 5a: Check if speaker is a teacher first
                        logger.info(f"  📚 Checking if speaker {speaker} is a teacher...")
                        logger.info(f"    📝 Calling get_all_teacher_embeddings()...")
                        teacher_embeddings = get_all_teacher_embeddings()
                        logger.info(f"    📝 Calling find_best_match() for teacher comparison...")
                        teacher_match, teacher_similarity = find_best_match(embedding, teacher_embeddings)
                        logger.info(f"    📊 Teacher match: {teacher_match}, similarity: {teacher_similarity:.3f}")
                        
                        if teacher_match and teacher_similarity > 0.7:
                            logger.info(f"  ✅ Teacher {teacher_match} detected (similarity: {teacher_similarity:.3f}) - skipping")
                            continue
                        
                        # Step 5b: If not a teacher, check if speaker is a known student
                        logger.info(f"  👨‍🎓 Checking if speaker {speaker} is a known student...")
                        logger.info(f"    📝 Calling get_all_student_embeddings()...")
                        student_embeddings = get_all_student_embeddings()
                        logger.info(f"    📝 Calling find_best_match() for student comparison...")
                        student_match, student_similarity = find_best_match(embedding, student_embeddings)
                        logger.info(f"    📊 Student match: {student_match}, similarity: {student_similarity:.3f}")
                        
                        if student_match and student_similarity > 0.7:
                            # Known student - update participation time
                            logger.info(f"    📝 Calling update_student_time({student_match}, {total_duration:.2f})...")
                            update_student_time(student_match, total_duration)
                            logger.info(f"    ✅ Student time updated in database")
                            logger.info(f"  ✅ Known student {student_match} participated for {total_duration:.2f}s (similarity: {student_similarity:.3f})")
                        else:
                            # Step 5c: Unknown speaker - register as new student
                            logger.info(f"  ❓ Unknown speaker {speaker} detected (best match: {student_match}, similarity: {student_similarity:.3f})")
                            logger.info(f"  🎯 Attempting to register as new student...")
                            
                            # Transcribe full audio for roll number extraction
                            logger.info(f"    📝 Calling transcribe_full_audio() for roll number extraction...")
                            full_transcription = transcribe_full_audio(audio_path)
                            logger.info(f"    📝 Full transcription: '{full_transcription}'")
                            
                            logger.info(f"    📝 Calling process_unknown_speaker() for new student registration...")
                            roll_no = process_unknown_speaker(
                                embedding, 
                                audio_path, 
                                segments, 
                                full_transcription,
                                save_embedding,
                                add_student,
                                get_student_by_roll_no
                            )
                            logger.info(f"    📊 process_unknown_speaker() returned: {roll_no}")
                            
                            if roll_no:
                                # Update participation time for newly registered student
                                logger.info(f"    📝 Calling update_student_time({roll_no}, {total_duration:.2f}) for new student...")
                                update_student_time(roll_no, total_duration)
                                logger.info(f"    ✅ New student time updated in database")
                                logger.info(f"  ✅ New student {roll_no} registered and participated for {total_duration:.2f}s")
                            else:
                                logger.warning(f"  ❌ Could not register unknown speaker {speaker} as student")
                    
                    except Exception as e:
                        logger.error(f"Error processing speaker {speaker}: {e}")
                        continue
                
                # Step 6: Update and display leaderboard
                logger.info("🏆 Step 6: Updating leaderboard...")
                logger.info("  📝 Calling get_leaderboard()...")
                leaderboard = get_leaderboard()
                logger.info(f"  📊 Leaderboard data retrieved: {len(leaderboard)} students")
                if leaderboard:
                    logger.info("\n" + "=" * 60)
                    logger.info("🏆 PARTICIPATION LEADERBOARD 🏆")
                    logger.info("=" * 60)
                    logger.info(f"{'Rank':<4} {'Roll No':<10} {'Time (s)':<12} {'Points':<8}")
                    logger.info("-" * 60)
                    for i, (roll_no, time) in enumerate(leaderboard[:5], 1):
                        points = int(time / 5.0)  # 1 point per 5 seconds
                        logger.info(f"{i:<4} {roll_no:<10} {time:<12.2f} {points:<8}")
                    logger.info("=" * 60)
                
                logger.info("✅ Processing cycle completed successfully!")
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping the system...")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                continue
    
    except Exception as e:
        logger.error(f"💥 Critical error in main function: {e}")

if __name__ == "__main__":
    main() 