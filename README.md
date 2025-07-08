# Student Voice Tracking System in Online Meets

A real-time audio processing system that tracks student participation in classroom discussions using speaker diarization, voice embeddings, and speech-to-text transcription.

## Features

- 🎤 **Real-time Audio Recording**: Continuous 10-second audio chunks
- 🎭 **Speaker Diarization**: Identifies different speakers using pyannote.audio
- 🎵 **Voice Embeddings**: Extracts unique voice signatures for each speaker
- 👥 **Speaker Recognition**: Matches speakers to known teachers and students
- 📝 **Speech Transcription**: Converts speech to text using faster-whisper
- 🔍 **Roll Number Extraction**: Automatically extracts student roll numbers from speech using regex and transformers
- 🆔 **Unknown Speaker Registration**: Handles unknown speakers by transcribing, extracting roll numbers, and registering new students
- 🏆 **Participation Leaderboard**: Tracks and ranks student participation time
- 🔧 **Audio Preprocessing**: Noise reduction and speech enhancement for better transcription

## System Architecture

```
Audio Recording → Speaker Diarization → Voice Embeddings → Speaker Matching → Transcription → Roll Number Extraction → Database Update → Leaderboard
```

## Requirements

- Python 3.8+
- Microphone access
- Internet connection (for model downloads)
- See `requirements.txt` for all Python dependencies (including `transformers` for roll number extraction)
- `sqlite3` is used for the database and is included in the Python standard library

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cyberinnovator/student_interaction_tracker.git
   cd student_interaction_tracker
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Hugging Face token:**
   - Get your token from [Hugging Face](https://huggingface.co/settings/tokens)
   - Update the token in `diarization.py` if required

## Usage

1. **Start the system:**
   ```bash
   python main.py
   ```

2. **Register teachers first** (optional):
   - The system will prompt you to add teacher voices
   - Teachers are used as reference for speaker identification

3. **System will automatically:**
   - Record 10-second audio chunks
   - Identify speakers using diarization
   - Match speakers to known students/teachers
   - Transcribe speech for new/unknown speakers
   - Extract roll numbers from speech (using regex and transformers)
   - Register new students if roll number is found
   - Update participation leaderboard

## File Structure

```
student_interaction_tracker/
├── main.py                 # Main application entry point
├── db.py                   # Database operations (SQLite)
├── diarization.py          # Speaker diarization using pyannote.audio
├── embedding.py            # Voice embedding extraction and comparison
├── audio_processing.py     # Audio preprocessing for better transcription
├── unknown_speaker.py      # Unknown speaker processing and transcription
├── rollno_extractor.py     # Roll number extraction using regex and transformers
├── student_voice_track.db  # SQLite database
├── embeddings/             # Stored voice embeddings
└── README.md               # This file
```

## Configuration

### Audio Processing
- **Preprocessing**: Gentle noise reduction and speech enhancement
- **Transcription**: CPU-optimized faster-whisper with tiny model
- **Language**: English (configurable)

### Speaker Detection
- **Diarization**: pyannote.audio 3.1
- **Embeddings**: Resemblyzer voice encoder
- **Similarity Threshold**: 0.6 (configurable)

### Database
- **Storage**: SQLite database (via Python stdlib)
- **Tables**: students, teachers
- **Data**: Roll numbers, embedding paths, participation time

## How It Works

1. **Audio Recording**: Records 10-second chunks continuously
2. **Diarization**: Separates audio into speaker segments
3. **Embedding Extraction**: Creates voice signatures for each speaker
4. **Speaker Matching**: Compares embeddings with known speakers
5. **Transcription**: Converts speech to text for unknown speakers
6. **Roll Number Extraction**: Uses regex and transformers to find roll numbers
7. **Database Update**: Saves new students and updates participation time
8. **Leaderboard**: Displays participation rankings

## Roll Number Patterns

The system recognizes these patterns in speech:
- "roll no. is 123"
- "roll number is 123"
- "role number is 123" (common transcription)
- "my roll no. is 123"
- "my role number is 123"
- "i am 123" (fallback)

## Troubleshooting

### Common Issues

1. **No speakers detected**: Check microphone permissions and audio levels
2. **Transcription errors**: Ensure clear speech and minimal background noise
3. **Roll number not extracted**: Speak clearly and use supported patterns
4. **Model download issues**: Check internet connection and Hugging Face token

### Performance Tips

- Use a quiet environment for better transcription
- Speak clearly when providing roll numbers
- Ensure stable internet for model downloads
- Close other audio applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) for speech transcription
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) for voice embeddings
- [librosa](https://librosa.org/) for audio processing 
- [transformers](https://github.com/huggingface/transformers) for roll number extraction
