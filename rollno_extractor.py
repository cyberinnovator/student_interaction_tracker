from transformers.pipelines import pipeline
import re
from typing import Optional

# Load the QA pipeline once
qa = pipeline("question-answering")

def extract_roll_number(text: str) -> Optional[str]:
    """
    Extracts the roll number from the given transcription text using a QA model.
    Returns only the digit as a string, or None if not found.
    """
    question = "What is the roll number?"
    result = qa(question=question, context=text)
    answer = result['answer'] if isinstance(result, dict) and 'answer' in result else ''
    # Extract the first number from the answer
    match = re.search(r'\d+', answer)
    if match:
        return match.group(0)
    return None
