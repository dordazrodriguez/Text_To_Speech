import os
import re
import PyPDF2
from ebooklib import epub
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

device = "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device, dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

def extract_text_from_pdf(pdf_path, verbose=False):
    if verbose:
        print(f"Extracting text from PDF: {pdf_path}")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_epub(epub_path, verbose=False):
    if verbose:
        print(f"Extracting text from EPUB: {epub_path}")
    book = epub.read_epub(epub_path)
    text = ''
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            text += item.get_body_content().decode('utf-8')
    return text

def split_into_chapters(text, verbose=False):
    if verbose:
        print("Splitting text into chapters")

    # Improved regular expression to capture more chapter heading patterns
    chapter_patterns = [
        r'\n\s*(Chapter \d+|CHAPTER \d+|Chapter [IVXLCDM]+|CHAPTER [IVXLCDM]+)\s*\n',
        r'\n\s*(\d+\.\d+)\s*\n',  # Patterns like 1.1, 2.1, etc.
        r'\n\s*(\d+\s*[-–—]\s*\d+)\s*\n',  # Patterns like 1-1, 2-1, etc.
        r'\n\s*(\d+\.)\s*\n',  # Patterns like 1., 2., etc.
        r'\n\s*([IVXLCDM]+)\s*\n',  # Roman numerals
        r'\n\s*(SECTION \d+|Section \d+)\s*\n'  # Sections
    ]
    
    # Combine all patterns into a single pattern
    combined_pattern = '|'.join(chapter_patterns)
    
    # Split text based on the combined pattern
    potential_chapters = re.split(combined_pattern, text, flags=re.IGNORECASE)
    
    chapters = []
    current_chapter = []
    
    for section in potential_chapters:
        if section:  # Check if section is not None
            if len(section.split()) > 100:  # Threshold to consider a section as a chapter
                if current_chapter:
                    chapters.append(' '.join(current_chapter))
                current_chapter = [section]
            else:
                current_chapter.append(section)

    if current_chapter:
        chapters.append(' '.join(current_chapter))

    if verbose:
        print(f"Identified {len(chapters)} chapters")

    return chapters

def text_to_speech(text, description, output_path, verbose=False):
    chunk_size = 500  # Adjust based on model limits
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    audio_arr = []
    for i, chunk in enumerate(text_chunks):
        if verbose:
            print(f"Processing chunk {i + 1}/{len(text_chunks)}")
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(chunk, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
        audio_arr.extend(generation.cpu().numpy().squeeze())

    sf.write(output_path, audio_arr, model.config.sampling_rate)
    if verbose:
        print(f"Audio saved to {output_path}")

def convert_file_to_audiobook(file_path, output_path, description, verbose=False):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        text = extract_text_from_pdf(file_path, verbose)
    elif file_extension.lower() == '.epub':
        text = extract_text_from_epub(file_path, verbose)
    else:
        raise ValueError('Unsupported file format')

    chapters = split_into_chapters(text, verbose)
    for i, chapter_text in enumerate(chapters):
        chapter_output_path = f"{output_path}_chapter_{i+1}.wav"
        if verbose:
            print(f"Converting Chapter {i+1} to speech")
        text_to_speech(chapter_text, description, chapter_output_path, verbose)
        if verbose:
            print(f'Chapter {i+1} audiobook saved to {chapter_output_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert PDF or EPUB to Audiobook using Parler TTS')
    parser.add_argument('input_file', type=str, help='Path to the input PDF or EPUB file')
    parser.add_argument('output_file', type=str, help='Path to the output audiobook file(s)')
    parser.add_argument('--description', type=str, required=True, help='Description for the TTS model voice characteristics')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    convert_file_to_audiobook(args.input_file, args.output_file, args.description, args.verbose)
