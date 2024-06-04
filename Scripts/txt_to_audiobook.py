import os
import re
import PyPDF2
from ebooklib import epub
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
# elif torch.backends.mps.is_available():
#     device = "mps"
# elif torch.xpu.is_available():
#     # device = "xpu"
#     device = "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device, dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    text = ''
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text += item.get_body_content().decode('utf-8')
    return text

def split_into_chapters(text):
    chapters = re.split(r'\bCHAPTER\b|\bChapter\b|\bchapter\b', text)
    return ['Chapter ' + chapter.strip() for chapter in chapters if chapter.strip()]

def text_to_speech(text, description, output_path):
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_path, audio_arr, model.config.sampling_rate)

def convert_file_to_audiobook(file_path, output_path, description):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif file_extension.lower() == '.epub':
        text = extract_text_from_epub(file_path)
    else:
        raise ValueError('Unsupported file format')

    chapters = split_into_chapters(text)
    for i, chapter_text in enumerate(chapters):
        chapter_output_path = f"{output_path}_chapter_{i+1}.wav"
        text_to_speech(chapter_text, description, chapter_output_path)
        print(f'Chapter {i+1} audiobook saved to {chapter_output_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert PDF or EPUB to Audiobook using Parler TTS')
    parser.add_argument('input_file', type=str, help='Path to the input PDF or EPUB file')
    parser.add_argument('output_file', type=str, help='Path to the output audiobook file(s)')
    parser.add_argument('--description', type=str, required=True, help='Description for the TTS model voice characteristics')

    args = parser.parse_args()

    convert_file_to_audiobook(args.input_file, args.output_file, args.description)
