# Text-To-Speech with Python

This repository contains Python scripts to convert any PDF or EPUB file into an audiobook.

## Using gTTS (Google Text-to-Speech):

To use the gTTS (Google Text-to-Speech) script, you'll need to install the required packages:

```bash
pip install gtts PyPDF2 EbookLib SoundFile
```

Then, you can run the script with the following command:

```bash
python ./Scripts/simple_script_gtts.py "path/to/input.pdf" "path/to/output_folder" --verbose
```

## Using Parler-TTS:

The Parler-TTS model from Hugging Face requires text input along with a description prompt to adjust the voice settings.

Check out the [Hugging Face Demo](https://huggingface.co/spaces/parler-tts/parler-tts-expresso) to customize your voice.

You can run the Parler-TTS script as follows:

```bash
python ./Scripts/simple_script_parler-tts.py "path/to/input.pdf" "path/to/output_folder" --description="A female speaker that delivers her words clearly and easy to understand. High quality." --verbose
```

Feel free to explore and experiment with these scripts to create personalized audiobooks!
