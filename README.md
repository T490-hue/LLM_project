# Notes Generation from Video Lectures

## Overview
This project utilizes **Streamlit**, **Whisper**, **BART**, and **IndicTrans** to generate structured notes from video lectures. The pipeline involves **speech-to-text transcription**, **text summarization**, **detailed explanation**, and **language translation** for better accessibility.

## Features
- **Transcription**: Converts audio from video lectures into text using OpenAI's **Whisper**.
- **Summarization**: Extracts key points using **BART**.
- **Explanation**: Provides detailed explanations using **Google Gemini AI**.
- **Translation**: Translates notes into multiple Indian languages using **IndicTrans**.
- **Interactive UI**: Built with **Streamlit** for easy use.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- PyTorch
- Streamlit
- Transformers
- Whisper
- IndicTransToolkit
- Google Generative AI API


### Create Virtual Environment 
```sh
python -m venv new_env
source new_env/bin/activate  # macOS/Linux
new_env\Scripts\activate    # Windows
```

### Install Dependencies
```sh
pip install torch torchvision streamlit whisper transformers indictrans-toolkit google-generativeai
```

## Usage
### Run the Application
```sh
streamlit run app.py
```
### How to Use
1. Upload an **audio file** (MP3, WAV, M4A) in the Streamlit UI.
2. Click **Transcribe Audio** to generate **text notes**.
3. Click **Summarize Text** to extract key points.
4. Click **Explain** to get a detailed breakdown of the content.
5. Select a language and click **Translate Text** for multilingual support.

## Supported Languages for Translation
- Hindi
- Malayalam
- Kannada
- Tamil
## Results
!(images/1.png)
!(images/2.png)


