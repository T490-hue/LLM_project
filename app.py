#source new_env/bin/activate


import streamlit as st
import whisper
import torch
import requests
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    BartForConditionalGeneration, 
    BartTokenizer
)
from IndicTransToolkit import IndicProcessor
import os
import google.generativeai as genai

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Notes Generation from Video Lectures",
    page_icon="📚",
    layout="centered"
)

# Force CPU usage if you don't have a CUDA GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create a directory for temporary files
if not os.path.exists("temp"):
    os.makedirs("temp")

def get_gemini_response(api_key, prompt):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "contents": [{
                "parts":[{"text": prompt}]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            # Parse the response JSON
            response_data = response.json()
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                return response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                return "No content generated"
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        st.error(f"An error occurred with Gemini: {str(e)}")
        return f"An error occurred: {str(e)}"

@st.cache_resource
def load_summarization_model():
    """Load and cache the BART model and tokenizer"""
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

def summarize_text(text, model, tokenizer):
    """
    Summarize English text using BART with improved error handling
    Returns original text if summarization fails or if text is too short
    """
    if not text or len(text.split()) < 30:  # Don't summarize very short texts
        return text
    
    try:
        # Prepare the input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        # Generate summary
        summary_ids = model.generate(
            **inputs,
            max_length=200,
            min_length=150,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Return original text if summary is too short or empty
        if not summary or len(summary.split()) < 3:
            return text
            
        return summary
        
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return text  # Return original text if summarization fails

# Load the Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base").to(device)

# Initialize translation variables
BATCH_SIZE = 2  # Reduced batch size for CPU

# Initialize model and tokenizer for translation
@st.cache_resource
def initialize_model_and_tokenizer(ckpt_dir):
    print(f"Loading model from {ckpt_dir}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model

# Translation function
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return translations

# Initialize translation models and processor
@st.cache_resource
def initialize_processor():
    return IndicProcessor(inference=True)

# Initialize models and processor
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir)
ip = initialize_processor()
model = load_whisper_model()
summarization_model, summarization_tokenizer = load_summarization_model()

if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None
if 'translation' not in st.session_state:
    st.session_state.translation = None

# Streamlit UI
st.title("🎥 Notes Generation from Video Lectures")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, etc.)", type=["mp3", "wav", "m4a"])

# Create containers for each output section
transcription_container = st.container()
summary_container = st.container()
explanation_container = st.container()
translation_container = st.container()

if uploaded_file is not None:
    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing audio... Please wait."):
            temp_path = os.path.join("temp", "temp_audio.mp3")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Transcribe and store in session state
            st.session_state.transcription = model.transcribe(temp_path)
            os.remove(temp_path)

# Display transcription if available
with transcription_container:
    if st.session_state.transcription is not None:
        st.markdown("### 📝 Transcribed Notes:")
        st.text_area("Generated Notes:", st.session_state.transcription["text"], height=300)

# Add Summarize button
if st.button("Summarize Text"):
    if st.session_state.transcription is not None:
        with st.spinner("Generating summary... Please wait."):
            try:
                original_text = st.session_state.transcription["text"]
                st.session_state.summary = summarize_text(original_text, summarization_model, summarization_tokenizer)
            except Exception as e:
                st.error("Summarization failed. Showing original text.")
                st.session_state.summary = st.session_state.transcription["text"]

# Display summary if available
with summary_container:
    if st.session_state.summary is not None:
        st.markdown("### 📌 Summary:")
        st.text_area("Generated Summary:", st.session_state.summary, height=200)

# Add Explain button
if st.button("Explain"):
    if st.session_state.summary is not None:
        with st.spinner("Getting explanation... Please wait."):
            api_key = "AIzaSyDcSfnVoYStW61KSwbsvnFPhuZNbDvDRoo"  # Using your working API key
            prompt = f"Explain the following text in detail, breaking down the main concepts: {st.session_state.summary}"
            
            explanation = get_gemini_response(api_key, prompt)
            if not explanation.startswith("An error occurred"):
                st.session_state.explanation = explanation
            else:
                st.error(explanation)

# Display explanation if available
with explanation_container:
    if st.session_state.explanation is not None:
        st.markdown("### 🔍 Detailed Explanation:")
        st.text_area("AI Explanation:", st.session_state.explanation, height=300)

# Language selection and translation
lang_options = {
    "Hindi": "hin_Deva",
    "Malayalam": "mal_Mlym",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
}

selected_language = st.selectbox("Translate to:", list(lang_options.keys()))

if st.button("Translate Text"):
    if st.session_state.transcription is not None:
        with st.spinner(f"Translating to {selected_language}..."):
            tgt_lang = lang_options[selected_language]
            src_lang = "eng_Latn"
            
            try:
                translations = batch_translate(
                    [st.session_state.transcription["text"]], 
                    src_lang, 
                    tgt_lang, 
                    en_indic_model, 
                    en_indic_tokenizer, 
                    ip
                )
                
                if translations:
                    st.session_state.translation = translations[0]
                else:
                    st.session_state.translation = "Translation failed. Please try again."
            except Exception as e:
                st.session_state.translation = f"Translation failed: {str(e)}"

# Display translation if available
with translation_container:
    if st.session_state.translation is not None:
        st.markdown(f"### 🌐 Translated Notes in {selected_language}:")
        st.text_area("Translated Notes:", st.session_state.translation, height=300)
