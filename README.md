# Multimodal-Emotion-Recognition
# This project aims to develop a comprehensive system for multimodal emotion recognition and impression analysis by leveraging advanced machine learning and deep learning techniques.
# Install required libraries
!pip install tensorflow opencv-python keras numpy matplotlib librosa SpeechRecognition transformers torch pydub scikit-learn openai-whisper datasets evaluate torchaudio tabulate
!pip install silero-vad  # Install the Silero VAD package

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from google.colab.patches import cv2_imshow
from collections import Counter
import librosa
import whisper  # Import Whisper for speech-to-text
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment
import os
from google.colab import files
import torch
from sklearn.metrics import accuracy_score
from datasets import Dataset  # Import Dataset from datasets
import evaluate  # Import evaluate for loading metrics
import torchaudio
from IPython import display as ipd
from moviepy.editor import VideoFileClip
from silero_vad import get_speech_timestamps, read_audio  # Import Silero VAD functions
import pandas as pd  # For tabular output
from tabulate import tabulate  # For formatted output
import re  # For regex-based sentence splitting

# Step 3: Upload the video file
uploaded = files.upload()
video_path = list(uploaded.keys())[0]  # Get the name of the uploaded file

# Step 4: Extract audio from the video file
def extract_audio_from_video(video_path):
    """
    Extract audio from the video file and save it as a WAV file.
    """
    try:
        video = AudioSegment.from_file(video_path)
        audio = video.set_channels(1).set_frame_rate(16000)  # Convert to mono and 16kHz
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

# Step 5: Audio Normalization (Fine-Tuned)
def normalize_audio(audio_path, target_dBFS=-20.0):
    """
    Normalize the audio file to a target dBFS level.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        normalized_audio_path = os.path.splitext(audio_path)[0] + "_normalized.wav"
        normalized_audio.export(normalized_audio_path, format="wav")
        return normalized_audio_path
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return None

# Step 6: Analyze Acoustic Features for Emotion Recognition (Using a More Powerful Fine-Tuned Model)
def analyze_acoustic_emotion(audio_path):
    """
    Analyze emotion from audio using a more powerful fine-tuned Wav2Vec 2.0 model for emotion recognition.
    """
    try:
        # Load the pre-trained fine-tuned emotion recognition model
        model_name = "superb/wav2vec2-large-superb-er"  # More powerful model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        # Load and preprocess the audio
        y, sr = librosa.load(audio_path, sr=16000)
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)

        # Predict emotion
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        # Map class index to emotion label
        emotion_labels = ["angry", "happy", "sad", "neutral"]
        emotion = emotion_labels[predicted_class]

        return emotion
    except Exception as e:
        print(f"Error analyzing acoustic emotion: {e}")
        return None

# Step 7: Speech-to-Text using Whisper (Enhanced)
def transcribe_audio(audio_path):
    """
    Transcribe the audio file to text using OpenAI's Whisper model.
    """
    try:
        # Load the Whisper model
        model = whisper.load_model("medium")  # Use a larger model for better accuracy
        result = model.transcribe(audio_path)
        return result["text"], result["segments"]  # Return both text and segments
    except Exception as e:
        print(f"Error transcribing audio with Whisper: {e}")
        return None, None

# Step 8: Text-Based Emotion Analysis using Hugging Face Transformers
def analyze_text_emotion(text):
    """
    Analyze emotion from text using a pre-trained RoBERTa model.
    """
    try:
        # Load the emotion analysis pipeline
        emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        result = emotion_classifier(text)
        return result[0]['label']  # Return the dominant emotion
    except Exception as e:
        print(f"Error analyzing text emotion: {e}")
        return None

# Step 9: Video-Based Emotion Detection
def process_video(video_path, start_time, end_time):
    """
    Process the video file to detect emotions from facial expressions for a specific segment.
    """
    !wget https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5
    model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    frame_skip = 20
    frame_count = 0
    emotion_list = []

    # Set the start and end time for the video segment
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    end_time_ms = end_time * 1000

    while True:
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) >= end_time_ms:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi.astype("float") / 255.0
            face_roi = img_to_array(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)

            preds = model.predict(face_roi)[0]
            emotion = emotion_labels[np.argmax(preds)]
            emotion_list.append(emotion)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2_imshow(frame)

    cap.release()

    if emotion_list:
        overall_emotion = Counter(emotion_list).most_common(1)[0][0]
        return overall_emotion
    else:
        print("No emotions detected in the video.")
        return None

# Step 10: Segment Audio into Sentences using Whisper Segments
def segment_audio_into_sentences(audio_path, segments):
    """
    Segment the audio file into sentences using Whisper's timestamps.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)

        # Save and analyze each sentence
        segment_paths = []
        for i, segment in enumerate(segments):
            start_time = int(segment['start'] * 1000)  # Convert to milliseconds
            end_time = int(segment['end'] * 1000)  # Convert to milliseconds
            segment_audio = audio[start_time:end_time]
            segment_path = f"segment_{i+1}.wav"
            segment_audio.export(segment_path, format="wav")
            segment_paths.append(segment_path)

        return segment_paths
    except Exception as e:
        print(f"Error segmenting audio into sentences: {e}")
        return None

# Step 11: Calculate Impression Count and Cumulative Impression Count
def calculate_impression_counts(text_emotion, audio_emotion, video_emotion):
    """
    Calculate impression counts and cumulative impression count based on the given rules.
    """
    # Map emotions to impression counts
    impression_map = {
        # Text emotions (from j-hartmann/emotion-english-distilroberta-base)
        "anger": -1,
        "disgust": -1,
        "fear": -1,
        "joy": 1,
        "neutral": 0,
        "sadness": -1,
        "surprise": 0,

        # Audio emotions (from Wav2Vec2 model)
        "angry": -1,
        "happy": 1,
        "sad": -1,
        "neutral": 0,

        # Video emotions (from FER2013 model)
        "Angry": -1,
        "Disgust": -1,
        "Fear": -1,
        "Happy": 1,
        "Sad": -1,
        "Surprise": 0,
        "Neutral": 0
    }

    # Debugging: Print the emotions being passed
    print(f"Debug - Text Emotion: {text_emotion}, Audio Emotion: {audio_emotion}, Video Emotion: {video_emotion}")

    # Initialize impression counts
    text_impression = impression_map.get(text_emotion, "NULL") if text_emotion != "unknown" else "NULL"
    audio_impression = impression_map.get(audio_emotion, "NULL") if audio_emotion != "unknown" else "NULL"
    video_impression = impression_map.get(video_emotion, "NULL") if video_emotion != "unknown" else "NULL"

    # Calculate cumulative impression count
    cumulative_impression = 0
    if text_impression != "NULL":
        cumulative_impression += text_impression
    if audio_impression != "NULL":
        cumulative_impression += audio_impression
    if video_impression != "NULL":
        cumulative_impression += video_impression

    # Determine overall impression
    if cumulative_impression < 0:
        overall_impression = "negative impression"
    elif cumulative_impression > 0:
        overall_impression = "positive impression"
    else:
        overall_impression = "neutral impression"

    return text_impression, audio_impression, video_impression, cumulative_impression, overall_impression
# Step 12: Main Execution (Fully Updated and Modified)
try:
    print("Extracting audio from the video...")
    audio_path = extract_audio_from_video(video_path)
    if not audio_path:
        raise Exception("Audio extraction failed.")

    print("\nNormalizing audio...")
    normalized_audio_path = normalize_audio(audio_path, target_dBFS=-20.0)  # Fine-tuned normalization
    if not normalized_audio_path:
        raise Exception("Audio normalization failed.")

    print("\nPlaying normalized audio:")
    ipd.display(ipd.Audio(normalized_audio_path))

    print("\nTranscribing audio to text using Whisper...")
    text, segments = transcribe_audio(normalized_audio_path)
    if text:
        print(f"Transcribed Text: {text}")
        print(f"\n{len(segments)} sentences have been identified based on Whisper segments.")

        # Initialize lists to store results for each sentence
        results = []
        analysis_count = 0  # Initialize analysis count

        # Process each Whisper segment
        for i, segment in enumerate(segments):
            print(f"\nProcessing Sentence {i+1}:")
            sentence_text = segment['text'].strip()  # Get the sentence text from Whisper
            print(f"Sentence Text: {sentence_text}")

            # Analyze text-based emotion
            text_emotion = analyze_text_emotion(sentence_text)
            if not text_emotion:
                print("Text emotion analysis failed. Skipping this step for the segment.")
                text_emotion = "unknown"  # Assign a default value

            # Segment audio for the sentence
            segment_paths = segment_audio_into_sentences(normalized_audio_path, [segment])
            if not segment_paths:
                print("Audio segmentation failed. Skipping this step for the segment.")
                audio_emotion = "unknown"  # Assign a default value
            else:
                segment_path = segment_paths[0]
                print(f"Segment Audio Path: {segment_path}")

                # Normalize the segmented audio
                normalized_segment_path = normalize_audio(segment_path, target_dBFS=-20.0)
                if not normalized_segment_path:
                    print("Audio normalization failed. Skipping this step for the segment.")
                    audio_emotion = "unknown"  # Assign a default value
                else:
                    print(f"Normalized Segment Audio Path: {normalized_segment_path}")

                    # Play the normalized segmented audio
                    print("Playing normalized segmented audio:")
                    ipd.display(ipd.Audio(normalized_segment_path))

                    # Analyze audio-based emotion
                    audio_emotion = analyze_acoustic_emotion(normalized_segment_path)
                    if not audio_emotion:
                        print("Audio emotion analysis failed. Skipping this step for the segment.")
                        audio_emotion = "unknown"  # Assign a default value

            # Extract video segment for the sentence
            start_time = segment['start']
            end_time = segment['end']
            print(f"Video Segment: {start_time}s to {end_time}s")

            # Analyze video-based emotion
            video_emotion = process_video(video_path, start_time, end_time)
            if not video_emotion:
                print("Video emotion analysis failed. Skipping this step for the segment.")
                video_emotion = "unknown"  # Assign a default value

            # Calculate impression counts and cumulative impression count
            text_impression, audio_impression, video_impression, cumulative_impression, overall_impression = calculate_impression_counts(
                text_emotion, audio_emotion, video_emotion
            )

            print(f"Impression Counts: Text={text_impression}, Audio={audio_impression}, Video={video_impression}")
            print(f"Cumulative Impression Count: {cumulative_impression}")
            print(f"Overall Impression: {overall_impression}")

            # Increment analysis count
            analysis_count += 1
            print(f"Analysis count {analysis_count} completed.")

            # Store results for the sentence
            results.append({
                "Serial Number": analysis_count,  # Chronological serial number
                "Sentence": sentence_text,  # Use the actual sentence text
                "Text Emotion": text_emotion,
                "Audio Emotion": audio_emotion,
                "Video Emotion": video_emotion,
                "Text Impression": text_impression,
                "Audio Impression": audio_impression,
                "Video Impression": video_impression,
                "Cumulative Impression": cumulative_impression,
                "Overall Impression": overall_impression
            })

        # Final overall emotion and impression
        if results:
            # Create a DataFrame for the results
            df = pd.DataFrame(results)

            # Calculate Ultimate Impression
            overall_impressions = df['Overall Impression'].tolist()
            valid_impressions = [imp for imp in overall_impressions if imp != "can't be determined"]

            if valid_impressions:
                # Count the frequency of each impression
                impression_counts = Counter(valid_impressions)
                max_count = max(impression_counts.values())
                modes = [imp for imp, count in impression_counts.items() if count == max_count]

                if len(modes) == 1:
                    ultimate_impression = modes[0]
                else:
                    # Calculate cumulative result for tied modes
                    cumulative_result = 0
                    for imp in valid_impressions:
                        if imp == "positive impression":
                            cumulative_result += 1
                        elif imp == "negative impression":
                            cumulative_result -= 1
                        elif imp == "neutral impression":
                            cumulative_result += 0

                    if cumulative_result > 0:
                        ultimate_impression = "positive impression"
                    elif cumulative_result < 0:
                        ultimate_impression = "negative impression"
                    else:
                        ultimate_impression = "neutral impression"
            else:
                ultimate_impression = "can't be determined, more data needed"

            # Add Ultimate Impression as a new row in the DataFrame
            ultimate_row = pd.DataFrame({
                "Serial Number": ["Ultimate Impression"],
                "Sentence": [""],
                "Text Emotion": [""],
                "Audio Emotion": [""],
                "Video Emotion": [""],
                "Text Impression": [""],
                "Audio Impression": [""],
                "Video Impression": [""],
                "Cumulative Impression": [""],
                "Overall Impression": [ultimate_impression]
            })

            # Concatenate the new row to the DataFrame
            df = pd.concat([df, ultimate_row], ignore_index=True)

            # Print the final comprehensive results in a tabular format with dotted lines
            print("\nFinal Comprehensive Results:")
            print(tabulate(df, headers="keys", tablefmt="grid"))

            # Verify that the analysis count and serial number match the number of sentences
            if analysis_count == len(segments):
                print(f"\nVerification: Analysis count ({analysis_count}) and serial number ({analysis_count}) both match the number of sentences ({len(segments)}).")
            else:
                print(f"\nVerification failed: Analysis count ({analysis_count}) or serial number ({analysis_count}) does not match the number of sentences ({len(segments)}).")
        else:
            print("No overall emotions detected.")
    else:
        print("No text transcribed. Please check the audio file.")

except Exception as e:
    print(f"Error in main execution: {e}")
