import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
import base64
import torch
import whisper_timestamped
from openai import OpenAI
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from pydantic import BaseModel

OPENAI_API_KEY = ""
HUGGINGFACE_TOKEN = ""

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

class AudioRequest(BaseModel):
    fileName: str
    fileData: str  


def transcribe_audio(wav_file):
    """ Transcribe audio and force word-level timestamps using whisper-timestamped """
    model = whisper_timestamped.load_model("large")  
    result = whisper_timestamped.transcribe(model, wav_file)

    words = []
    for segment in result["segments"]:
        for word in segment["words"]:
            words.append({
                "text": word["text"],
                "start": word["start"],
                "end": word["end"]
            })

    return words

def diarize_audio(audio_path):
    """ Perform speaker diarization using Pyannote """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipeline = SpeakerDiarization.from_pretrained(
        "pyannote/speaker-diarization-3.1", 
        use_auth_token=HUGGINGFACE_TOKEN
    )
    pipeline.to(device) 

    diarization = pipeline(audio_path)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    
    return segments

def match_words_to_speakers(words, diarization):
    """Assign words to speakers using timestamp alignment and return structured output."""
    speaker_words = []
    current_speaker = None
    current_text = []
    start_timestamp = None  

    for word in words:
        word_text = word["text"]
        word_start = word["start"]
        speaker_label = "SPEAKER_00"  

        for segment in diarization:
            if segment["start"] <= word_start <= segment["end"]:
                speaker_label = segment["speaker"]
                break  

        if speaker_label != current_speaker:
            if current_speaker is not None:
                speaker_words.append({
                    "speaker": current_speaker,
                    "timestamp": start_timestamp,  
                    "text": " ".join(current_text)
                })

            
            current_speaker = speaker_label
            current_text = [word_text]
            start_timestamp = word_start  

        else:
            current_text.append(word_text)

   
    if current_speaker is not None:
        speaker_words.append({
            "speaker": current_speaker,
            "timestamp": start_timestamp,
            "text": " ".join(current_text)
        })

    return speaker_words



def process_audio(wav_file):
    """ Full pipeline: Transcribe, Diarize, and Assign Words to Speakers """
    print("Transcribing audio...")
    transcript = transcribe_audio(wav_file)
    print(transcript)
    print("Performing diarization...")
    diarization = diarize_audio(wav_file)
    print(diarization)  
    print("Matching words to speakers...")
    formatted_output = match_words_to_speakers(transcript, diarization)
    
    return formatted_output

@app.post("/transcribe/")
async def transcribe_audio_endpoint(request: AudioRequest):
    print("Processing audio...")
    print("Received request:")
    print("File Name:", request.fileName)
    print("File Data (First 100 chars):", request.fileData[:100])
    try:
        file_location = f"temp_{request.fileName}"
        print(f"Saving file to {file_location}...")
        with open(file_location, "wb") as buffer:
            buffer.write(base64.b64decode(request.fileData))
        print("File saved successfully. Processing audio...")
        result = process_audio(file_location)
        os.remove(file_location)
        print("transcription: ", result)
        return {"transcription": result}
    
    except Exception as e:
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
