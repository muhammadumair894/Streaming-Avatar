from fastapi import FastAPI, Request, File, UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,validator
from fastapi.responses import FileResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pyngrok import ngrok
import sys
from gtts import gTTS
import cv2
import librosa
import numpy as np
import moviepy.editor as mp
import re
import soundfile as sf
# Initialize FastAPI application
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) configuration
origins = [
    "http://localhost",          # Allow requests from localhost (default React development server)
    "http://localhost:3000",     # Adjust port if your React app runs on a different port
    # Add other origins as needed for production or testing environments
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

class GenerateRequest(BaseModel):
    prompt: str

    @validator('prompt')
    def check_english(cls, value):
        if not re.match(r'^[A-Za-z0-9\s.,?!\'"-]+$', value):
            raise ValueError('Prompt must be in English.')
        return value

class TextToSpeechRequest(BaseModel):
    text: str

def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def generate_speech(response):
    custom_speaker_embedding = torch.randn(1, 512)  # Example of a random speaker embedding
    max_length = 512  # You might need to adjust this value
    truncated_response = response[:max_length]
    speech = synthesiser(truncated_response, forward_params={"speaker_embeddings": custom_speaker_embedding})
    audio_path = "speech.wav"
    sf.write(audio_path, speech["audio"], samplerate=speech["sampling_rate"])
    return audio_path

def generate_avatar_video(audio_path):
    def check_image_loaded(image, image_path):
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
        return image

    avatar_image_path = "avatar_image_reshaped.jpg"
    mouth_open_path = "mouth_image_reshaped.jpg"
    avatar_image = cv2.imread(cv2.samples.findFile(avatar_image_path))
    mouth_open_image = cv2.imread(cv2.samples.findFile(mouth_open_path))
    avatar_image = check_image_loaded(avatar_image, avatar_image_path)
    mouth_open_image = check_image_loaded(mouth_open_image, mouth_open_path)

    target_size = min(avatar_image.shape[:2], mouth_open_image.shape[:2])
    avatar_image = cv2.resize(avatar_image, (target_size[1], target_size[0]))
    mouth_open_image = cv2.resize(mouth_open_image, (target_size[1], target_size[0]))

    audio, sample_rate = librosa.load(audio_path)
    amplitude_envelope = np.abs(librosa.core.amplitude_to_db(audio))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("lipsync.mp4", fourcc, 30, (avatar_image.shape[1], avatar_image.shape[0]))

    frame_duration = 1.0 / 30
    total_frames = int(len(audio) / (sample_rate * frame_duration))

    for i in range(total_frames):
        start_sample = int(i * sample_rate * frame_duration)
        end_sample = int((i + 1) * sample_rate * frame_duration)
        frame_audio = audio[start_sample:end_sample]

        if np.mean(np.abs(frame_audio)) > 0.025:
            frame = mouth_open_image
        else:
            frame = avatar_image

        video_writer.write(frame)

    video_writer.release()
    video = mp.VideoFileClip("lipsync.mp4")
    audio = mp.AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video_path = "lipsync_with_audio.mp4"
    final_video.write_videofile(final_video_path, codec='libx264')

    return final_video_path

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        response = generate_response(request.prompt)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/text-to-speech")
async def text_to_speech_endpoint(request: TextToSpeechRequest):
    try:
        audio_path = generate_speech(request.text)
        video_path = generate_avatar_video(audio_path)
        return FileResponse(video_path, media_type='video/mp4', filename='lipsync_with_audio.mp4')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
# Main block to run FastAPI server                       
if __name__ == "__main__":
    import uvicorn

    # Set ngrok authentication token (if needed)
    ngrok.set_auth_token("2iB9rF3URCDowzEjh53v30QY9nJ_7mkqf51DD9YMBs8Laf44N")
    # Create ngrok tunnel to expose the server
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)

    # Use uvicorn.run only when not in a Jupyter Notebook environment
    if 'ipykernel' not in sys.modules:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        import nest_asyncio
        nest_asyncio.apply()  # Apply nest_asyncio to allow nested event loops
        uvicorn.run(app, host="0.0.0.0", port=8000)
