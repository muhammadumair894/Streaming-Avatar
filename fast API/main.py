from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pyngrok import ngrok
import sys
from gtts import gTTS
import cv2
import librosa
import numpy as np
import moviepy.editor as mp
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

# Define request body structure using Pydantic BaseModel
class PromptRequest(BaseModel):
    prompt: str

# Endpoint to generate response
@app.post("/generate")
async def generate_response(request: PromptRequest):
    prompt = request.prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {"response": response}
# Define request body structure for text-to-speech
class TextToSpeechRequest(BaseModel):
    text: str

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    audio_path = "response.mp3"
    tts.save(audio_path)
    return audio_path

# Endpoint to convert text to speech
@app.post("/text-to-speech")
async def text_to_speech_endpoint(request: TextToSpeechRequest):
    audio_path = text_to_speech(request.text)
    return FileResponse(audio_path, media_type='audio/mpeg', filename="response.mp3")

# Function to check if an image is loaded correctly
def check_image_loaded(image, image_path):
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    return image

# Function to generate lipsynced video with audio
def generate_lipsync_video(audio_path):
    # Load the images
    avatar_image_path = "avatar_image_reshaped.jpg"
    mouth_open_path = "mouth_image_reshaped.jpg"

    avatar_image = cv2.imread(cv2.samples.findFile(avatar_image_path))
    mouth_open_image = cv2.imread(cv2.samples.findFile(mouth_open_path))

    avatar_image = check_image_loaded(avatar_image, avatar_image_path)
    mouth_open_image = check_image_loaded(mouth_open_image, mouth_open_path)

    # Ensure both images are the same size
    target_size = min(avatar_image.shape[:2], mouth_open_image.shape[:2])
    avatar_image = cv2.resize(avatar_image, (target_size[1], target_size[0]))
    mouth_open_image = cv2.resize(mouth_open_image, (target_size[1], target_size[0]))

    # Load the audio file
    audio, sample_rate = librosa.load(audio_path)

    # Extract the amplitude envelope
    amplitude_envelope = np.abs(librosa.core.amplitude_to_db(audio))

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("lipsync.mp4", fourcc, 30, (avatar_image.shape[1], avatar_image.shape[0]))

    # Iterate over the audio frames and update the avatar's mouth shape
    frame_duration = 1.0 / 30  # Duration of each video frame
    total_frames = int(len(audio) / (sample_rate * frame_duration))

    for i in range(total_frames):
        start_sample = int(i * sample_rate * frame_duration)
        end_sample = int((i + 1) * sample_rate * frame_duration)
        frame_audio = audio[start_sample:end_sample]

        # Determine the mouth shape based on the amplitude
        if np.mean(np.abs(frame_audio)) > 0.025:  # Adjust the threshold as needed
            frame = mouth_open_image
        else:
            frame = avatar_image

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    # Add audio to the video
    video = mp.VideoFileClip("lipsync.mp4")
    audio = mp.AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile("lipsync_with_audio.mp4", codec='libx264')

    return "lipsync_with_audio.mp4"

# Endpoint to create lipsync video
@app.post("/lipsync")
async def lipsync_endpoint(file: UploadFile = File(...)):
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as audio_file:
        audio_file.write(await file.read())
    
    video_path = generate_lipsync_video(audio_path)
    return FileResponse(video_path, media_type='video/mp4', filename="lipsync_with_audio.mp4")

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
