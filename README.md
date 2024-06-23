# Streaming-Avatar
This project provides a user interface for interacting with a language model. It includes features for generating text responses from user input, converting the generated text to speech, and creating lipsync videos.

Features
Generate Response: Generate a textual response based on the user's input prompt.
Text to Speech: Convert the generated text response into speech.
Lipsync Video: Create a lipsync video from the generated speech.
Installation
To run this project locally, follow these steps:

Prerequisites
Node.js (for running the React app)
Python (for running the backend server)
axios (for making HTTP requests in the React app)

Backend Setup
1. Install Dependencies:
   pip install fastapi pydantic transformers torch pyngrok gtts opencv-python-headless librosa numpy moviepy uvicorn nest-asyncio
2. Start the backend server:
   python main.py
3. Run File:
   python main.py
   
Frontend Setup
1. Create React Application:
   npx create-react-app language-model-ui
   cd language-model-ui
2. Dependencies
    npm install react axios
3. Run file:
   npm start

Usage
Enter your text prompt in the input field.
Click on "Generate Response" to get the text response from the language model.
Click on "Convert to Speech" to generate speech from the text response.
Click on "Generate Lipsync Video" to create a lipsync video from the generated speech.
Project Structure
frontend: Contains the React app.
App.js: Main component for the user interface.
backend: Contains the backend server code.
main.py: Main server file handling requests for text generation, text-to-speech, and lipsync video creation.
