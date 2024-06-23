import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [audioSrc, setAudioSrc] = useState(null);
  const [videoSrc, setVideoSrc] = useState(null);

  // Function to handle generating response from input text
  const generateResponse = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/generate', {
        prompt: inputText
      });

      setOutputText(response.data.response);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  // Function to convert generated text to speech
  const generateSpeech = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/text-to-speech', {
        text: outputText
      }, {
        responseType: 'blob'  // Important to get the response as a blob
      });
      const audioURL = URL.createObjectURL(response.data);
      setAudioSrc(audioURL);
    } catch (error) {
      console.error('Error generating speech:', error);
    }
  };

  // Function to generate lipsync video
  const generateLipsyncVideo = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:8000/lipsync', {
        // Optionally send any other data needed
      }, {
        responseType: 'blob'  // Important to get the response as a blob
      });
      const videoURL = URL.createObjectURL(response.data);
      setVideoSrc(videoURL);
    } catch (error) {
      console.error('Error generating lipsync video:', error);
    }
  };

  return (
    <div className="App">
      <h1>Language Model UI</h1>
      <div className="input-group">
        <label htmlFor="inputText">Input Text:</label>
        <textarea
          id="inputText"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter your prompt here..."
          required
        />
      </div>
      <button className="btn" onClick={generateResponse}>Generate Response</button>
      <div className="output-group">
        <label htmlFor="outputText">Generated Response:</label>
        <textarea
          id="outputText"
          value={outputText}
          placeholder="Generated response will appear here..."
          readOnly
        />
      </div>
      {outputText && (
        <div>
          <button className="btn" onClick={generateSpeech}>Convert to Speech</button>
        </div>
      )}
      {audioSrc && (
        <div>
          <h3>Generated Speech:</h3>
          <audio controls src={audioSrc}></audio>
          <button className="btn" onClick={generateLipsyncVideo}>Generate Lipsync Video</button>
        </div>
      )}
      {videoSrc && (
        <div>
          <h3>Generated Lipsync Video:</h3>
          <video controls src={videoSrc}></video>
        </div>
      )}
    </div>
  );
}

export default App;
