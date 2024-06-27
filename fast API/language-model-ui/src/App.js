import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [audioSrc, setAudioSrc] = useState('');
  const [videoSrc, setVideoSrc] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setResponse("");
    setAudioSrc("");
    setVideoSrc("");
    if (!prompt.trim()) {
      setError('The prompt is empty. Please enter some text.');
      return;
    }
    try {
      // Generate response
      const generateResponse = await axios.post('http://localhost:8000/generate', { prompt });
      const generatedText = generateResponse.data.response;
      setResponse(generatedText);

      // Convert text to speech and generate lipsync video
      const ttsResponse = await axios.post('http://localhost:8000/text-to-speech', { text: generatedText }, {
        responseType: 'blob'
      });

      const audioURL = window.URL.createObjectURL(new Blob([ttsResponse.data], { type: 'audio/wav' }));
      setAudioSrc(audioURL);

      const videoURL = window.URL.createObjectURL(new Blob([ttsResponse.data], { type: 'video/mp4' }));
      setVideoSrc(videoURL);

    } catch (error) {
      if (error.response && error.response.status === 422) {
        const errorMessage = error.response.data.detail;

        // If errorMessage is an object or array, convert it to a string
        const formattedErrorMessage = Array.isArray(errorMessage)
          ? errorMessage.map(e => e.msg).join(', ')
          : typeof errorMessage === 'object'
            ? JSON.stringify(errorMessage)
            : errorMessage;

        setError(formattedErrorMessage);

        // Convert error message to speech and generate lipsync video
        const ttsResponse = await axios.post('http://localhost:8000/text-to-speech', { text: formattedErrorMessage }, {
          responseType: 'blob'
        });
        const audioURL = window.URL.createObjectURL(new Blob([ttsResponse.data], { type: 'audio/wav' }));
        setAudioSrc(audioURL);
        const videoURL = window.URL.createObjectURL(new Blob([ttsResponse.data], { type: 'video/mp4' }));
        setVideoSrc(videoURL);
      } else {
        console.error("Error generating response or lipsync video:", error);
        setError("An error occurred while processing your request.");
      }
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Text to Avatar Speech</h1>
        <form onSubmit={handleSubmit}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your text"
            rows="4"
            cols="50"
          />
          <br />
          <button type="submit">Generate</button>
        </form>
        {error && <div style={{ color: 'red' }}>{error}</div>}
        <div>
          {response && (
            <div>
              <h2>Generated Response:</h2>
              <p>{response}</p>
            </div>
          )}
          {audioSrc && (
            <div>
              <h2>Generated Audio:</h2>
              <audio controls src={audioSrc}></audio>
            </div>
          )}
          {videoSrc && (
            <div>
              <h2>Generated Video:</h2>
              <video controls src={videoSrc} width="600" height="400"></video>
            </div>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
