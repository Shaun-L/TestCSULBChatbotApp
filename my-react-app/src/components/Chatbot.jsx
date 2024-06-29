import React, { useState } from 'react';
import axios from 'axios';
import './Chatbot.css';

function Chatbot() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [dbCreated, setDbCreated] = useState(false);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    try {
      setLoading(true);
      const response = await axios.post('http://127.0.0.1:5000/query', { query: input });
      setMessages([...messages, { user: input, bot: response.data.response }]);
      setInput('');
      setLoading(false);
    } catch (error) {
      console.error('Error querying the chatbot:', error);
      alert('Failed to get a response from the chatbot.');
      setLoading(false);
    }
  };

  const createDatabase = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/create-db');
      setDbCreated(true);
      alert('Database created successfully');
    } catch (error) {
      console.error('Error creating the database:', error);
      alert('Failed to create the database.');
    }
  };

  return (
    <div>
      <div className="chat-container">
        {messages.map((msg, index) => (
          <div key={index} className="message">
            <p><strong>You:</strong> {msg.user}</p>
            <p><strong>Bot:</strong> {msg.bot}</p>
          </div>
        ))}
      </div>
      {loading ? (
        <div className="loading">Loading...</div>
      ) : (
        <div>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message here..."
          />
          <button onClick={sendMessage}>Send</button>
        </div>
      )}
      {!dbCreated && <button onClick={createDatabase}>Create Database</button>}
    </div>
  );
}

export default Chatbot;
