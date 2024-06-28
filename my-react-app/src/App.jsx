import { useState } from 'react'
import './App.css'
import Chatbot from './components/Chatbot'

function App() {
  const [count, setCount] = useState(0)
  return (
    <div className="App">
      <h1>CSULB Chatbot</h1>
      <Chatbot />
    </div>
  );
}

export default App
