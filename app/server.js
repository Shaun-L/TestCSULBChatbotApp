const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();
const port = 3001;

app.use(bodyParser.json());

app.post('/chat', async (req, res) => {
    const userInput = req.body.message;
    try {
        const response = await axios.post('http://localhost:5000/chat', { message: userInput });
        res.json({ response: response.data.response });
    } catch (error) {
        res.status(500).send('Error communicating with Python backend');
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
