import axios from 'axios';

const API_URL = 'http://localhost:5000';

export const sendQuery = async (query) => {
  const response = await axios.post(`${API_URL}/query`, { query });
  return response.data;
};

export const createDatabase = async () => {
  const response = await axios.post(`${API_URL}/create-db`);
  return response.data;
};
