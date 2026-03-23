const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Hello World endpoint
app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from TabulaDrone Viewer Server!' });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`TabulaDrone Viewer Server running on http://localhost:${PORT}`);
  console.log(`Try: http://localhost:${PORT}/api/hello`);
});
