const express = require('express');
const cors = require('cors');
const path = require('path');

// Import Routes
const policiesRoutes = require('./routes/policies.routes');
const environmentRoutes = require('./routes/environment.routes');
const reportRoutes = require('./routes/report.routes');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Register Routes
app.use('/api/policies', policiesRoutes);
app.use('/api/environment', environmentRoutes);
app.use('/api/report', reportRoutes);

// Hello World endpoint
app.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from TabulaDrone Viewer Server!' });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`TabulaDrone Viewer Server running on http://localhost:${PORT}`);
    console.log(`Try: http://localhost:${PORT}/api/policies`);
  });
}

module.exports = app;
