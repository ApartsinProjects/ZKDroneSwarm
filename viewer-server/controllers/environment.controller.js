const logDiscovery = require('../services/log-discovery.service');
const fs = require('fs');

const environmentController = {
  getEnvironment: (req, res) => {
    try {
      const environmentPath = logDiscovery.getEnvironmentFile();
      
      if (!environmentPath) {
        return res.status(404).json({ error: 'NO_ENVIRONMENT_FOUND', message: 'No environment.json found in the latest run.' });
      }

      const environmentData = JSON.parse(fs.readFileSync(environmentPath, 'utf8'));
      res.json(environmentData);
    } catch (error) {
      res.status(500).json({
        error: 'ENVIRONMENT_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown environment loading error.'
      });
    }
  }
};

module.exports = environmentController;
