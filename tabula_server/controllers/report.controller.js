const fs = require('fs');
const path = require('path');
const logDiscovery = require('../services/log-discovery.service');

const reportController = {
  getComparison: (req, res) => {
    try {
      const latestRun = logDiscovery.getLatestRunFolder();
      if (!latestRun) {
        return res.status(404).json({ error: 'NO_RUN_FOUND', message: 'No run folder found.' });
      }

      const comparisonPath = path.join(logDiscovery.LOGS_DIR, latestRun, 'report', 'comparison.json');
      if (!fs.existsSync(comparisonPath)) {
        return res.status(404).json({ error: 'NO_COMPARISON_FOUND', message: 'No comparison.json found in the latest run report.' });
      }

      const comparisonData = JSON.parse(fs.readFileSync(comparisonPath, 'utf8'));
      res.json(comparisonData);
    } catch (error) {
      res.status(500).json({
        error: 'COMPARISON_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown comparison loading error.'
      });
    }
  }
};

module.exports = reportController;
