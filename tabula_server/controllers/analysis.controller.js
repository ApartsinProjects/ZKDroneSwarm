const fs = require('fs');
const logDiscovery = require('../services/log-discovery.service');

const analysisController = {
  getLatestAnalysis: (req, res) => {
    try {
      const { policyId } = req.params;
      
      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      const candidates = logDiscovery.getFilesByContext(policyId, 'analysis', 'analysis_');
      const analysisPath = logDiscovery.getLatestFile(candidates);

      if (!analysisPath) {
        return res.status(404).json({ error: 'NO_ANALYSIS_FOUND', message: `No analysis artifacts found for policy: ${policyId}` });
      }

      const analysisData = JSON.parse(fs.readFileSync(analysisPath, 'utf8'));
      res.json(analysisData);
    } catch (error) {
      res.status(500).json({
        error: 'ANALYSIS_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown analysis loading error.'
      });
    }
  }
};

module.exports = analysisController;
