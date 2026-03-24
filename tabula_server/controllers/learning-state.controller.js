const fs = require('fs');
const logDiscovery = require('../services/log-discovery.service');

const learningStateController = {
  getLatestLearningState: (req, res) => {
    try {
      const { policyId } = req.params;
      
      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      const candidates = logDiscovery.getFilesByContext(policyId, 'learning_state', 'learning_state_');
      const learningStatePath = logDiscovery.getLatestFile(candidates);

      if (!learningStatePath) {
        return res.status(404).json({ error: 'NO_LEARNING_STATE_FOUND', message: `No learning state artifacts found for policy: ${policyId}` });
      }

      const learningStateData = JSON.parse(fs.readFileSync(learningStatePath, 'utf8'));
      res.json(learningStateData);
    } catch (error) {
      res.status(500).json({
        error: 'LEARNING_STATE_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown learning state loading error.'
      });
    }
  }
};

module.exports = learningStateController;
