const logDiscovery = require('../services/log-discovery.service');

const policiesController = {
  getPolicies: (req, res) => {
    try {
      const policies = logDiscovery.getAvailablePolicies();
      res.json({ policies });
    } catch (error) {
      res.status(500).json({
        error: 'POLICIES_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown policies loading error.'
      });
    }
  }
};

module.exports = policiesController;
