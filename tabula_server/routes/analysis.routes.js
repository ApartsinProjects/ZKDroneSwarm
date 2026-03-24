const express = require('express');
const router = express.Router({ mergeParams: true });
const analysisController = require('../controllers/analysis.controller');

// GET /api/policies/:policyId/analysis/latest
router.get('/latest', analysisController.getLatestAnalysis);

module.exports = router;
