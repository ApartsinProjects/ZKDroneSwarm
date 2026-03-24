const express = require('express');
const router = express.Router({ mergeParams: true });
const learningStateController = require('../controllers/learning-state.controller');

// GET /api/policies/:policyId/learning-state
router.get('/', learningStateController.getAllLearningStates);

// GET /api/policies/:policyId/learning-state/latest
router.get('/latest', learningStateController.getLatestLearningState);

module.exports = router;
