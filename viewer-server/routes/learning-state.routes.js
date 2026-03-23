const express = require('express');
const router = express.Router({ mergeParams: true });
const learningStateController = require('../controllers/learning-state.controller');

// GET /api/learning-state/latest
router.get('/latest', learningStateController.getLatestLearningState);

module.exports = router;
