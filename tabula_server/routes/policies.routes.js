const express = require('express');
const router = express.Router();
const policiesController = require('../controllers/policies.controller');

// Import context routers
const episodesRoutes = require('./episodes.routes');
const analysisRoutes = require('./analysis.routes');
const learningStateRoutes = require('./learning-state.routes');

// GET /api/policies
router.get('/', policiesController.getPolicies);

// Mount context routes under /api/policies/:policyId
router.use('/:policyId/episodes', episodesRoutes);
router.use('/:policyId/analysis', analysisRoutes);
router.use('/:policyId/learning-state', learningStateRoutes);

module.exports = router;
