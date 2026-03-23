const express = require('express');
const router = express.Router({ mergeParams: true });
const episodesController = require('../controllers/episodes.controller');

// GET /api/policies/:policyId/episodes
router.get('/', episodesController.getAllEpisodes);

// GET /api/policies/:policyId/episodes/best
router.get('/best', episodesController.getBestEpisode);

// GET /api/policies/:policyId/episodes/latest
router.get('/latest', episodesController.getMapScene);

module.exports = router;
