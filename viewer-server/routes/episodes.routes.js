const express = require('express');
const router = express.Router({ mergeParams: true });
const episodesController = require('../controllers/episodes.controller');

// GET /api/episodes/latest
router.get('/latest', episodesController.getMapScene);

module.exports = router;
