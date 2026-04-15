const express = require('express');
const router = express.Router();
const reportController = require('../controllers/report.controller');

// GET /api/report/comparison
router.get('/comparison', reportController.getComparison);

// GET /api/report/manifest
router.get('/manifest', reportController.getManifest);

module.exports = router;
