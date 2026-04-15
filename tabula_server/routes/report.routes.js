const express = require('express');
const router = express.Router();
const reportController = require('../controllers/report.controller');

// GET /api/report/comparison
router.get('/comparison', reportController.getComparison);

module.exports = router;
