const express = require('express');
const router = express.Router();
const environmentController = require('../controllers/environment.controller');

// GET /api/environment
router.get('/', environmentController.getEnvironment);

module.exports = router;
