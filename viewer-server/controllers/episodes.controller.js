const fs = require('fs');
const path = require('path');
const logDiscovery = require('../services/log-discovery.service');

function sumAttributes(attributes) {
  return Object.values(attributes || {}).reduce(
    (sum, value) => sum + (Number.isFinite(value) ? value : 0),
    0
  );
}

function buildMapSceneDto(episodePath, episodeData) {
  const scenario = episodeData.scenario || {};
  const config = episodeData.config || {};
  const classAttributeMapping = config.class_attribute_mapping || {};
  const weaponAssignments = scenario.weapon_assignments || {};
  const dronePositions = scenario.drone_positions || [];
  const targetPositions = scenario.target_positions || [];
  const targetClasses = scenario.target_classes || [];
  const worldSize = config.world_size || [1000, 1000];

  const drones = dronePositions.map((position, index) => ({
    id: `drone_${index}`,
    position: position,
    weaponType: weaponAssignments[`drone_${index}`] || 'unknown'
  }));

  const targets = targetPositions.map((position, index) => {
    const classType = targetClasses[index] || 'unknown';

    return {
      id: `target_${index}`,
      position: position,
      classType,
      hp: sumAttributes(classAttributeMapping[classType] || {})
    };
  });

  return {
    episode: {
      fileName: path.basename(episodePath),
      episodeNum: episodeData.episode_num ?? null,
      version: episodeData.version || 'unknown',
      policyType: config.policy_type || 'unknown',
      sourcePath: path.relative(logDiscovery.REPO_ROOT, episodePath)
    },
    map: {
      title: `World State Map ( Policy: ${config.policy_type || 'unknown'} )`,
      worldSize: {
        width: Number(worldSize[0]) || 1000,
        height: Number(worldSize[1]) || 1000
      },
      drones,
      targets
    }
  };
}

const episodesController = {
  getMapScene: (req, res) => {
    try {
      const { policyId } = req.params;
      
      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      // Prioritize best episodes, fallback to any episode
      let candidates = logDiscovery.getFilesByContext(policyId, 'episodes', 'episode_best_');
      if (candidates.length === 0) {
        candidates = logDiscovery.getFilesByContext(policyId, 'episodes', 'episode_');
      }

      const episodePath = logDiscovery.getLatestFile(candidates);
      if (!episodePath) {
        return res.status(404).json({ error: 'NO_EPISODES_FOUND', message: `No episode artifacts found for policy: ${policyId}` });
      }

      const episodeData = JSON.parse(fs.readFileSync(episodePath, 'utf8'));
      res.json(buildMapSceneDto(episodePath, episodeData));
    } catch (error) {
      res.status(500).json({
        error: 'MAP_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown map loading error.'
      });
    }
  }
};

module.exports = episodesController;
