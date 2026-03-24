const fs = require('fs');
const path = require('path');
const logDiscovery = require('../services/log-discovery.service');

function buildLearningStateDto(learningStatePath, learningStateData) {
  return {
    episode: {
      fileName: path.basename(learningStatePath),
      episodeNum: learningStateData.episode_num ?? null,
      policyType: learningStateData.policy_type ?? 'unknown',
      sourcePath: path.relative(logDiscovery.REPO_ROOT, learningStatePath),
    },
    learningState: {
      version: learningStateData.version || 'unknown',
      scenarioId: learningStateData.scenario_id ?? null,
      numAgents: learningStateData.num_agents ?? null,
      numTargets: learningStateData.num_targets ?? null,
      latentDim: learningStateData.latent_dim ?? null,
      episodeState: learningStateData.episode_state ?? null,
    },
  };
}

function compareLearningStateDtos(left, right) {
  const leftEpisodeNum = Number.isFinite(left.episode.episodeNum) ? left.episode.episodeNum : Infinity;
  const rightEpisodeNum = Number.isFinite(right.episode.episodeNum) ? right.episode.episodeNum : Infinity;

  if (leftEpisodeNum !== rightEpisodeNum) {
    return leftEpisodeNum - rightEpisodeNum;
  }

  return left.episode.fileName.localeCompare(right.episode.fileName);
}

const learningStateController = {
  getAllLearningStates: (req, res) => {
    try {
      const { policyId } = req.params;

      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      const candidates = logDiscovery.getFilesByContext(policyId, 'learning_state', 'learning_state_ep');
      if (candidates.length === 0) {
        return res.status(404).json({
          error: 'NO_LEARNING_STATE_FOUND',
          message: `No learning state artifacts found for policy: ${policyId}`
        });
      }

      const learningStates = candidates
        .map((learningStatePath) => {
          const learningStateData = JSON.parse(fs.readFileSync(learningStatePath, 'utf8'));
          return buildLearningStateDto(learningStatePath, learningStateData);
        })
        .sort(compareLearningStateDtos);

      return res.json(learningStates);
    } catch (error) {
      return res.status(500).json({
        error: 'LEARNING_STATE_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown learning state loading error.'
      });
    }
  },
  getLatestLearningState: (req, res) => {
    try {
      const { policyId } = req.params;
      
      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      const candidates = logDiscovery.getFilesByContext(policyId, 'learning_state', 'learning_state_');
      const learningStatePath = logDiscovery.getLatestFile(candidates);

      if (!learningStatePath) {
        return res.status(404).json({ error: 'NO_LEARNING_STATE_FOUND', message: `No learning state artifacts found for policy: ${policyId}` });
      }

      const learningStateData = JSON.parse(fs.readFileSync(learningStatePath, 'utf8'));
      res.json(learningStateData);
    } catch (error) {
      res.status(500).json({
        error: 'LEARNING_STATE_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown learning state loading error.'
      });
    }
  }
};

learningStateController.__private__ = {
  buildLearningStateDto,
  compareLearningStateDtos,
};

module.exports = learningStateController;
