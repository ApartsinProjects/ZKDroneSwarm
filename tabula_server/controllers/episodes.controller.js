const fs = require('fs');
const path = require('path');
const logDiscovery = require('../services/log-discovery.service');

function sumAttributes(attributes) {
  return Object.values(attributes || {}).reduce(
    (sum, value) => sum + (Number.isFinite(value) ? value : 0),
    0
  );
}

function loadEnvironmentData(episodePath, episodeData) {
  const scenario = episodeData.scenario || {};
  const config = episodeData.config || {};

  if (scenario && Object.keys(scenario).length > 0 && config.world_size) {
    return { scenario, config };
  }

  const environmentRef = episodeData.environment_path;
  if (!environmentRef) {
    return { scenario, config };
  }

  const environmentPath = path.resolve(path.dirname(episodePath), environmentRef);
  const environmentData = JSON.parse(fs.readFileSync(environmentPath, 'utf8'));
  return {
    scenario: environmentData.scenario || {},
    config: {
      ...(environmentData.config || {}),
      ...config
    }
  };
}

function buildEpisodeDto(episodePath, episodeData) {
  const { scenario, config } = loadEnvironmentData(episodePath, episodeData);

  return {
    episode: {
      fileName: path.basename(episodePath),
      episodeNum: episodeData.episode_num ?? null,
      version: episodeData.version || 'unknown',
      policyType: config.policy_type || 'unknown',
      sourcePath: path.relative(logDiscovery.REPO_ROOT, episodePath)
    },
    scenario: scenario,
    steps: episodeData.steps || [],
    summary: episodeData.summary || {},
    metrics: episodeData.metrics || {}
  };
}

function loadPolicySummary(policyId) {
  const latestRun = logDiscovery.getLatestRunFolder();
  if (!latestRun) {
    return null;
  }

  const summaryPath = path.join(logDiscovery.LOGS_DIR, latestRun, policyId, 'episodes_summary.json');
  if (!fs.existsSync(summaryPath)) {
    return null;
  }

  return JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
}

function resolveBestEpisodePath(policyId, candidates) {
  const summary = loadPolicySummary(policyId);
  if (summary && summary.best_episode_path) {
    const latestRun = logDiscovery.getLatestRunFolder();
    if (latestRun) {
      const bestEpisodePath = path.join(
        logDiscovery.LOGS_DIR,
        latestRun,
        policyId,
        summary.best_episode_path
      );
      if (fs.existsSync(bestEpisodePath)) {
        return bestEpisodePath;
      }
    }
  }

  if (candidates.length === 1) {
    return candidates[0];
  }

  const bestCandidates = candidates.filter(p => path.basename(p).includes('best'));
  return logDiscovery.getLatestFile(bestCandidates.length > 0 ? bestCandidates : candidates);
}

const episodesController = {
  getMapScene: (req, res) => {
    try {
      const { policyId } = req.params;
      
      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      const candidates = logDiscovery.getFilesByContext(policyId, 'episodes', 'episode_');
      const episodePath = resolveBestEpisodePath(policyId, candidates);
      if (!episodePath) {
        return res.status(404).json({ error: 'NO_EPISODES_FOUND', message: `No episode artifacts found for policy: ${policyId}` });
      }

      const episodeData = JSON.parse(fs.readFileSync(episodePath, 'utf8'));
      res.json(buildEpisodeDto(episodePath, episodeData));
    } catch (error) {
      res.status(500).json({
        error: 'MAP_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown map loading error.'
      });
    }
  },

  getAllEpisodes: (req, res) => {
    try {
      const { policyId } = req.params;
      
      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      const candidates = logDiscovery.getFilesByContext(policyId, 'episodes', 'episode_');
      if (candidates.length === 0) {
        return res.status(404).json({ error: 'NO_EPISODES_FOUND', message: `No episode artifacts found for policy: ${policyId}` });
      }

      const episodes = candidates.map(episodePath => {
        const episodeData = JSON.parse(fs.readFileSync(episodePath, 'utf8'));
        return buildEpisodeDto(episodePath, episodeData);
      });

      res.json(episodes);
    } catch (error) {
      res.status(500).json({
        error: 'EPISODES_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown episodes loading error.'
      });
    }
  },

  getBestEpisode: (req, res) => {
    try {
      const { policyId } = req.params;
      
      if (!policyId) {
        return res.status(400).json({ error: 'MISSING_POLICY_ID', message: 'Policy ID is required.' });
      }

      const candidates = logDiscovery.getFilesByContext(policyId, 'episodes', 'episode_');
      if (candidates.length === 0) {
        return res.status(404).json({ error: 'NO_EPISODES_FOUND', message: `No episode artifacts found for policy: ${policyId}` });
      }

      const bestEpisodePath = resolveBestEpisodePath(policyId, candidates);
      if (!bestEpisodePath) {
        return res.status(404).json({ error: 'NO_BEST_EPISODE_FOUND', message: `No suitable best episode artifact found for policy: ${policyId}` });
      }

      const episodeData = JSON.parse(fs.readFileSync(bestEpisodePath, 'utf8'));
      res.json(buildEpisodeDto(bestEpisodePath, episodeData));
    } catch (error) {
      res.status(500).json({
        error: 'BEST_EPISODE_LOAD_FAILED',
        message: error instanceof Error ? error.message : 'Unknown episodes loading error.'
      });
    }
  }
};

module.exports = {
  ...episodesController,
  buildEpisodeDto,
  loadEnvironmentData,
  loadPolicySummary,
  resolveBestEpisodePath
};
