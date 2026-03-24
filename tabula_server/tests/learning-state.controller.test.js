const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const learningStateController = require('../controllers/learning-state.controller');
const logDiscovery = require('../services/log-discovery.service');

function createMockResponse() {
  return {
    statusCode: 200,
    body: undefined,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.body = payload;
      return this;
    },
  };
}

test('getAllLearningStates returns ordered DTOs for episodic learning-state files', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'learning-state-controller-'));
  const policyDir = path.join(tempDir, 'logs', 'run_20260324_000000', 'policy_x', 'learning_state');
  fs.mkdirSync(policyDir, { recursive: true });

  const episodeTwoPath = path.join(policyDir, 'learning_state_ep02.json');
  const episodeOnePath = path.join(policyDir, 'learning_state_ep01.json');

  fs.writeFileSync(episodeTwoPath, JSON.stringify({
    version: '1.0',
    scenario_id: 'scenario_alpha',
    episode_num: 2,
    policy_type: 'policy_x',
    num_agents: 1,
    num_targets: 2,
    latent_dim: 2,
    episode_state: {
      target_classes: ['A', 'B'],
      agents: [{ agent_idx: 0, agent_lv: [2, 2] }],
    },
  }));
  fs.writeFileSync(episodeOnePath, JSON.stringify({
    version: '1.0',
    scenario_id: 'scenario_alpha',
    episode_num: 1,
    policy_type: 'policy_x',
    num_agents: 1,
    num_targets: 2,
    latent_dim: 2,
    episode_state: {
      target_classes: ['A', 'B'],
      agents: [{ agent_idx: 0, agent_lv: [1, 1] }],
    },
  }));

  const originalGetFilesByContext = logDiscovery.getFilesByContext;
  const originalRepoRoot = logDiscovery.REPO_ROOT;
  logDiscovery.getFilesByContext = () => [episodeTwoPath, episodeOnePath];
  logDiscovery.REPO_ROOT = tempDir;

  try {
    const req = { params: { policyId: 'policy_x' } };
    const res = createMockResponse();

    learningStateController.getAllLearningStates(req, res);

    assert.equal(res.statusCode, 200);
    assert.deepEqual(res.body, [
      {
        episode: {
          fileName: 'learning_state_ep01.json',
          episodeNum: 1,
          policyType: 'policy_x',
          sourcePath: path.join('logs', 'run_20260324_000000', 'policy_x', 'learning_state', 'learning_state_ep01.json'),
        },
        learningState: {
          version: '1.0',
          scenarioId: 'scenario_alpha',
          numAgents: 1,
          numTargets: 2,
          latentDim: 2,
          episodeState: {
            target_classes: ['A', 'B'],
            agents: [{ agent_idx: 0, agent_lv: [1, 1] }],
          },
        },
      },
      {
        episode: {
          fileName: 'learning_state_ep02.json',
          episodeNum: 2,
          policyType: 'policy_x',
          sourcePath: path.join('logs', 'run_20260324_000000', 'policy_x', 'learning_state', 'learning_state_ep02.json'),
        },
        learningState: {
          version: '1.0',
          scenarioId: 'scenario_alpha',
          numAgents: 1,
          numTargets: 2,
          latentDim: 2,
          episodeState: {
            target_classes: ['A', 'B'],
            agents: [{ agent_idx: 0, agent_lv: [2, 2] }],
          },
        },
      },
    ]);
  } finally {
    logDiscovery.getFilesByContext = originalGetFilesByContext;
    logDiscovery.REPO_ROOT = originalRepoRoot;
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});

test('getAllLearningStates returns 404 when no episodic learning-state artifacts exist', async () => {
  const originalGetFilesByContext = logDiscovery.getFilesByContext;
  logDiscovery.getFilesByContext = () => [];

  try {
    const req = { params: { policyId: 'policy_missing' } };
    const res = createMockResponse();

    learningStateController.getAllLearningStates(req, res);

    assert.equal(res.statusCode, 404);
    assert.deepEqual(res.body, {
      error: 'NO_LEARNING_STATE_FOUND',
      message: 'No learning state artifacts found for policy: policy_missing',
    });
  } finally {
    logDiscovery.getFilesByContext = originalGetFilesByContext;
  }
});
