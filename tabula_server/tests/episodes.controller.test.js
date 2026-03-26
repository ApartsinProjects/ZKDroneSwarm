const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { buildEpisodeDto } = require('../controllers/episodes.controller');
const logDiscovery = require('../services/log-discovery.service');

test('buildEpisodeDto exposes top-level episode metrics for viewer clients', async () => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'episodes-controller-'));
  const episodePath = path.join(
    tempDir,
    'logs',
    'run_20260326_000000',
    'policy_x',
    'episodes',
    'episode_ep01.json',
  );
  fs.mkdirSync(path.dirname(episodePath), { recursive: true });

  const originalRepoRoot = logDiscovery.REPO_ROOT;
  logDiscovery.REPO_ROOT = tempDir;

  try {
    const dto = buildEpisodeDto(episodePath, {
      version: '1.3',
      episode_num: 1,
      config: { policy_type: 'policy_x' },
      scenario: { num_targets: 2 },
      steps: [],
      summary: {
        total_steps: 7,
        metrics: { total_ammo_used: 12 },
      },
      metrics: {
        total_ammo_used: 12,
        total_net_damage: 9,
        total_gross_damage: 15,
        total_collisions: 3,
        dmg_eff: 0.8,
        shots_per_target: 2.4,
      },
    });

    assert.deepEqual(dto.metrics, {
      total_ammo_used: 12,
      total_net_damage: 9,
      total_gross_damage: 15,
      total_collisions: 3,
      dmg_eff: 0.8,
      shots_per_target: 2.4,
    });
    assert.equal(dto.summary.total_steps, 7);
    assert.equal(dto.episode.sourcePath, path.join(
      'logs',
      'run_20260326_000000',
      'policy_x',
      'episodes',
      'episode_ep01.json',
    ));
  } finally {
    logDiscovery.REPO_ROOT = originalRepoRoot;
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
