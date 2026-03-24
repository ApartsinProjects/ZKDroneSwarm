import { TestBed } from '@angular/core/testing';
import { computed, signal } from '@angular/core';
import { of } from 'rxjs';
import { EmbeddingBrowserService } from './embedding-browser.service';
import { CrossEpisodeBrowserService } from './cross-episode-browser.service';
import { LearningStateEpisodeDto, PoliciesService } from './policies.service';

class StubCrossEpisodeBrowserService {
  private readonly _episodes = signal<any[]>([
    { episode: { episodeNum: 1 } },
    { episode: { episodeNum: 2 } },
  ]);
  private readonly _currentIndex = signal(0);

  readonly episodes = this._episodes.asReadonly();
  readonly currentIndex = this._currentIndex.asReadonly();
  readonly currentEpisodeSnapshot = computed(() => ({
    episodeNum: this._episodes()[this._currentIndex()]?.episode?.episodeNum ?? null,
    totalSteps: 0,
    hpHistory: [],
    activeTargetsHistory: [],
  }));

  setIndex(index: number): void {
    this._currentIndex.set(index);
  }
}

class StubPoliciesService {
  getAllLearningStates() {
    return of<LearningStateEpisodeDto[]>([
      {
        episode: {
          fileName: 'learning_state_ep01.json',
          episodeNum: 1,
          policyType: 'policy_x',
          sourcePath: 'logs/run_x/policy_x/learning_state/learning_state_ep01.json',
        },
        learningState: {
          version: '1.0',
          scenarioId: 'scenario_x',
          numAgents: 1,
          numTargets: 2,
          latentDim: 2,
          episodeState: {
            agents: [{ agent_lv: [0.1, 0.2], target_lv: [[0.2, 0.3], [0.4, 0.5]] }],
          },
        },
      },
      {
        episode: {
          fileName: 'learning_state_ep02.json',
          episodeNum: 2,
          policyType: 'policy_x',
          sourcePath: 'logs/run_x/policy_x/learning_state/learning_state_ep02.json',
        },
        learningState: {
          version: '1.0',
          scenarioId: 'scenario_x',
          numAgents: 1,
          numTargets: 2,
          latentDim: 2,
          episodeState: {
            agents: [{ agent_lv: [0.6, 0.7], target_lv: [[0.8, 0.9], [1.0, 1.1]] }],
          },
        },
      },
    ]);
  }
}

describe('EmbeddingBrowserService', () => {
  let service: EmbeddingBrowserService;
  let crossEpisodeBrowser: StubCrossEpisodeBrowserService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        EmbeddingBrowserService,
        { provide: CrossEpisodeBrowserService, useClass: StubCrossEpisodeBrowserService },
        { provide: PoliciesService, useClass: StubPoliciesService },
      ],
    });

    service = TestBed.inject(EmbeddingBrowserService);
    crossEpisodeBrowser = TestBed.inject(CrossEpisodeBrowserService) as unknown as StubCrossEpisodeBrowserService;
  });

  it('synchronizes the current embedding snapshot to the selected episode number', () => {
    service.loadSnapshots('policy_x');

    expect(service.currentSnapshot()?.episode.episodeNum).toBe(1);

    crossEpisodeBrowser.setIndex(1);

    expect(service.currentSnapshot()?.episode.episodeNum).toBe(2);
    expect(service.currentAgent()).toEqual({
      agent_lv: [0.6, 0.7],
      target_lv: [[0.8, 0.9], [1.0, 1.1]],
    });
  });
});
