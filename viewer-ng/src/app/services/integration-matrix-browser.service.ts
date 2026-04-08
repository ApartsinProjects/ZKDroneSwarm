import { Injectable, computed, inject, signal } from '@angular/core';
import { CrossEpisodeBrowserService } from './cross-episode-browser.service';
import { LearningStateEpisodeDto, PoliciesService } from './policies.service';

@Injectable({
  providedIn: 'root'
})
export class IntegrationMatrixBrowserService {
  private policiesService = inject(PoliciesService);
  private crossEpisodeBrowser = inject(CrossEpisodeBrowserService);

  private readonly _snapshots = signal<LearningStateEpisodeDto[]>([]);
  private readonly _isLoading = signal(false);
  private readonly _error = signal<string | null>(null);
  private readonly _selectedAgent = signal(0);

  readonly snapshots = this._snapshots.asReadonly();
  readonly isLoading = this._isLoading.asReadonly();
  readonly error = this._error.asReadonly();
  readonly selectedAgent = this._selectedAgent.asReadonly();

  readonly currentEpisodeNum = computed<number | null>(() => {
    const episodes = this.crossEpisodeBrowser.episodes();
    const currentIndex = this.crossEpisodeBrowser.currentIndex();
    if (currentIndex < 0 || currentIndex >= episodes.length) {
      return null;
    }

    return episodes[currentIndex]?.episode?.episodeNum ?? null;
  });

  readonly currentSnapshot = computed<LearningStateEpisodeDto | null>(() => {
    const episodeNum = this.currentEpisodeNum();
    if (episodeNum === null) {
      return null;
    }

    return this._snapshots().find((snapshot) => snapshot.episode.episodeNum === episodeNum) ?? null;
  });

  readonly currentAgents = computed<Array<Record<string, unknown>>>(() => {
    return this.currentSnapshot()?.learningState.episodeState?.agents ?? [];
  });

  readonly currentAgent = computed<Record<string, unknown> | null>(() => {
    const agents = this.currentAgents();
    if (agents.length === 0) {
      return null;
    }

    const selectedAgent = Math.min(this._selectedAgent(), agents.length - 1);
    return agents[selectedAgent] ?? null;
  });

  loadSnapshots(policyId: string): void {
    this._isLoading.set(true);
    this._error.set(null);
    this._selectedAgent.set(0);

    this.policiesService.getAllLearningStates(policyId).subscribe({
      next: (snapshots) => {
        this._snapshots.set(snapshots);
        this._isLoading.set(false);
      },
      error: () => {
        this._snapshots.set([]);
        this._error.set('Unable to load integration matrix snapshots for this policy.');
        this._isLoading.set(false);
      }
    });
  }

  clear(): void {
    this._snapshots.set([]);
    this._error.set(null);
    this._isLoading.set(false);
    this._selectedAgent.set(0);
  }

  setSelectedAgent(index: number): void {
    if (!Number.isInteger(index) || index < 0) {
      return;
    }

    this._selectedAgent.set(index);
  }
}
