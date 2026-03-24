import { Injectable, signal, computed, inject } from '@angular/core';
import { PoliciesService } from './policies.service';

export interface EpisodeSnapshot {
  episodeNum: number | null;
  totalSteps: number;
  hpHistory: number[];
  activeTargetsHistory: number[];
}

@Injectable({
  providedIn: 'root'
})
export class CrossEpisodeBrowserService {
  private policiesService = inject(PoliciesService);

  private readonly _episodes = signal<any[]>([]);
  private readonly _currentIndex = signal(-1);
  private readonly _isLoading = signal(false);

  readonly episodes = this._episodes.asReadonly();
  readonly currentIndex = this._currentIndex.asReadonly();
  readonly isLoading = this._isLoading.asReadonly();

  readonly totalEpisodes = computed(() => this._episodes().length);

  readonly currentEpisodeSnapshot = computed<EpisodeSnapshot | null>(() => {
    const episodes = this._episodes();
    const idx = this._currentIndex();
    if (idx < 0 || idx >= episodes.length) return null;

    const episode = episodes[idx];
    const steps = episode.steps || [];
    const hpHistory: number[] = [];
    const activeTargetsHistory: number[] = [];

    for (const step of steps) {
      let totalHp = 0;
      let activeCount = 0;
      const targetHps: number[] = step.info?.target_hps ?? [];
      for (const hp of targetHps) {
        totalHp += (hp > 0 ? hp : 0);
        if (hp > 0) activeCount += 1;
      }
      hpHistory.push(totalHp);
      activeTargetsHistory.push(activeCount);
    }

    return {
      episodeNum: episode.episode?.episodeNum ?? null,
      totalSteps: steps.length,
      hpHistory,
      activeTargetsHistory
    };
  });

  loadEpisodes(policyId: string): void {
    this._isLoading.set(true);
    this._currentIndex.set(-1);
    this.policiesService.getAllEpisodes(policyId).subscribe({
      next: (episodes) => {
        // Sort episodes by episodeNum if available
        const sorted = [...episodes].sort((a, b) => {
            const numA = a.episode?.episodeNum ?? 0;
            const numB = b.episode?.episodeNum ?? 0;
            return numA - numB;
        });
        this._episodes.set(sorted);
        if (sorted.length > 0) {
          this._currentIndex.set(0);
        }
        this._isLoading.set(false);
      },
      error: () => {
        this._episodes.set([]);
        this._isLoading.set(false);
      }
    });
  }

  setIndex(index: number): void {
    if (index >= 0 && index < this._episodes().length) {
      this._currentIndex.set(index);
    }
  }
}
