import { Injectable, signal, computed, inject } from '@angular/core';
import { EpisodeDto, PoliciesService } from './policies.service';

export interface EpisodeMetricRow {
  label: string;
  value: string;
}

export interface EpisodeSnapshot {
  episodeNum: number | null;
  totalSteps: number;
  hpHistory: number[];
  activeTargetsHistory: number[];
  metricRows: EpisodeMetricRow[];
}

@Injectable({
  providedIn: 'root'
})
export class CrossEpisodeBrowserService {
  private policiesService = inject(PoliciesService);

  private readonly _episodes = signal<EpisodeDto[]>([]);
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
      const stepInfo = this.readRecord(step['info']);
      const targetHps = this.readNumberArray(stepInfo['target_hps']);
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
      activeTargetsHistory,
      metricRows: this.buildMetricRows(episode.metrics),
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

  private buildMetricRows(metrics: Record<string, unknown>): EpisodeMetricRow[] {
    const coordinationValue = this.readCoordinationValue(metrics);

    const candidates: Array<[string, unknown]> = [
      ['Targets Neutralized', metrics['targets_neutralized']],
      ['Total Ammo Used', metrics['total_ammo_used']],
      ['Total Overkill', metrics['total_overkill']],
      ['Total Effective Damage', metrics['total_effective_damage']],
      ['Total Potential Damage', metrics['total_potential_damage']],
      ['Total Collisions', metrics['total_collisions']],
      ['Ammo Eff', metrics['ammo_eff']],
      ['DMG Eff', metrics['dmg_eff']],
      ['Shots / Target', metrics['shots_per_target']],
      ['Throughput', metrics['throughput']],
      ['Coordination', coordinationValue],
    ];

    return candidates.flatMap(([label, rawValue]) => {
      const value = this.formatMetricValue(label, rawValue);
      return value === null ? [] : [{ label, value }];
    });
  }

  private readCoordinationValue(metrics: Record<string, unknown>): unknown {
    const coordinationStr = metrics['coordination_str'];
    if (typeof coordinationStr === 'string' && coordinationStr.length > 0) {
      return coordinationStr;
    }

    return metrics['coordination_score'];
  }

  private formatMetricValue(label: string, value: unknown): string | null {
    if (value === null || value === undefined) {
      return null;
    }

    if (typeof value === 'string') {
      return value;
    }

    if (typeof value === 'number' && Number.isFinite(value)) {
      if (Number.isInteger(value)) {
        return value.toLocaleString();
      }

      const formatted = value.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });

      return label === 'Throughput' ? `${formatted}%` : formatted;
    }

    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }

    return String(value);
  }

  private readRecord(value: unknown): Record<string, unknown> {
    return value !== null && typeof value === 'object' ? (value as Record<string, unknown>) : {};
  }

  private readNumberArray(value: unknown): number[] {
    if (!Array.isArray(value)) {
      return [];
    }

    return value.filter((item): item is number => typeof item === 'number' && Number.isFinite(item));
  }
}
