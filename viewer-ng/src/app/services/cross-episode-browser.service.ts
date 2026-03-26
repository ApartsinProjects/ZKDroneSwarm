import { Injectable, signal, computed, inject } from '@angular/core';
import { EpisodeDto, PoliciesService } from './policies.service';

type MetricDirection = 'higher' | 'lower';

export interface EpisodeMetricRow {
  label: string;
  value: string;
  numericValue: number | null;
  progress: EpisodeMetricProgress | null;
}

export interface EpisodeMetricProgress {
  deltaValue: number | null;
  deltaLabel: string | null;
  deltaTone: 'better' | 'worse' | 'neutral';
  normalizedScore: number;
  percentileLabel: string;
}

interface EpisodeMetricDefinition {
  label: string;
  betterDirection: MetricDirection | null;
  readValue: (metrics: Record<string, unknown>) => unknown;
}

const EPISODE_METRIC_DEFINITIONS: ReadonlyArray<EpisodeMetricDefinition> = [
  {
    label: 'Total Ammo Used',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['total_ammo_used'],
  },
  {
    label: 'Total Overkill',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['total_overkill'],
  },
  {
    label: 'Total Gross Damage',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['total_gross_damage'],
  },
  {
    label: 'Total Collisions',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['total_collisions'],
  },
  {
    label: 'Ammo Eff',
    betterDirection: 'higher',
    readValue: (metrics) => metrics['ammo_eff'],
  },
  {
    label: 'DMG Eff',
    betterDirection: 'higher',
    readValue: (metrics) => metrics['dmg_eff'],
  },
  {
    label: 'Shots / Target',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['shots_per_target'],
  },
  {
    label: 'Throughput',
    betterDirection: null,
    readValue: (metrics) => metrics['throughput'],
  },
  {
    label: 'Coordination',
    betterDirection: null,
    readValue: (metrics) => {
      const coordinationStr = metrics['coordination_str'];
      if (typeof coordinationStr === 'string' && coordinationStr.length > 0) {
        return coordinationStr;
      }

      return metrics['coordination_score'];
    },
  },
] as const;

export interface EpisodeSnapshot {
  episodeNum: number | null;
  totalSteps: number;
  hpHistory: number[];
  activeTargetsHistory: number[];
  metricRows: EpisodeMetricRow[];
  totalNetDamageValue: string | null;
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
    const metrics = this.readMetrics(episode);
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
      metricRows: this.buildMetricRows(episodes, idx),
      totalNetDamageValue: this.formatMetricValue('Total Net Damage', metrics['total_net_damage']),
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

  private buildMetricRows(episodes: EpisodeDto[], currentIndex: number): EpisodeMetricRow[] {
    const metrics = this.readMetrics(episodes[currentIndex]);

    return EPISODE_METRIC_DEFINITIONS.flatMap((definition) => {
      const rawValue = definition.readValue(metrics);
      const value = this.formatMetricValue(definition.label, rawValue);
      const numericValue = this.readMetricNumber(rawValue);

      return value === null ? [] : [{
        label: definition.label,
        value,
        numericValue,
        progress: this.buildMetricProgress(definition, episodes, currentIndex, numericValue),
      }];
    });
  }

  private buildMetricProgress(
    definition: EpisodeMetricDefinition,
    episodes: EpisodeDto[],
    currentIndex: number,
    currentValue: number | null,
  ): EpisodeMetricProgress | null {
    if (definition.betterDirection === null || currentValue === null) {
      return null;
    }

    const historicalValues = episodes
      .map((episode) => this.readMetricNumber(definition.readValue(this.readMetrics(episode))))
      .filter((value): value is number => value !== null);

    if (historicalValues.length === 0) {
      return null;
    }

    const minValue = Math.min(...historicalValues);
    const maxValue = Math.max(...historicalValues);
    const normalizedScore = this.normalizeMetricValue(
      currentValue,
      minValue,
      maxValue,
      definition.betterDirection,
    );

    const previousValue = currentIndex > 0
      ? this.readMetricNumber(definition.readValue(this.readMetrics(episodes[currentIndex - 1])))
      : null;
    const deltaValue = previousValue === null ? null : currentValue - previousValue;

    return {
      deltaValue,
      deltaLabel: deltaValue === null ? null : this.formatDeltaValue(definition.label, deltaValue),
      deltaTone: this.resolveDeltaTone(definition.betterDirection, deltaValue),
      normalizedScore,
      percentileLabel: `${Math.round(normalizedScore * 100)}%`,
    };
  }

  private normalizeMetricValue(
    value: number,
    minValue: number,
    maxValue: number,
    betterDirection: MetricDirection,
  ): number {
    if (minValue === maxValue) {
      return 0.5;
    }

    const span = maxValue - minValue;
    const normalized = betterDirection === 'higher'
      ? (value - minValue) / span
      : (maxValue - value) / span;

    return Math.min(1, Math.max(0, normalized));
  }

  private formatDeltaValue(label: string, deltaValue: number): string {
    if (deltaValue === 0) {
      return '0';
    }

    const formattedValue = this.formatMetricValue(label, Math.abs(deltaValue)) ?? '0';
    const sign = deltaValue > 0 ? '+' : '-';

    return `${sign}${formattedValue}`;
  }

  private resolveDeltaTone(
    betterDirection: MetricDirection,
    deltaValue: number | null,
  ): 'better' | 'worse' | 'neutral' {
    if (deltaValue === null || deltaValue === 0) {
      return 'neutral';
    }

    if (betterDirection === 'higher') {
      return deltaValue > 0 ? 'better' : 'worse';
    }

    return deltaValue < 0 ? 'better' : 'worse';
  }

  private readMetrics(episode: EpisodeDto | undefined): Record<string, unknown> {
    return this.readRecord(episode?.metrics);
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

  private readMetricNumber(value: unknown): number | null {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }

    return null;
  }
}
