import { Injectable, signal, computed, inject } from '@angular/core';
import { EpisodeDto, PoliciesService } from './policies.service';
import { catchError } from 'rxjs/operators';
import { forkJoin, of } from 'rxjs';

type MetricDirection = 'higher' | 'lower';
const RANDOM_POLICY_ID = 'random';

export interface EpisodeMetricRow {
  label: string;
  value: string;
  numericValue: number | null;
  progress: EpisodeMetricProgress | null;
  baseline: EpisodeMetricBaseline;
}

export interface EpisodeMetricProgress {
  deltaValue: number | null;
  deltaLabel: string | null;
  deltaTone: 'better' | 'worse' | 'neutral';
  normalizedScore: number;
  percentileLabel: string;
  isFixed: boolean;
}

export interface EpisodeMetricBaseline {
  label: string;
  tone: 'better' | 'worse' | 'neutral' | 'base' | 'unavailable';
}

interface EpisodeMetricDefinition {
  label: string;
  betterDirection: MetricDirection | null;
  readValue: (metrics: Record<string, unknown>) => unknown;
}

const EPISODE_METRIC_DEFINITIONS: ReadonlyArray<EpisodeMetricDefinition> = [
  {
    label: 'Match Quality',
    betterDirection: 'higher',
    readValue: (metrics) => metrics['avg_latent_match_quality'],
  },
  {
    label: 'Shots / Target',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['shots_per_target'],
  },
  {
    label: 'Total Overkill',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['total_overkill'],
  },
  {
    label: 'Total Collisions',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['total_collisions'],
  },
  {
    label: 'Steps',
    betterDirection: 'lower',
    readValue: (metrics) => metrics['steps'],
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
  private readonly _randomEpisodes = signal<EpisodeDto[]>([]);
  private readonly _selectedPolicyId = signal<string | null>(null);
  private readonly _currentIndex = signal(-1);
  private readonly _isLoading = signal(false);
  private readonly _policyIndexMap = new Map<string, number>();

  readonly episodes = this._episodes.asReadonly();
  readonly randomEpisodes = this._randomEpisodes.asReadonly();
  readonly selectedPolicyId = this._selectedPolicyId.asReadonly();
  readonly currentIndex = this._currentIndex.asReadonly();
  readonly isLoading = this._isLoading.asReadonly();

  readonly totalEpisodes = computed(() => this._episodes().length);

  readonly currentEpisodeSnapshot = computed<EpisodeSnapshot | null>(() => {
    const episodes = this._episodes();
    const randomEpisodes = this._randomEpisodes();
    const selectedPolicyId = this._selectedPolicyId();
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
      metricRows: this.buildMetricRows(episodes, randomEpisodes, selectedPolicyId, idx),
      totalNetDamageValue: this.formatMetricValue('Total Net Damage', metrics['total_net_damage']),
    };
  });

  loadEpisodes(policyId: string): void {
    this._isLoading.set(true);
    this._selectedPolicyId.set(policyId);

    const selectedEpisodes$ = this.policiesService.getAllEpisodes(policyId);
    const randomEpisodes$ = policyId === RANDOM_POLICY_ID
      ? of([] as EpisodeDto[])
      : this.policiesService.getAllEpisodes(RANDOM_POLICY_ID).pipe(
          catchError(() => of([] as EpisodeDto[])),
        );

    forkJoin({
      selectedEpisodes: selectedEpisodes$,
      randomEpisodes: randomEpisodes$,
    }).subscribe({
      next: ({ selectedEpisodes, randomEpisodes }) => {
        const sortedSelectedEpisodes = this.sortEpisodes(selectedEpisodes);
        const sortedRandomEpisodes = policyId === RANDOM_POLICY_ID
          ? sortedSelectedEpisodes
          : this.sortEpisodes(randomEpisodes);

        this._episodes.set(sortedSelectedEpisodes);
        this._randomEpisodes.set(sortedRandomEpisodes);
        if (sortedSelectedEpisodes.length > 0) {
          const savedIndex = this._policyIndexMap.get(policyId) ?? 0;
          const validIndex = Math.min(savedIndex, sortedSelectedEpisodes.length - 1);
          this._currentIndex.set(validIndex);
        } else {
          this._currentIndex.set(-1);
        }
        this._isLoading.set(false);
      },
      error: () => {
        this._episodes.set([]);
        this._randomEpisodes.set([]);
        this._isLoading.set(false);
      },
    });
  }

  setIndex(index: number): void {
    if (index >= 0 && index < this._episodes().length) {
      this._currentIndex.set(index);
      const policyId = this._selectedPolicyId();
      if (policyId !== null) {
        this._policyIndexMap.set(policyId, index);
      }
    }
  }

  private buildMetricRows(
    episodes: EpisodeDto[],
    randomEpisodes: EpisodeDto[],
    selectedPolicyId: string | null,
    currentIndex: number,
  ): EpisodeMetricRow[] {
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
        baseline: this.buildMetricBaseline(
          definition,
          episodes[currentIndex],
          randomEpisodes,
          selectedPolicyId,
          numericValue,
        ),
      }];
    });
  }

  private sortEpisodes(episodes: EpisodeDto[]): EpisodeDto[] {
    return [...episodes].sort((a, b) => {
      const numA = a.episode?.episodeNum ?? 0;
      const numB = b.episode?.episodeNum ?? 0;
      return numA - numB;
    });
  }

  private buildMetricBaseline(
    definition: EpisodeMetricDefinition,
    currentEpisode: EpisodeDto,
    randomEpisodes: EpisodeDto[],
    selectedPolicyId: string | null,
    currentValue: number | null,
  ): EpisodeMetricBaseline {
    if (selectedPolicyId === RANDOM_POLICY_ID) {
      return { label: 'Base', tone: 'base' };
    }

    if (definition.betterDirection === null || currentValue === null) {
      return { label: 'N/A', tone: 'unavailable' };
    }

    const currentEpisodeNum = currentEpisode.episode?.episodeNum ?? null;
    if (currentEpisodeNum === null) {
      return { label: 'N/A', tone: 'unavailable' };
    }

    const randomEpisode = this.resolveRandomBaselineEpisode(randomEpisodes, currentEpisodeNum);
    if (!randomEpisode) {
      return { label: 'N/A', tone: 'unavailable' };
    }

    const randomValue = this.readMetricNumber(definition.readValue(this.readMetrics(randomEpisode)));
    if (randomValue === null || randomValue === 0) {
      return { label: 'N/A', tone: 'unavailable' };
    }

    const signedPercent = this.computeBaselinePercent(
      currentValue,
      randomValue,
      definition.betterDirection,
    );

    return {
      label: this.formatSignedPercent(signedPercent),
      tone: signedPercent === 0 ? 'neutral' : signedPercent > 0 ? 'better' : 'worse',
    };
  }

  private resolveRandomBaselineEpisode(
    randomEpisodes: EpisodeDto[],
    currentEpisodeNum: number,
  ): EpisodeDto | null {
    if (randomEpisodes.length === 1) {
      return randomEpisodes[0];
    }

    return randomEpisodes.find(
      (episode) => episode.episode?.episodeNum === currentEpisodeNum,
    ) ?? null;
  }

  private computeBaselinePercent(
    currentValue: number,
    baselineValue: number,
    betterDirection: MetricDirection,
  ): number {
    const rawPercent = ((currentValue - baselineValue) / baselineValue) * 100;
    return betterDirection === 'higher' ? rawPercent : -rawPercent;
  }

  private formatSignedPercent(value: number): string {
    const roundedValue = Math.round(value);
    if (roundedValue === 0) {
      return '0%';
    }

    const sign = roundedValue > 0 ? '+' : '';
    return `${sign}${roundedValue}%`;
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
    const isFixed = minValue === maxValue;
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
      percentileLabel: isFixed ? 'Fixed' : `${Math.round(normalizedScore * 100)}%`,
      isFixed,
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
