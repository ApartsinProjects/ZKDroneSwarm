import { Injectable, signal, computed, OnDestroy } from '@angular/core';
import { MapSceneViewModel, ViewMapEntity, EngagementLine } from './map.service';

const PLAYBACK_INTERVAL_MS = 500;

@Injectable({
  providedIn: 'root'
})
export class EpisodeStateService implements OnDestroy {
  private readonly _currentEpisode = signal<any | null>(null);
  private readonly _currentStepIndex = signal(-1);
  private readonly _isPlaying = signal(false);
  private readonly _hpHistory = signal<number[]>([]);
  private readonly _activeTargetsHistory = signal<number[]>([]);
  private _timerId: ReturnType<typeof setInterval> | null = null;

  readonly currentEpisode = this._currentEpisode.asReadonly();
  readonly currentStepIndex = this._currentStepIndex.asReadonly();
  readonly isPlaying = this._isPlaying.asReadonly();
  readonly hpHistory = this._hpHistory.asReadonly();
  readonly activeTargetsHistory = this._activeTargetsHistory.asReadonly();

  readonly totalSteps = computed(() => {
    const ep = this._currentEpisode();
    return ep?.steps?.length ?? 0;
  });

  readonly currentStep = computed(() => {
    const ep = this._currentEpisode();
    const idx = this._currentStepIndex();
    if (!ep?.steps || idx < 0 || idx >= ep.steps.length) {
      return null;
    }
    return ep.steps[idx];
  });

  readonly animatedScene = computed<MapSceneViewModel | null>(() => {
    const ep = this._currentEpisode();
    const step = this.currentStep();
    if (!ep || !step) {
      return null;
    }
    return this.buildAnimatedScene(ep, step);
  });

  setEpisode(episode: any): void {
    this.pause();
    this._currentEpisode.set(episode);
    this._currentStepIndex.set(-1);

    const hpList: number[] = [];
    const activeList: number[] = [];
    if (episode?.steps) {
      for (const step of episode.steps) {
        let totalHp = 0;
        let activeCount = 0;
        const targetHps: number[] = step.info?.target_hps ?? [];
        for (const hp of targetHps) {
          totalHp += (hp > 0 ? hp : 0);
          if (hp > 0) activeCount++;
        }
        hpList.push(totalHp);
        activeList.push(activeCount);
      }
    }
    this._hpHistory.set(hpList);
    this._activeTargetsHistory.set(activeList);
  }

  clearEpisode(): void {
    this.pause();
    this._currentEpisode.set(null);
    this._currentStepIndex.set(-1);
    this._hpHistory.set([]);
    this._activeTargetsHistory.set([]);
  }

  play(): void {
    if (this._isPlaying() || this.totalSteps() === 0) {
      return;
    }

    // If at the end, restart from beginning
    if (this._currentStepIndex() >= this.totalSteps() - 1) {
      this._currentStepIndex.set(-1);
    }

    this._isPlaying.set(true);
    this.tick();
    this._timerId = setInterval(() => {
      // Because we called tick() immediately, it's possible _isPlaying became false
      if (this._isPlaying()) {
        this.tick();
      }
    }, PLAYBACK_INTERVAL_MS);
  }

  pause(): void {
    this._isPlaying.set(false);
    if (this._timerId !== null) {
      clearInterval(this._timerId);
      this._timerId = null;
    }
  }

  reset(): void {
    this.pause();
    this._currentStepIndex.set(-1);
  }

  ngOnDestroy(): void {
    this.pause();
  }

  private tick(): void {
    const nextIndex = this._currentStepIndex() + 1;
    if (nextIndex >= this.totalSteps()) {
      this.pause();
      return;
    }
    this._currentStepIndex.set(nextIndex);
  }

  private buildAnimatedScene(episode: any, step: any): MapSceneViewModel {
    const scenario = episode.scenario ?? {};
    const dronePositions: [number, number][] = scenario.drone_positions ?? [];
    const targetPositions: [number, number][] = scenario.target_positions ?? [];
    const weaponAssignments: Record<string, string> = scenario.weapon_assignments ?? {};
    const targetClasses: string[] = scenario.target_classes ?? [];

    // Use default world size — MapComponent's static scene provides actual values
    const width = 1000;
    const height = 1000;

    const targetActive: boolean[] = step.info?.target_active ?? [];
    const targetHps: number[] = step.info?.target_hps ?? [];

    const drones: ViewMapEntity[] = dronePositions.map((pos, i) => ({
      id: `drone_${i}`,
      position: pos,
      weaponType: weaponAssignments[`drone_${i}`] ?? 'unknown',
      leftPct: width > 0 ? (pos[0] / width) * 100 : 0,
      topPct: height > 0 ? 100 - (pos[1] / height) * 100 : 0,
    }));

    const targets: ViewMapEntity[] = targetPositions.map((pos, i) => ({
      id: `target_${i}`,
      position: pos,
      classType: targetClasses[i] ?? 'unknown',
      hp: targetHps[i] ?? 0,
      isActive: targetActive[i] ?? true,
      leftPct: width > 0 ? (pos[0] / width) * 100 : 0,
      topPct: height > 0 ? 100 - (pos[1] / height) * 100 : 0,
    }));

    // Compute engagements from step actions
    const actions: Record<string, number> = step.action ?? {};
    const engagements: EngagementLine[] = [];
    const hitTargetIds = new Set<string>();

    for (const [droneId, targetIdx] of Object.entries(actions)) {
      // Skip NoOp (-1) and invalid indices
      if (targetIdx < 0 || targetIdx >= targets.length) {
        continue;
      }

      const droneIdx = parseInt(droneId.split('_')[1], 10);
      if (droneIdx < 0 || droneIdx >= drones.length) {
        continue;
      }

      const drone = drones[droneIdx];
      const target = targets[targetIdx];

      engagements.push({
        droneId: drone.id,
        targetId: target.id,
        droneLeftPct: drone.leftPct + 3,
        droneTopPct: drone.topPct + 1,
        targetLeftPct: target.leftPct,
        targetTopPct: target.topPct,
      });

      // Flame only on active targets being hit
      if (target.isActive !== false) {
        hitTargetIds.add(target.id);
      }
    }

    return {
      run: {
        scenarioId: this.formatEpisodeLabel(episode.episode?.sourcePath),
        version: episode.episode?.version ?? 'unknown',
      },
      title: `Step ${step.step_num} of ${episode.steps.length}`,
      width,
      height,
      drones,
      targets,
      engagements,
      hitTargetIds,
    };
  }

  private formatEpisodeLabel(sourcePath: unknown): string {
    if (typeof sourcePath !== 'string' || sourcePath.length === 0) {
      return 'animation';
    }

    return sourcePath.split(/[\\/]/).filter(Boolean).at(-1) ?? sourcePath;
  }
}
