import { Component, computed, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { catchError, finalize, tap } from 'rxjs/operators';
import { of } from 'rxjs';
import { MapService, MapSceneViewModel, ViewMapEntity } from '../../services/map.service';
import { EpisodeStateService } from '../../services/episode-state.service';
import { EnvironmentInfo, DynamicEnvMetrics } from '../environment-info/environment-info';
import { LiveChart } from '../environment-info/live-chart';

const MIN_TARGET_OPACITY = 0.25;

type EpisodeMetricPayload = {
  total_ammo_used?: unknown;
  dmg_eff?: unknown;
  shots_per_target?: unknown;
  total_collisions?: unknown;
};

@Component({
  selector: 'app-map',
  imports: [DecimalPipe, EnvironmentInfo, LiveChart],
  templateUrl: './map.component.html',
  styleUrl: './map.component.scss'
})
export class MapComponent {
  private readonly mapService = inject(MapService);
  private readonly episodeState = inject(EpisodeStateService);

  protected readonly assetPaths = {
    background: '/assets/map/background.png',
    drone: '/assets/map/drone.png',
    target: '/assets/map/target_1.png',
    targetDestroyed: '/assets/map/target_destroyed.png',
    targetFlame: '/assets/map/target_with_flame.png'
  };

  protected readonly isLoading = signal(false);
  protected readonly errorMessage = signal<string | null>(null);

  // Bridge RxJS Observable to Signal using toSignal
  protected readonly scene = toSignal(
    this.mapService.getMapScene().pipe(
      tap(() => {
        this.isLoading.set(true);
        this.errorMessage.set(null);
      }),
      tap({
        next: () => this.isLoading.set(false),
        error: (error: Error) => {
          this.errorMessage.set(error.message || 'Unknown error while loading the map scene.');
          this.isLoading.set(false);
        }
      }),
      catchError(() => of(null))
    ),
    { initialValue: null }
  );

  // Prefer animated scene when playback is active, fall back to static scene
  protected readonly activeScene = computed(() => {
    return this.episodeState.animatedScene() ?? this.scene();
  });

  protected readonly runLabel = computed(() => {
    const episode = this.episodeState.currentEpisode();
    const episodeLabel = episode?.episode?.fileName ?? episode?.episode?.sourcePath;

    if (typeof episodeLabel === 'string' && episodeLabel.length > 0) {
      return episodeLabel.split(/[\\/]/).filter(Boolean).at(-1) ?? episodeLabel;
    }

    return this.activeScene()?.run.scenarioId ?? 'unknown';
  });

  protected readonly episodeNumber = computed(() => {
    return this.episodeState.currentEpisode()?.episode?.episodeNum ?? null;
  });

  protected readonly mapAspectRatio = computed(() => {
    const scene = this.activeScene();

    if (!scene) {
      return '1 / 1';
    }

    const width = scene.width === 1000 ? 1200 : scene.width;
    return `${width} / ${scene.height}`;
  });

  protected readonly targetMaxHpById = computed(() => {
    const targets = this.scene()?.targets ?? [];
    return new Map(targets.map((target) => [target.id, Math.max(0, target.hp ?? 0)]));
  });

  protected readonly episodeMetrics = computed(() => {
    const ep = this.episodeState.currentEpisode();
    if (!ep) {
      return null;
    }

    const episodeMetrics = (ep.metrics ?? {}) as EpisodeMetricPayload;

    return {
      totalSteps: this.readMetricNumber(ep.summary?.total_steps),
      totalAmmoUsed: this.readMetricNumber(episodeMetrics.total_ammo_used),
      dmgEff: this.readMetricNumber(episodeMetrics.dmg_eff),
      shotsPerTarget: this.readMetricNumber(episodeMetrics.shots_per_target),
      totalCollisions: this.readMetricNumber(episodeMetrics.total_collisions),
    };
  });

  protected readonly dynamicEnvMetrics = computed<DynamicEnvMetrics | null>(() => {
    const ep = this.episodeState.currentEpisode();
    if (!ep) {
      return null;
    }

    const staticScene = this.scene();
    const activeScene = this.activeScene();
    if (!staticScene || !activeScene) {
      return null;
    }

    const currentStep = Math.max(0, this.episodeState.currentStepIndex() + 1);

    const maxHp = staticScene.targets.reduce((sum, t) => sum + (t.hp ?? 0), 0);
    const currentHp = activeScene.targets.reduce((sum, t) => sum + (t.hp ?? 0), 0);
    
    const hpReduction = Math.max(0, maxHp - currentHp);
    const avgHpReductionPerStep = hpReduction / Math.max(1, currentStep);

    return {
      currentStep,
      maxHp,
      currentHp,
      avgHpReductionPerStep
    };
  });

  protected trackById(_: number, entity: ViewMapEntity): string {
    return entity.id;
  }

  protected getTargetImage(target: any): string {
    if (target.isActive === false) {
      return this.assetPaths.targetDestroyed;
    }
    const scene = this.activeScene();
    if (scene?.hitTargetIds?.has(target.id)) {
      return this.assetPaths.targetFlame;
    }
    return this.assetPaths.target;
  }

  protected getTargetOpacity(target: ViewMapEntity): number {
    if (target.isActive === false) {
      return 1;
    }

    const maxHp = this.targetMaxHpById().get(target.id);

    if (!maxHp || maxHp <= 0) {
      return 1;
    }

    const currentHp = Math.max(0, target.hp ?? maxHp);
    const hpRatio = Math.min(1, currentHp / maxHp);

    return MIN_TARGET_OPACITY + (1 - MIN_TARGET_OPACITY) * hpRatio;
  }

  protected readMetricNumber(value: unknown): number | null {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }

    return null;
  }
}
