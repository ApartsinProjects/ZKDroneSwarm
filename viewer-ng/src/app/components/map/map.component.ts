import { Component, computed, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { catchError, finalize, tap } from 'rxjs/operators';
import { of } from 'rxjs';
import { MapService, MapSceneViewModel, ViewMapEntity } from '../../services/map.service';
import { EpisodeStateService } from '../../services/episode-state.service';
import { EnvironmentInfo, DynamicEnvMetrics } from '../environment-info/environment-info';

@Component({
  selector: 'app-map',
  imports: [EnvironmentInfo],
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

  protected readonly mapAspectRatio = computed(() => {
    const scene = this.activeScene();

    if (!scene) {
      return '1 / 1';
    }

    return `${scene.width} / ${scene.height}`;
  });

  protected readonly episodeMetrics = computed(() => {
    const ep = this.episodeState.currentEpisode();
    if (!ep) {
      return null;
    }
    return {
      totalSteps: ep.summary?.total_steps ?? ep.steps?.length ?? 0,
      totalAmmoUsed: ep.summary?.metrics?.total_ammo_used ?? 0
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
}
