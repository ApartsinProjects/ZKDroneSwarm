import { Component, computed, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { catchError, finalize, tap } from 'rxjs/operators';
import { of } from 'rxjs';
import { MapService, MapSceneViewModel, ViewMapEntity } from '../../services/map.service';

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrl: './map.component.scss'
})
export class MapComponent {
  private readonly mapService = inject(MapService);

  protected readonly assetPaths = {
    background: '/assets/map/background.png',
    drone: '/assets/map/drone.png',
    target: '/assets/map/target_1.png'
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

  protected readonly mapAspectRatio = computed(() => {
    const scene = this.scene();

    if (!scene) {
      return '1 / 1';
    }

    return `${scene.width} / ${scene.height}`;
  });

  protected trackById(_: number, entity: ViewMapEntity): string {
    return entity.id;
  }
}
