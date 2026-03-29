import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';

export interface ApiMapEntity {
  id: string;
  position: [number, number];
  weaponType?: string;
  classType?: string;
  hp?: number;
}

export interface ApiEnvironmentResponse {
  version: string;
  scenario_id: string;
  config: {
    world_size?: [number, number];
    class_attribute_mapping?: Record<string, Record<string, number>>;
  };
  scenario: {
    drone_positions?: [number, number][];
    target_positions?: [number, number][];
    weapon_assignments?: Record<string, string>;
    target_classes?: string[];
    target_initial_hps?: number[];
  };
}

export interface ViewMapEntity extends ApiMapEntity {
  leftPct: number;
  topPct: number;
  isActive?: boolean;
}

export interface MapRunViewModel {
  scenarioId: string;
  version: string;
}

export interface EngagementLine {
  droneId: string;
  targetId: string;
  droneLeftPct: number;
  droneTopPct: number;
  targetLeftPct: number;
  targetTopPct: number;
}

export interface MapSceneViewModel {
  run: MapRunViewModel;
  title: string;
  width: number;
  height: number;
  drones: ViewMapEntity[];
  targets: ViewMapEntity[];
  engagements?: EngagementLine[];
  hitTargetIds?: Set<string>;
}

@Injectable({
  providedIn: 'root'
})
export class MapService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://localhost:3001/api/environment';

  getMapScene(): Observable<MapSceneViewModel> {
    return this.http.get<ApiEnvironmentResponse>(this.apiUrl).pipe(
      map(payload => this.buildScene(payload))
    );
  }

  private buildScene(payload: ApiEnvironmentResponse): MapSceneViewModel {
    const [rawWidth = 1000, rawHeight = 1000] = payload.config.world_size ?? [];
    const width = rawWidth || 1000;
    const height = rawHeight || 1000;
    const weaponAssignments = payload.scenario.weapon_assignments ?? {};
    const dronePositions = payload.scenario.drone_positions ?? [];
    const targetPositions = payload.scenario.target_positions ?? [];
    const targetClasses = payload.scenario.target_classes ?? [];
    const targetInitialHps = payload.scenario.target_initial_hps ?? [];

    return {
      run: {
        scenarioId: payload.scenario_id || 'unknown',
        version: payload.version || 'unknown'
      },
      title: `World State Map ( Run: ${payload.scenario_id || 'unknown'} )`,
      width,
      height,
      drones: dronePositions.map((position, index) =>
        this.toViewEntity(
          {
            id: `drone_${index}`,
            position,
            weaponType: weaponAssignments[`drone_${index}`] || 'unknown'
          },
          width,
          height
        )
      ),
      targets: targetPositions.map((position, index) => {
        const classType = targetClasses[index] || 'unknown';
        const hp = targetInitialHps[index] ?? 0;

        return this.toViewEntity(
          {
            id: `target_${index}`,
            position,
            classType,
            hp
          },
          width,
          height
        );
      })
    };
  }

  private toViewEntity(entity: ApiMapEntity, width: number, height: number): ViewMapEntity {
    const [x, y] = entity.position;

    return {
      ...entity,
      leftPct: width > 0 ? (x / width) * 100 : 0,
      topPct: height > 0 ? 100 - (y / height) * 100 : 0
    };
  }
}
