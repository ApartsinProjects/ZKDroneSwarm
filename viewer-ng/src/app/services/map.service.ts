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

export interface ApiMapResponse {
  episode: {
    fileName: string;
    episodeNum: number | null;
    version: string;
    policyType: string;
    sourcePath: string;
  };
  map: {
    title: string;
    worldSize: {
      width: number;
      height: number;
    };
    drones: ApiMapEntity[];
    targets: ApiMapEntity[];
  };
}

export interface ViewMapEntity extends ApiMapEntity {
  leftPct: number;
  topPct: number;
}

export interface MapSceneViewModel {
  episode: ApiMapResponse['episode'];
  title: string;
  width: number;
  height: number;
  drones: ViewMapEntity[];
  targets: ViewMapEntity[];
}

@Injectable({
  providedIn: 'root'
})
export class MapService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://localhost:3001/api/map';

  getMapScene(): Observable<MapSceneViewModel> {
    return this.http.get<ApiMapResponse>(this.apiUrl).pipe(
      map(payload => this.buildScene(payload))
    );
  }

  private buildScene(payload: ApiMapResponse): MapSceneViewModel {
    const width = payload.map.worldSize.width || 1000;
    const height = payload.map.worldSize.height || 1000;

    return {
      episode: payload.episode,
      title: payload.map.title,
      width,
      height,
      drones: payload.map.drones.map((entity) => this.toViewEntity(entity, width, height)),
      targets: payload.map.targets.map((entity) => this.toViewEntity(entity, width, height))
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
