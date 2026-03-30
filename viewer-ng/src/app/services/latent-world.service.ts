import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';

export interface LatentVectorEntity {
  id: string;
  mode_id: number;
  tsne_coords: [number, number];
}

export interface LatentVectorsData {
  drones: LatentVectorEntity[];
  targets: LatentVectorEntity[];
}

export interface LatentWorldResponse {
  scenario: {
    latent_vectors?: LatentVectorsData;
  };
}

@Injectable({
  providedIn: 'root'
})
export class LatentWorldService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://localhost:3001/api/environment';

  getLatentVectors(): Observable<LatentVectorsData | null> {
    return this.http.get<LatentWorldResponse>(this.apiUrl).pipe(
      map(response => response.scenario.latent_vectors ?? null)
    );
  }
}
