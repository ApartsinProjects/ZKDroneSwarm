import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

export interface PoliciesResponse {
  policies: string[];
}

export interface EpisodeDto {
  episode: {
    fileName: string;
    episodeNum: number | null;
    version: string;
    policyType: string;
    sourcePath: string;
  };
  scenario: Record<string, unknown>;
  steps: Array<Record<string, unknown>>;
  summary: Record<string, unknown>;
}

export interface LearningStateEpisodeDto {
  episode: {
    fileName: string;
    episodeNum: number | null;
    policyType: string;
    sourcePath: string;
  };
  learningState: {
    version: string;
    scenarioId: string | null;
    numAgents: number | null;
    numTargets: number | null;
    latentDim: number | null;
    episodeState: {
      agents: Array<Record<string, unknown>>;
    } | null;
  };
}

@Injectable({
  providedIn: 'root'
})
export class PoliciesService {
  private http = inject(HttpClient);

  getPolicies(): Observable<string[]> {
    return this.http.get<PoliciesResponse>('http://localhost:3001/api/policies').pipe(
      map(response => response.policies)
    );
  }

  getBestEpisode(policyId: string): Observable<EpisodeDto> {
    return this.http.get<EpisodeDto>(`http://localhost:3001/api/policies/${policyId}/episodes/best`);
  }

  getAllEpisodes(policyId: string): Observable<EpisodeDto[]> {
    return this.http.get<EpisodeDto[]>(`http://localhost:3001/api/policies/${policyId}/episodes`);
  }

  getAllLearningStates(policyId: string): Observable<LearningStateEpisodeDto[]> {
    return this.http.get<LearningStateEpisodeDto[]>(`http://localhost:3001/api/policies/${policyId}/learning-state`);
  }
}
