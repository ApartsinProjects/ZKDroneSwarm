import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, map } from 'rxjs';

export interface PoliciesResponse {
  policies: string[];
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

  getBestEpisode(policyId: string): Observable<any> {
    return this.http.get<any>(`http://localhost:3001/api/policies/${policyId}/episodes/best`);
  }

  getAllEpisodes(policyId: string): Observable<any[]> {
    return this.http.get<any[]>(`http://localhost:3001/api/policies/${policyId}/episodes`);
  }
}
