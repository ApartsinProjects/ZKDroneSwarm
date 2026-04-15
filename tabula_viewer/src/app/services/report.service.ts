import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ComparisonMetric {
  category: string;
  description: string;
  direction: 'higher_is_better' | 'lower_is_better' | 'context_dependent';
  display_name: string;
  formula: string;
  max_damage_oracle: number | boolean;
  mf_value: number | boolean;
  mf_vs_max_damage_oracle_pct?: number;
  mf_vs_random_pct?: number;
  random: number | boolean;
  unit: string;
}

export interface KeyFinding {
  type: string;
  metric?: string;
  category?: string;
  direction?: string;
  first_value?: number;
  last_value?: number;
  status?: string;
  best_episode?: number;
  agent_count?: number;
  episode?: number;
  unique_best_target_count?: number;
}

export interface ComparisonResponse {
  comparison_vs_baseline: Record<string, ComparisonMetric>;
  key_findings: KeyFinding[];
}

@Injectable({ providedIn: 'root' })
export class ReportService {
  private readonly http = inject(HttpClient);
  private readonly apiUrl = 'http://localhost:3001/api/report';

  getComparison(): Observable<ComparisonResponse> {
    return this.http.get<ComparisonResponse>(`${this.apiUrl}/comparison`);
  }
}
