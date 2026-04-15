import { Component, input, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ComparisonMetric } from '../../../../services/report.service';

interface MetricRow {
  key: string;
  metric: ComparisonMetric;
  mfDisplay: string;
  randomDisplay: string;
  oracleDisplay: string;
  vsRandomPct: string | null;
  vsOraclePct: string | null;
  vsRandomTone: 'better' | 'worse' | 'neutral' | null;
  vsOracleTone: 'better' | 'worse' | 'neutral' | null;
}

interface CategoryGroup {
  category: string;
  label: string;
  rows: MetricRow[];
}

const CATEGORY_LABELS: Record<string, string> = {
  task_completion: 'Task Completion',
  efficiency: 'Efficiency',
  coordination: 'Coordination',
  environment: 'Environment',
};

const CATEGORY_ORDER = ['task_completion', 'efficiency', 'coordination', 'environment'];

const EXCLUDED_METRICS = ['total_ammo_used'];

function formatValue(value: number | boolean, unit: string): string {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (unit === 'ratio') return value.toFixed(4);
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

function formatPct(pct: number): string {
  const sign = pct > 0 ? '+' : '';
  return `${sign}${pct.toFixed(1)}%`;
}

function getTone(pct: number, direction: string): 'better' | 'worse' | 'neutral' {
  if (Math.abs(pct) < 0.1) return 'neutral';
  if (direction === 'higher_is_better') return pct > 0 ? 'better' : 'worse';
  if (direction === 'lower_is_better') return pct < 0 ? 'better' : 'worse';
  return 'neutral';
}

@Component({
  selector: 'app-comparison-tab',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './comparison-tab.html',
  styleUrl: './comparison-tab.scss',
})
export class ComparisonTab {
  readonly data = input.required<Record<string, ComparisonMetric>>();
  readonly episodeNumber = input<number | null>(null);

  protected readonly groups = computed<CategoryGroup[]>(() => {
    const metrics = this.data();
    const grouped = new Map<string, MetricRow[]>();

    for (const [key, metric] of Object.entries(metrics)) {
      if (EXCLUDED_METRICS.includes(key)) continue;
      
      const cat = metric.category;
      if (!grouped.has(cat)) grouped.set(cat, []);

      const row: MetricRow = {
        key,
        metric,
        mfDisplay: formatValue(metric.mf_value, metric.unit),
        randomDisplay: formatValue(metric.random, metric.unit),
        oracleDisplay: formatValue(metric.max_damage_oracle, metric.unit),
        vsRandomPct: metric.mf_vs_random_pct != null ? formatPct(metric.mf_vs_random_pct) : null,
        vsOraclePct: metric.mf_vs_max_damage_oracle_pct != null ? formatPct(metric.mf_vs_max_damage_oracle_pct) : null,
        vsRandomTone: metric.mf_vs_random_pct != null ? getTone(metric.mf_vs_random_pct, metric.direction) : null,
        vsOracleTone: metric.mf_vs_max_damage_oracle_pct != null ? getTone(metric.mf_vs_max_damage_oracle_pct, metric.direction) : null,
      };

      grouped.get(cat)!.push(row);
    }

    return CATEGORY_ORDER
      .filter(cat => grouped.has(cat))
      .map(cat => ({
        category: cat,
        label: CATEGORY_LABELS[cat] ?? cat,
        rows: grouped.get(cat)!,
      }));
  });
}
