import { Component, computed, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, ChartType, registerables, Chart } from 'chart.js';
import { EpisodeSnapshot } from '../../../services/cross-episode-browser.service';

Chart.register(...registerables);

@Component({
  selector: 'app-episode-analysis-chart',
  standalone: true,
  imports: [CommonModule, BaseChartDirective],
  template: `
    <div class="analysis-chart-container">
      @if (chartData().datasets.length > 0) {
        <canvas baseChart
          [data]="chartData()"
          [options]="chartOptions"
          [type]="chartType">
        </canvas>
      } @else {
        <div class="chart-placeholder">
          <span>No episode data available</span>
        </div>
      }
    </div>
  `,
  styles: [`
    .analysis-chart-container {
      position: relative;
      width: 100%;
      height: 220px;
      margin-top: 1rem;
    }
    .chart-placeholder {
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-muted, #7f8c8d);
      font-size: 0.9rem;
      border: 1px dashed var(--border-color, #ecf0f1);
      border-radius: 8px;
    }
  `]
})
export class EpisodeAnalysisChart {
  readonly snapshot = input.required<EpisodeSnapshot | null>();

  readonly chartType: ChartType = 'line';
  
  readonly chartOptions: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    elements: {
      line: {
        tension: 0.1,
        borderWidth: 2
      },
      point: {
        radius: 0,
        hitRadius: 10,
        hoverRadius: 4
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: (val) => val + '%',
          font: { size: 9 }
        }
      },
      x: {
        display: true,
        grid: { display: false },
        ticks: {
          maxTicksLimit: 10,
          font: { size: 8 }
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        align: 'end',
        labels: {
          boxWidth: 4,
          boxHeight: 4,
          useBorderRadius: true,
          borderRadius: 2,
          font: { size: 9 }
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        titleColor: '#2c3e50',
        bodyColor: '#2c3e50',
        borderColor: '#ecf0f1',
        borderWidth: 1,
        padding: 8,
        bodyFont: { size: 10 },
        titleFont: { size: 10, weight: 'bold' },
        callbacks: {
            label: (context) => {
                let label = context.dataset.label || '';
                if (label) label += ': ';
                if (context.parsed.y !== null) label += context.parsed.y.toFixed(1) + '%';
                return label;
            }
        }
      }
    }
  };

  readonly chartData = computed<ChartConfiguration['data']>(() => {
    const s = this.snapshot();
    if (!s || s.totalSteps === 0) {
      return { datasets: [], labels: [] };
    }

    const hpHist = s.hpHistory;
    const activeHist = s.activeTargetsHistory;
    
    const initialHp = hpHist.length > 0 && hpHist[0] > 0 ? hpHist[0] : 1;
    const initialTargets = activeHist.length > 0 && activeHist[0] > 0 ? activeHist[0] : 1;

    const hpData = hpHist.map(v => (v / initialHp) * 100);
    const targetData = activeHist.map(v => (v / initialTargets) * 100);
    
    const fullLabels = Array.from({ length: s.totalSteps }, (_, i) => String(i + 1));

    return {
      datasets: [
        {
          data: hpData,
          label: 'Total HP',
          borderColor: '#3498db',
          backgroundColor: 'rgba(52,152,219,0.05)',
          pointBackgroundColor: '#3498db',
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 3
        },
        {
          data: targetData,
          label: 'Active Targets',
          borderColor: '#e67e22',
          backgroundColor: 'rgba(230,126,34,0.05)',
          pointBackgroundColor: '#e67e22',
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 3
        }
      ],
      labels: fullLabels
    };
  });
}
