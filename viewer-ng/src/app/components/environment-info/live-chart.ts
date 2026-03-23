import { Component, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartOptions, ChartType, registerables, Chart } from 'chart.js';
import { EpisodeStateService } from '../../services/episode-state.service';

Chart.register(...registerables);

@Component({
  selector: 'app-live-chart',
  standalone: true,
  imports: [CommonModule, BaseChartDirective],
  templateUrl: './live-chart.html',
  styleUrl: './live-chart.scss',
})
export class LiveChart {
  private episodeState = inject(EpisodeStateService);

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
          callback: (val) => val + '%'
        }
      },
      x: {
        display: false
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          boxWidth: 8,
          usePointStyle: true,
          font: { size: 10 }
        }
      }
    }
  };

  readonly chartData = computed<ChartConfiguration['data']>(() => {
    const totalSteps = this.episodeState.totalSteps();
    const currentStepIndex = this.episodeState.currentStepIndex();
    
    if (totalSteps === 0 || currentStepIndex < 0) {
      return { datasets: [], labels: [] };
    }

    const hpHist = this.episodeState.hpHistory();
    const activeHist = this.episodeState.activeTargetsHistory();
    
    const sliceEnd = Math.min(currentStepIndex + 1, totalSteps);
    
    const initialHp = hpHist.length > 0 && hpHist[0] > 0 ? hpHist[0] : 1;
    const initialTargets = activeHist.length > 0 && activeHist[0] > 0 ? activeHist[0] : 1;

    const hpData = hpHist.slice(0, sliceEnd).map(v => (v / initialHp) * 100);
    const targetData = activeHist.slice(0, sliceEnd).map(v => (v / initialTargets) * 100);
    
    const fullLabels = Array.from({ length: totalSteps }, (_, i) => String(i + 1));

    return {
      datasets: [
        {
          data: hpData,
          label: 'Total HP',
          borderColor: '#3498db',
          backgroundColor: 'rgba(52,152,219,0.1)',
          fill: true
        },
        {
          data: targetData,
          label: 'Active Targets',
          borderColor: '#e67e22',
          backgroundColor: 'rgba(230,126,34,0.1)',
          fill: true
        }
      ],
      labels: fullLabels
    };
  });
}
