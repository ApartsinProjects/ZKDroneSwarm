import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { forkJoin } from 'rxjs';
import { ReportService, ComparisonResponse, ReportManifest } from '../../../services/report.service';
import { ComparisonTab } from './comparison-tab/comparison-tab';

type ReportSubTab = 'comparison';

interface ReportSubTabDefinition {
  id: ReportSubTab;
  label: string;
}

const REPORT_SUB_TABS: ReadonlyArray<ReportSubTabDefinition> = [
  { id: 'comparison', label: 'Comparison' },
];

@Component({
  selector: 'app-report-panel',
  standalone: true,
  imports: [CommonModule, ComparisonTab],
  templateUrl: './report-panel.html',
  styleUrl: './report-panel.scss',
})
export class ReportPanel {
  private readonly reportService = inject(ReportService);

  protected readonly subTabs = REPORT_SUB_TABS;
  protected readonly activeSubTab = signal<ReportSubTab>('comparison');
  protected readonly comparisonData = signal<ComparisonResponse | null>(null);
  protected readonly manifestData = signal<ReportManifest | null>(null);
  protected readonly isLoading = signal(true);
  protected readonly error = signal<string | null>(null);
  protected readonly bestEpisodeNumber = signal<number | null>(null);

  constructor() {
    forkJoin({
      comparison: this.reportService.getComparison(),
      manifest: this.reportService.getManifest(),
    }).subscribe({
      next: ({ comparison, manifest }) => {
        this.comparisonData.set(comparison);
        this.manifestData.set(manifest);
        
        // Extract episode number from path like "policies/matrix_factorization_cf/episodes/episode_ep35.json"
        const match = manifest.artifacts.best_episode_path.match(/episode_ep(\d+)\.json$/);
        if (match) {
          this.bestEpisodeNumber.set(parseInt(match[1], 10));
        }
        
        this.isLoading.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.message ?? 'Failed to load report data.');
        this.isLoading.set(false);
      },
    });
  }

  protected selectSubTab(tabId: ReportSubTab): void {
    this.activeSubTab.set(tabId);
  }
}
