import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReportService, ComparisonResponse } from '../../../services/report.service';
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
  protected readonly isLoading = signal(true);
  protected readonly error = signal<string | null>(null);

  constructor() {
    this.reportService.getComparison().subscribe({
      next: (data) => {
        this.comparisonData.set(data);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.message ?? 'Failed to load comparison data.');
        this.isLoading.set(false);
      },
    });
  }

  protected selectSubTab(tabId: ReportSubTab): void {
    this.activeSubTab.set(tabId);
  }
}
