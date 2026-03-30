import { Component, signal, inject, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CrossEpisodeBrowserService } from '../../services/cross-episode-browser.service';
import { EpisodeAnalysisChart } from './episode-analysis-chart';
import { EmbeddingBrowserService } from '../../services/embedding-browser.service';
import { EmbeddingVisualizationPanel } from './embedding-visualization-panel';
import { LatentWorldService } from '../../services/latent-world.service';
import { LatentWorldVisualizationPanel } from './latent-world-visualization-panel';
import { SidebarTabService } from '../../services/sidebar-tab.service';

type SidebarTabId = 'hp-active-target' | 'embedding-visualization' | 'latent-world';

interface SidebarTabDefinition {
  id: SidebarTabId;
  label: string;
}

const SIDEBAR_TABS: ReadonlyArray<SidebarTabDefinition> = [
  { id: 'latent-world', label: 'Latent World' },
  { id: 'hp-active-target', label: 'HP & Active Target' },
  { id: 'embedding-visualization', label: 'Embedding Visualization' }
];

@Component({
  selector: 'app-viewer-sidebar-tabs',
  standalone: true,
  imports: [CommonModule, EpisodeAnalysisChart, EmbeddingVisualizationPanel, LatentWorldVisualizationPanel],
  templateUrl: './viewer-sidebar-tabs.html',
  styleUrl: './viewer-sidebar-tabs.scss',
})
export class ViewerSidebarTabs {
  private browserService = inject(CrossEpisodeBrowserService);
  private embeddingBrowser = inject(EmbeddingBrowserService);
  private latentWorldService = inject(LatentWorldService);
  private sidebarTabService = inject(SidebarTabService);

  protected readonly tabs = SIDEBAR_TABS;
  protected readonly activeTab = signal<SidebarTabId>('latent-world');

  protected readonly currentIndex = this.browserService.currentIndex;
  protected readonly currentSnapshot = this.browserService.currentEpisodeSnapshot;
  protected readonly isLoading = this.browserService.isLoading;
  protected readonly totalEpisodes = this.browserService.totalEpisodes;
  protected readonly embeddingSnapshots = this.embeddingBrowser.snapshots;
  protected readonly embeddingCurrentSnapshot = this.embeddingBrowser.currentSnapshot;
  protected readonly embeddingIsLoading = this.embeddingBrowser.isLoading;
  protected readonly embeddingError = this.embeddingBrowser.error;
  protected readonly embeddingSelectedAgent = this.embeddingBrowser.selectedAgent;
  protected readonly latentVectors = signal<any>(null);

  constructor() {
    this.latentWorldService.getLatentVectors().subscribe({
      next: (data) => this.latentVectors.set(data),
      error: (err) => console.error('Failed to load latent vectors:', err)
    });

    // Sync activeTab with service state
    effect(() => {
      const serviceTab = this.sidebarTabService.activeTab();
      if (serviceTab !== null) {
        this.activeTab.set(serviceTab);
      }
    });
  }

  protected selectTab(tabId: SidebarTabId): void {
    this.activeTab.set(tabId);
  }

  protected onSliderInput(event: Event): void {
    const value = (event.target as HTMLInputElement).valueAsNumber;
    this.browserService.setIndex(value);
  }

  protected tabButtonId(tabId: SidebarTabId): string {
    return `viewer-sidebar-tab-${tabId}`;
  }

  protected tabPanelId(tabId: SidebarTabId): string {
    return `viewer-sidebar-panel-${tabId}`;
  }

  protected onEmbeddingAgentSelected(index: number): void {
    this.embeddingBrowser.setSelectedAgent(index);
  }
}
