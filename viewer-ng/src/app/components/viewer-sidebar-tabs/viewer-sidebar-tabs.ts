import { Component, signal } from '@angular/core';

type SidebarTabId = 'hp-active-target' | 'embedding-visualization';

interface SidebarTabDefinition {
  id: SidebarTabId;
  label: string;
}

const SIDEBAR_TABS: ReadonlyArray<SidebarTabDefinition> = [
  { id: 'hp-active-target', label: 'HP & Active Target' },
  { id: 'embedding-visualization', label: 'Embedding Visualization' },
];

@Component({
  selector: 'app-viewer-sidebar-tabs',
  imports: [],
  templateUrl: './viewer-sidebar-tabs.html',
  styleUrl: './viewer-sidebar-tabs.scss',
})
export class ViewerSidebarTabs {
  protected readonly tabs = SIDEBAR_TABS;
  protected readonly activeTab = signal<SidebarTabId>('hp-active-target');

  protected selectTab(tabId: SidebarTabId): void {
    this.activeTab.set(tabId);
  }

  protected tabButtonId(tabId: SidebarTabId): string {
    return `viewer-sidebar-tab-${tabId}`;
  }

  protected tabPanelId(tabId: SidebarTabId): string {
    return `viewer-sidebar-panel-${tabId}`;
  }
}
