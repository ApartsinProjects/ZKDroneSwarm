import { Injectable, signal } from '@angular/core';

type SidebarTabId = 'hp-active-target' | 'embedding-visualization' | 'latent-world';

@Injectable({
  providedIn: 'root'
})
export class SidebarTabService {
  private readonly _activeTab = signal<SidebarTabId | null>(null);
  
  readonly activeTab = this._activeTab.asReadonly();
  
  setActiveTab(tabId: SidebarTabId): void {
    this._activeTab.set(tabId);
  }
}
