import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { PoliciesService } from '../../services/policies.service';
import { EpisodeStateService } from '../../services/episode-state.service';
import { CrossEpisodeBrowserService } from '../../services/cross-episode-browser.service';
import { EmbeddingBrowserService } from '../../services/embedding-browser.service';
import { SidebarTabService } from '../../services/sidebar-tab.service';

@Component({
  selector: 'app-policies',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './policies.component.html',
  styleUrl: './policies.component.scss'
})
export class PoliciesComponent {
  private policiesService = inject(PoliciesService);
  private episodeState = inject(EpisodeStateService);
  private crossEpisodeBrowser = inject(CrossEpisodeBrowserService);
  private embeddingBrowser = inject(EmbeddingBrowserService);
  private sidebarTabService = inject(SidebarTabService);
  public policies = toSignal(this.policiesService.getPolicies());
  public selectedPolicy = signal<string | null>(null);
  
  onPolicyClick(policyName: string): void {
    this.selectedPolicy.set(policyName);
    console.log(`Selected policy: ${policyName}`);
    this.policiesService.getBestEpisode(policyName).subscribe({
      next: (episode) => {
        this.episodeState.setEpisode(episode);
        console.log('Best episode retrieved:', episode);
      },
      error: (err) => {
        console.error('Error fetching best episode:', err);
      }
    });

    this.crossEpisodeBrowser.loadEpisodes(policyName);
    this.embeddingBrowser.loadSnapshots(policyName);
  }
}
