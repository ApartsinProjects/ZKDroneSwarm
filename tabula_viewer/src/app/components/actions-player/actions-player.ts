import { Component, inject } from '@angular/core';
import { EpisodeStateService } from '../../services/episode-state.service';

@Component({
  selector: 'app-actions-player',
  imports: [],
  templateUrl: './actions-player.html',
  styleUrl: './actions-player.scss',
})
export class ActionsPlayer {
  private readonly episodeState = inject(EpisodeStateService);

  readonly currentEpisode = this.episodeState.currentEpisode;

  onPlay(): void {
    const episode = this.currentEpisode();
    if (!episode) {
      console.warn('No episode loaded — select a policy first.');
      return;
    }
    this.episodeState.play();
  }

  onPause(): void {
    this.episodeState.pause();
  }
}
