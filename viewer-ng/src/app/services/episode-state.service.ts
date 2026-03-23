import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class EpisodeStateService {
  private readonly _currentEpisode = signal<any | null>(null);

  readonly currentEpisode = this._currentEpisode.asReadonly();

  setEpisode(episode: any): void {
    this._currentEpisode.set(episode);
  }

  clearEpisode(): void {
    this._currentEpisode.set(null);
  }
}
