import { Component } from '@angular/core';
import { MapComponent } from './components/map/map.component';
import { PoliciesComponent } from './components/policies/policies.component';
import { ActionsPlayer } from './components/actions-player/actions-player';

@Component({
  selector: 'app-root',
  imports: [MapComponent, ActionsPlayer, PoliciesComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {}
