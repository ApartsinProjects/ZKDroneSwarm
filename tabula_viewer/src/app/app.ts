import { Component } from '@angular/core';
import { MapComponent } from './components/map/map.component';
import { PoliciesComponent } from './components/policies/policies.component';
import { ActionsPlayer } from './components/actions-player/actions-player';
import { ViewerSidebarTabs } from './components/viewer-sidebar-tabs/viewer-sidebar-tabs';

@Component({
  selector: 'app-root',
  imports: [MapComponent, ActionsPlayer, PoliciesComponent, ViewerSidebarTabs],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {}
