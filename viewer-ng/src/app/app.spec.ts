import { TestBed } from '@angular/core/testing';
import { Component } from '@angular/core';
import { App } from './app';
import { ActionsPlayer } from './components/actions-player/actions-player';
import { MapComponent } from './components/map/map.component';
import { PoliciesComponent } from './components/policies/policies.component';

@Component({
  selector: 'app-map',
  template: '',
})
class StubMapComponent {}

@Component({
  selector: 'app-policies',
  template: '',
})
class StubPoliciesComponent {}

@Component({
  selector: 'app-actions-player',
  template: '',
})
class StubActionsPlayerComponent {}

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
    })
      .overrideComponent(App, {
        remove: {
          imports: [MapComponent, PoliciesComponent, ActionsPlayer],
        },
        add: {
          imports: [StubMapComponent, StubPoliciesComponent, StubActionsPlayerComponent],
        },
      })
      .compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the sidebar tabs scaffold', () => {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('app-viewer-sidebar-tabs')).not.toBeNull();
    expect(compiled.textContent).toContain('HP & Active Target');
    expect(compiled.textContent).toContain('Embedding Visualization');
  });
});
