import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ViewerSidebarTabs } from './viewer-sidebar-tabs';

describe('ViewerSidebarTabs', () => {
  let fixture: ComponentFixture<ViewerSidebarTabs>;
  let element: HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ViewerSidebarTabs],
    }).compileComponents();

    fixture = TestBed.createComponent(ViewerSidebarTabs);
    fixture.detectChanges();
    element = fixture.nativeElement as HTMLElement;
  });

  it('renders both tab labels', () => {
    const tabs = Array.from(element.querySelectorAll('[role="tab"]')).map(
      (tab) => tab.textContent?.trim(),
    );

    expect(tabs).toEqual(['HP & Active Target', 'Embedding Visualization']);
  });

  it('activates HP & Active Target by default and keeps both panels in the DOM', () => {
    const tabs = element.querySelectorAll<HTMLElement>('[role="tab"]');
    const panels = element.querySelectorAll<HTMLElement>('[role="tabpanel"]');

    expect(tabs[0].getAttribute('aria-selected')).toBe('true');
    expect(tabs[1].getAttribute('aria-selected')).toBe('false');
    expect(panels.length).toBe(2);
    expect(panels[0].hidden).toBeFalse();
    expect(panels[1].hidden).toBeTrue();
  });

  it('switches the active tab when Embedding Visualization is clicked', () => {
    const tabs = element.querySelectorAll<HTMLElement>('[role="tab"]');
    const panels = element.querySelectorAll<HTMLElement>('[role="tabpanel"]');

    tabs[1].click();
    fixture.detectChanges();

    expect(tabs[0].getAttribute('aria-selected')).toBe('false');
    expect(tabs[1].getAttribute('aria-selected')).toBe('true');
    expect(panels[0].hidden).toBeTrue();
    expect(panels[1].hidden).toBeFalse();
  });
});
