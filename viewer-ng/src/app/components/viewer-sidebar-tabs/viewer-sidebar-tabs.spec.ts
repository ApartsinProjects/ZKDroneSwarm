import { ComponentFixture, TestBed } from '@angular/core/testing';
import { computed, signal } from '@angular/core';
import { ViewerSidebarTabs } from './viewer-sidebar-tabs';
import { CrossEpisodeBrowserService } from '../../services/cross-episode-browser.service';
import { EmbeddingBrowserService } from '../../services/embedding-browser.service';

class StubCrossEpisodeBrowserService {
  private readonly _episodes = signal<any[]>([
    { episode: { episodeNum: 1 }, steps: [] },
  ]);
  private readonly _currentIndex = signal(0);
  private readonly _isLoading = signal(false);

  readonly episodes = this._episodes.asReadonly();
  readonly currentIndex = this._currentIndex.asReadonly();
  readonly isLoading = this._isLoading.asReadonly();
  readonly totalEpisodes = computed(() => this._episodes().length);
  readonly currentEpisodeSnapshot = computed(() => ({
    episodeNum: this._episodes()[this._currentIndex()]?.episode?.episodeNum ?? null,
    totalSteps: 0,
    hpHistory: [],
    activeTargetsHistory: [],
    metricRows: [
      {
        label: 'Total Ammo Used',
        value: '10',
        numericValue: 10,
        progress: {
          deltaValue: -2,
          deltaLabel: '-2',
          deltaTone: 'better',
          normalizedScore: 0.7,
          percentileLabel: '70%',
          isFixed: false,
        },
        baseline: {
          label: '+12%',
          tone: 'better',
        },
      },
    ],
    totalNetDamageValue: '120',
  }));

  setIndex(index: number): void {
    this._currentIndex.set(index);
  }
}

class StubEmbeddingBrowserService {
  private readonly _snapshots = signal<any[]>([]);
  private readonly _isLoading = signal(false);
  private readonly _error = signal<string | null>(null);
  private readonly _selectedAgent = signal(0);

  readonly snapshots = this._snapshots.asReadonly();
  readonly isLoading = this._isLoading.asReadonly();
  readonly error = this._error.asReadonly();
  readonly selectedAgent = this._selectedAgent.asReadonly();
  readonly currentSnapshot = computed(() => this._snapshots()[0] ?? null);

  setSelectedAgent(index: number): void {
    this._selectedAgent.set(index);
  }

  setLoading(value: boolean): void {
    this._isLoading.set(value);
  }
}

describe('ViewerSidebarTabs', () => {
  let fixture: ComponentFixture<ViewerSidebarTabs>;
  let element: HTMLElement;
  let embeddingBrowser: StubEmbeddingBrowserService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ViewerSidebarTabs],
      providers: [
        { provide: CrossEpisodeBrowserService, useClass: StubCrossEpisodeBrowserService },
        { provide: EmbeddingBrowserService, useClass: StubEmbeddingBrowserService },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(ViewerSidebarTabs);
    fixture.detectChanges();
    element = fixture.nativeElement as HTMLElement;
    embeddingBrowser = TestBed.inject(EmbeddingBrowserService) as unknown as StubEmbeddingBrowserService;
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

  it('renders the random baseline table alongside the episode metrics tables', () => {
    const cardTitles = Array.from(
      element.querySelectorAll<HTMLElement>('.analysis-metrics-card__title'),
    ).map((title) => title.textContent?.trim());

    expect(cardTitles).toContain('Episode Metrics');
    expect(cardTitles).toContain('Episode Progress');
    expect(cardTitles).toContain('Vs Random Baseline');
    expect(element.textContent).toContain('+12%');
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

  it('shows an empty embedding placeholder when no embedding snapshot matches the selected episode', () => {
    const tabs = element.querySelectorAll<HTMLElement>('[role="tab"]');

    tabs[1].click();
    fixture.detectChanges();

    expect(element.textContent).toContain('No embedding snapshot is available for the selected episode.');
  });

  it('shows the embedding loading state inside the second tab', () => {
    const tabs = element.querySelectorAll<HTMLElement>('[role="tab"]');

    embeddingBrowser.setLoading(true);
    tabs[1].click();
    fixture.detectChanges();

    expect(element.textContent).toContain('Loading embedding snapshots...');
  });
});
