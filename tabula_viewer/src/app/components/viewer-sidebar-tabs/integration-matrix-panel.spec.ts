import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LearningStateEpisodeDto } from '../../services/policies.service';
import { IntegrationMatrixPanel } from './integration-matrix-panel';

describe('IntegrationMatrixPanel', () => {
  let fixture: ComponentFixture<IntegrationMatrixPanel>;
  let element: HTMLElement;

  const snapshotWithPredicted: LearningStateEpisodeDto = {
    episode: {
      fileName: 'learning_state_ep01.json',
      episodeNum: 1,
      policyType: 'matrix_factorization_cf',
      sourcePath: 'logs/run_x/policy_x/learning_state/learning_state_ep01.json',
    },
    learningState: {
      version: '1.0',
      scenarioId: 'scenario_x',
      numAgents: 1,
      numTargets: 2,
      latentDim: 2,
      episodeState: {
        agents: [
          {
            integration_matrix: {
              M_avg: [[0.1, 0.2]],
              M_pred: [[0.9, 0.8]],
              M_count: [[3, 5]],
            },
          },
        ],
      },
    },
  };

  const snapshotObservedOnly: LearningStateEpisodeDto = {
    ...snapshotWithPredicted,
    learningState: {
      ...snapshotWithPredicted.learningState,
      episodeState: {
        agents: [
          {
            integration_matrix: {
              M_avg: [[0.4, 0.6]],
              M_count: [[1, 2]],
            },
          },
        ],
      },
    },
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [IntegrationMatrixPanel],
    }).compileComponents();
  });

  function render(snapshot: LearningStateEpisodeDto): void {
    fixture = TestBed.createComponent(IntegrationMatrixPanel);
    fixture.componentRef.setInput('snapshots', [snapshot]);
    fixture.componentRef.setInput('currentSnapshot', snapshot);
    fixture.componentRef.setInput('selectedAgent', 0);
    fixture.detectChanges();
    element = fixture.nativeElement as HTMLElement;
  }

  it('renders observed on top and predicted on bottom by default', () => {
    render(snapshotWithPredicted);

    const selects = element.querySelectorAll<HTMLSelectElement>('select');
    const tooltips = Array.from(element.querySelectorAll('title')).map((node) => node.textContent?.trim());

    expect(selects[0].value).toBe('observed');
    expect(selects[1].value).toBe('predicted');
    expect(tooltips).toContain('Observed · Agent 0 → Target 0: 0.100');
    expect(tooltips).toContain('Predicted · Agent 0 → Target 0: 0.900');
  });

  it('lets the bottom selector switch to count without affecting the top view', () => {
    render(snapshotWithPredicted);

    const selects = element.querySelectorAll<HTMLSelectElement>('select');
    selects[1].value = 'count';
    selects[1].dispatchEvent(new Event('change'));
    fixture.detectChanges();

    const tooltips = Array.from(element.querySelectorAll('title')).map((node) => node.textContent?.trim());

    expect(selects[0].value).toBe('observed');
    expect(selects[1].value).toBe('count');
    expect(tooltips).toContain('Observed · Agent 0 → Target 0: 0.100');
    expect(tooltips).toContain('Count · Agent 0 → Target 0: 3.000');
  });

  it('falls back gracefully when predicted data is absent', () => {
    render(snapshotObservedOnly);

    const selects = element.querySelectorAll<HTMLSelectElement>('select');
    const predictedOption = element.querySelectorAll<HTMLOptionElement>('option[value="predicted"]')[0];

    expect(selects[0].value).toBe('observed');
    expect(selects[1].value).toBe('observed');
    expect(predictedOption.disabled).toBeTrue();
    expect(element.textContent).not.toContain('No matrix data is available for the bottom comparison slot.');
  });
});
