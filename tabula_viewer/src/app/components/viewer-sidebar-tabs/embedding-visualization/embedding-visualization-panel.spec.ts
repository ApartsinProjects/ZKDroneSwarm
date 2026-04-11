import { ComponentFixture, TestBed } from '@angular/core/testing';
import { EmbeddingVisualizationPanel } from './embedding-visualization-panel';
import { LearningStateEpisodeDto } from '../../services/policies.service';

describe('EmbeddingVisualizationPanel', () => {
  let fixture: ComponentFixture<EmbeddingVisualizationPanel>;
  let element: HTMLElement;

  const snapshots: LearningStateEpisodeDto[] = [
    {
      episode: {
        fileName: 'learning_state_ep01.json',
        episodeNum: 1,
        policyType: 'policy_x',
        sourcePath: 'logs/run_x/policy_x/learning_state/learning_state_ep01.json',
      },
      learningState: {
        version: '1.0',
        scenarioId: 'scenario_x',
        numAgents: 1,
        numTargets: 3,
        latentDim: 2,
        episodeState: {
          target_classes: ['A', 'A', 'B'],
          agents: [
            {
              agent_lv: [0.1, 0.2],
              target_lv: [[0.2, 0.3], [0.4, 0.5], [0.7, 0.8]],
            },
          ],
        },
      },
    },
    {
      episode: {
        fileName: 'learning_state_ep02.json',
        episodeNum: 2,
        policyType: 'policy_x',
        sourcePath: 'logs/run_x/policy_x/learning_state/learning_state_ep02.json',
      },
      learningState: {
        version: '1.0',
        scenarioId: 'scenario_x',
        numAgents: 1,
        numTargets: 3,
        latentDim: 2,
        episodeState: {
          target_classes: ['A', 'A', 'B'],
          agents: [
            {
              agent_lv: [0.15, 0.25],
              target_lv: [[0.25, 0.35], [0.45, 0.55], [0.75, 0.85]],
            },
          ],
        },
      },
    },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EmbeddingVisualizationPanel],
    }).compileComponents();

    fixture = TestBed.createComponent(EmbeddingVisualizationPanel);
    fixture.componentRef.setInput('snapshots', snapshots);
    fixture.componentRef.setInput('currentSnapshot', snapshots[1]);
    fixture.componentRef.setInput('selectedAgent', 0);
    fixture.detectChanges();
    element = fixture.nativeElement as HTMLElement;
  });

  it('colors same-class targets consistently and shows class-aware labels', () => {
    const targetNodes = Array.from(
      element.querySelectorAll<SVGCircleElement>('.embedding-node--target'),
    );
    const labels = Array.from(element.querySelectorAll('text')).map((node) => node.textContent?.trim());

    expect(targetNodes).toHaveSize(3);
    expect(targetNodes[0].getAttribute('fill')).toBe(targetNodes[1].getAttribute('fill'));
    expect(targetNodes[0].getAttribute('fill')).not.toBe(targetNodes[2].getAttribute('fill'));
    expect(labels).toContain('T0 (A)');
    expect(labels).toContain('T2 (B)');
    expect(element.textContent).toContain('A');
    expect(element.textContent).toContain('B');
  });
});
