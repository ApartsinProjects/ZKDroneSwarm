import { CommonModule } from '@angular/common';
import { Component, computed, input, output, signal } from '@angular/core';
import { scaleSequential } from 'd3-scale';
import { interpolateBlues, interpolateRdBu } from 'd3-scale-chromatic';
import { LearningStateEpisodeDto } from '../../services/policies.service';

type MatrixView = 'observed' | 'predicted' | 'count';

interface MatrixViewOption {
  id: MatrixView;
  label: string;
}

const MATRIX_VIEW_OPTIONS: ReadonlyArray<MatrixViewOption> = [
  { id: 'observed', label: 'Observed' },
  { id: 'predicted', label: 'Predicted' },
  { id: 'count', label: 'Count' },
];

@Component({
  selector: 'app-integration-matrix-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="integration-matrix-panel">
      <div class="integration-matrix-panel__toolbar">
        <div class="integration-matrix-panel__agent-list" role="toolbar" aria-label="Integration matrix agents">
          @for (agentIndex of agentIndices(); track agentIndex) {
            <button
              type="button"
              class="integration-matrix-panel__agent-item"
              [class.integration-matrix-panel__agent-item--active]="agentIndex === selectedAgent()"
              (click)="selectedAgentChange.emit(agentIndex)"
            >
              <img
                class="integration-matrix-panel__agent-icon"
                src="assets/map/drone.png"
                alt="Drone {{ agentIndex }}"
              />
              <span class="integration-matrix-panel__agent-label">{{ agentIndex }}</span>
            </button>
          }
        </div>
      </div>

      <div class="integration-matrix-panel__compare">
        <section class="integration-matrix-panel__section">
          <div class="integration-matrix-panel__section-body">
            @if (topMatrixData(); as matrix) {
              <svg
                class="integration-matrix-panel__svg"
                viewBox="0 0 1080 360"
                role="img"
                [attr.aria-label]="'Top integration matrix heatmap: ' + matrixViewLabel(topEffectiveMatrixView())"
              >
                @for (row of matrix; track $index; let rowIndex = $index) {
                  @for (cell of row; track $index; let colIndex = $index) {
                    <rect
                      [attr.x]="colIndex * 40"
                      [attr.y]="rowIndex * 40"
                      [attr.width]="39"
                      [attr.height]="39"
                      [attr.fill]="getCellFill(cell, topEffectiveMatrixView())"
                      [attr.stroke]="topEffectiveMatrixView() === 'count' ? '#d9cfbf' : '#ffffff'"
                      stroke-width="1"
                    >
                      <title>{{ getCellTooltip(topEffectiveMatrixView(), rowIndex, colIndex, cell) }}</title>
                    </rect>
                    @if (topEffectiveMatrixView() === 'count') {
                      <text
                        [attr.x]="colIndex * 40 + 19.5"
                        [attr.y]="rowIndex * 40 + 23"
                        class="integration-matrix-panel__cell-text"
                      >
                        {{ formatCellValue(cell, topEffectiveMatrixView()) }}
                      </text>
                    }
                  }
                }
              </svg>
            } @else {
              <div class="analysis-placeholder">
                No matrix data is available for the top comparison slot.
              </div>
            }

            <label class="integration-matrix-panel__selector integration-matrix-panel__selector--side">
              <select
                [value]="topEffectiveMatrixView()"
                (change)="topMatrixView.set(asMatrixView($any($event.target).value))"
              >
                @for (option of matrixViewOptions; track option.id) {
                  <option
                    [value]="option.id"
                    [selected]="option.id === topEffectiveMatrixView()"
                    [disabled]="!isMatrixViewAvailable(option.id)"
                  >
                    {{ option.label }}
                  </option>
                }
              </select>
            </label>
          </div>
        </section>

        <section class="integration-matrix-panel__section">
          <div class="integration-matrix-panel__section-body">
            @if (bottomMatrixData(); as matrix) {
              <svg
                class="integration-matrix-panel__svg"
                viewBox="0 0 1080 360"
                role="img"
                [attr.aria-label]="'Bottom integration matrix heatmap: ' + matrixViewLabel(bottomEffectiveMatrixView())"
              >
                @for (row of matrix; track $index; let rowIndex = $index) {
                  @for (cell of row; track $index; let colIndex = $index) {
                    <rect
                      [attr.x]="colIndex * 40"
                      [attr.y]="rowIndex * 40"
                      [attr.width]="39"
                      [attr.height]="39"
                      [attr.fill]="getCellFill(cell, bottomEffectiveMatrixView())"
                      [attr.stroke]="bottomEffectiveMatrixView() === 'count' ? '#d9cfbf' : '#ffffff'"
                      stroke-width="1"
                    >
                      <title>{{ getCellTooltip(bottomEffectiveMatrixView(), rowIndex, colIndex, cell) }}</title>
                    </rect>
                    @if (bottomEffectiveMatrixView() === 'count') {
                      <text
                        [attr.x]="colIndex * 40 + 19.5"
                        [attr.y]="rowIndex * 40 + 23"
                        class="integration-matrix-panel__cell-text"
                      >
                        {{ formatCellValue(cell, bottomEffectiveMatrixView()) }}
                      </text>
                    }
                  }
                }
              </svg>
            } @else {
              <div class="analysis-placeholder">
                No matrix data is available for the bottom comparison slot.
              </div>
            }

            <label class="integration-matrix-panel__selector integration-matrix-panel__selector--side">
              <select
                [value]="bottomEffectiveMatrixView()"
                (change)="bottomMatrixView.set(asMatrixView($any($event.target).value))"
              >
                @for (option of matrixViewOptions; track option.id) {
                  <option
                    [value]="option.id"
                    [selected]="option.id === bottomEffectiveMatrixView()"
                    [disabled]="!isMatrixViewAvailable(option.id)"
                  >
                    {{ option.label }}
                  </option>
                }
              </select>
            </label>
          </div>
        </section>
      </div>
    </div>
  `,
  styles: [`
    .integration-matrix-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 16px;
    }

    .analysis-placeholder {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 200px;
      color: #7a6850;
      font-size: 14px;
    }

    .integration-matrix-panel__toolbar {
      display: flex;
      justify-content: center;
      padding-bottom: 8px;
    }

    .integration-matrix-panel__agent-list {
      display: flex;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .integration-matrix-panel__agent-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 6px 10px;
      border: 0.4px solid rgba(92, 77, 57, 0.15);
      border-radius: 10px;
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0) 100%);
      cursor: pointer;
      transition: all 0.2s ease;
      color: #5c4d39;
      font-size: 11px;
      font-weight: 600;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
      position: relative;
    }

    .integration-matrix-panel__agent-item:hover {
      background: linear-gradient(135deg, rgba(52, 152, 219, 0.12) 0%, rgba(52, 152, 219, 0.05) 100%);
      border-color: rgba(52, 152, 219, 0.4);
      box-shadow: 0 3px 8px rgba(52, 152, 219, 0.15);
      transform: translateY(-1px);
    }

    .integration-matrix-panel__agent-item--active {
      background: linear-gradient(135deg, rgba(52, 152, 219, 0.18) 0%, rgba(52, 152, 219, 0.08) 100%);
      border-color: rgba(52, 152, 219, 0.4);
      box-shadow: 0 4px 12px rgba(52, 152, 219, 0.25), inset 0 1px 2px rgba(255, 255, 255, 0.2), 0 0 0 1.6px rgba(52, 152, 219, 0.3);
      transform: translateY(-2px);
    }

    .integration-matrix-panel__agent-icon {
      width: 32px;
      height: 32px;
      object-fit: contain;
    }

    .integration-matrix-panel__agent-label {
      font-variant-numeric: tabular-nums;
    }

    .integration-matrix-panel__compare {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .integration-matrix-panel__section {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .integration-matrix-panel__section-body {
      display: flex;
      align-items: flex-start;
      gap: 12px;
    }

    .integration-matrix-panel__selector {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #5c4d39;
      font-size: 12px;
      font-weight: 600;
    }

    .integration-matrix-panel__selector--side {
      align-self: flex-start;
      min-width: 112px;
      padding: 10px;
      border: 1px solid rgba(92, 77, 57, 0.12);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.75);
      flex-shrink: 0;
    }

    .integration-matrix-panel__selector select {
      border: 1px solid rgba(92, 77, 57, 0.2);
      border-radius: 8px;
      background: #ffffff;
      color: #3a2f20;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
    }

    .integration-matrix-panel__svg {
      flex: 1 1 auto;
      width: 100%;
      height: auto;
      border: 1px solid rgba(92, 77, 57, 0.15);
      border-radius: 8px;
      background: #ffffff;
    }

    .integration-matrix-panel__svg rect {
      pointer-events: all;
      cursor: pointer;
    }

    .integration-matrix-panel__cell-text {
      fill: #3a2f20;
      font-size: 16px;
      font-weight: 600;
      text-anchor: middle;
      dominant-baseline: middle;
      pointer-events: none;
      font-variant-numeric: tabular-nums;
    }
  `],
})
export class IntegrationMatrixPanel {
  readonly snapshots = input<LearningStateEpisodeDto[]>([]);
  readonly currentSnapshot = input<LearningStateEpisodeDto | null>(null);
  readonly selectedAgent = input(0);
  readonly selectedAgentChange = output<number>();
  readonly matrixViewOptions = MATRIX_VIEW_OPTIONS;

  readonly agentIndices = computed(() => {
    const agents = this.currentSnapshot()?.learningState.episodeState?.agents ?? [];
    return Array.from({ length: agents.length }, (_, index) => index);
  });

  readonly topMatrixView = signal<MatrixView>('observed');
  readonly bottomMatrixView = signal<MatrixView>('predicted');

  readonly currentAgent = computed<Record<string, unknown> | null>(() => {
    const snapshot = this.currentSnapshot();
    if (!snapshot) {
      return null;
    }

    const agents = snapshot.learningState.episodeState?.agents ?? [];
    const selectedIdx = this.selectedAgent();
    
    if (selectedIdx < 0 || selectedIdx >= agents.length) {
      return null;
    }

    return agents[selectedIdx] ?? null;
  });

  readonly integrationMatrix = computed<Record<string, unknown> | null>(() => {
    const agent = this.currentAgent();
    if (!agent) {
      return null;
    }

    return agent['integration_matrix'] as Record<string, unknown> | null;
  });

  readonly topEffectiveMatrixView = computed<MatrixView>(() => {
    return this.resolveMatrixView(this.topMatrixView());
  });

  readonly bottomEffectiveMatrixView = computed<MatrixView>(() => {
    return this.resolveMatrixView(this.bottomMatrixView());
  });

  readonly topMatrixData = computed<number[][] | null>(() => {
    return this.getMatrixData(this.topEffectiveMatrixView());
  });

  readonly bottomMatrixData = computed<number[][] | null>(() => {
    return this.getMatrixData(this.bottomEffectiveMatrixView());
  });

  asMatrixView(value: string): MatrixView {
    if (value === 'predicted' || value === 'count') {
      return value;
    }
    return 'observed';
  }

  matrixViewLabel(view: MatrixView): string {
    return MATRIX_VIEW_OPTIONS.find((option) => option.id === view)?.label ?? 'Observed';
  }

  isMatrixViewAvailable(view: MatrixView): boolean {
    return this.getMatrixData(view) !== null;
  }

  private resolveMatrixView(requested: MatrixView): MatrixView {
    if (this.isMatrixDataAvailable(requested)) {
      return requested;
    }

    if (this.isMatrixDataAvailable('observed')) {
      return 'observed';
    }

    if (this.isMatrixDataAvailable('predicted')) {
      return 'predicted';
    }

    return 'count';
  }

  private isMatrixDataAvailable(view: MatrixView): boolean {
    return this.getMatrixData(view) !== null;
  }

  private getMatrixData(view: MatrixView): number[][] | null {
    const integrationMatrix = this.integrationMatrix();
    if (!integrationMatrix) {
      return null;
    }

    const key =
      view === 'predicted'
        ? 'M_pred'
        : view === 'count'
          ? 'M_count'
          : 'M_avg';
    const matrix = integrationMatrix[key];

    if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])) {
      return null;
    }

    return matrix as number[][];
  }

  formatCellValue(value: number, view: MatrixView): string {
    if (view === 'count') {
      return Math.round(value).toString();
    }
    return value.toFixed(2);
  }

  getCellFill(value: number, view: MatrixView): string {
    if (view === 'count') {
      return '#fffdf8';
    }

    const scale = scaleSequential(interpolateRdBu).domain([1.24, -0.51]);
    if (value === 0) {
      return '#f5f5f5';
    }
    return scale(value);
  }

  getCellTooltip(view: MatrixView, rowIndex: number, colIndex: number, value: number): string {
    return `${this.matrixViewLabel(view)} · Agent ${rowIndex} → Target ${colIndex}: ${value.toFixed(3)}`;
  }
}
