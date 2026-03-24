import { CommonModule } from '@angular/common';
import { Component, computed, input, output } from '@angular/core';
import { extent } from 'd3-array';
import { scaleLinear } from 'd3-scale';
import { LearningStateEpisodeDto } from '../../services/policies.service';

type PlotPoint = {
  x: number;
  y: number;
};

type PlotNode = PlotPoint & {
  id: string;
  label: string;
  className?: string;
  color: string;
  px: number;
  py: number;
};

type PlotTick = {
  value: string;
  x?: number;
  y?: number;
};



type PlotLegendItem = {
  className: string;
  color: string;
};

type PlotModel = {
  width: number;
  height: number;
  selectedAgent: number;
  agentCount: number;
  currentEpisodeNum: number | null;
  currentAgent: PlotNode;
  currentTargets: PlotNode[];
  legendItems: PlotLegendItem[];
  xTicks: PlotTick[];
  yTicks: PlotTick[];
  zeroX: number | null;
  zeroY: number | null;
};

const WIDTH = 560;
const HEIGHT = 360;
const MARGIN = { top: 20, right: 20, bottom: 40, left: 54 };
const AGENT_COLOR = '#3498db';
const TARGET_CLASS_COLORS = [
  '#e67e22',
  '#16a085',
  '#c0392b',
  '#8e44ad',
  '#2c3e50',
  '#f39c12',
  '#2980b9',
  '#27ae60',
];

@Component({
  selector: 'app-embedding-visualization-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (plotModel(); as model) {
      <div class="embedding-panel">
        <div class="embedding-panel__toolbar">
          <div class="embedding-panel__agent-list" role="toolbar" aria-label="Embedding agents">
            @for (agentIndex of agentIndices(); track agentIndex) {
              <button
                type="button"
                class="embedding-panel__agent-item"
                [class.embedding-panel__agent-item--active]="agentIndex === model.selectedAgent"
                (click)="selectedAgentChange.emit(agentIndex)"
              >
                <img
                  class="embedding-panel__agent-icon"
                  src="assets/map/drone.png"
                  alt="Drone {{ agentIndex }}"
                />
                <span class="embedding-panel__agent-label">{{ agentIndex }}</span>
              </button>
            }
          </div>

          @if (model.legendItems.length > 0) {
            <div class="embedding-panel__legend" aria-label="Target class legend">
              @for (item of model.legendItems; track item.className) {
                <span class="embedding-panel__legend-item">
                  <span
                    class="embedding-panel__legend-swatch"
                    [style.background]="item.color"
                  ></span>
                  {{ item.className }}
                </span>
              }
            </div>
          }
        </div>

        <div class="embedding-panel__stage">
          <svg
            class="embedding-panel__svg"
            [attr.viewBox]="'0 0 ' + model.width + ' ' + model.height"
            role="img"
            aria-label="Embedding trajectory chart"
          >
            <g>
              @for (tick of model.xTicks; track tick.value) {
                <line
                  class="embedding-grid__line"
                  [attr.x1]="tick.x"
                  [attr.x2]="tick.x"
                  [attr.y1]="MARGIN.top"
                  [attr.y2]="model.height - MARGIN.bottom"
                  stroke="rgba(126, 104, 62, 0.12)"
                  stroke-dasharray="4 6"
                />
                <text
                  [attr.x]="tick.x"
                  [attr.y]="model.height - 12"
                  text-anchor="middle"
                  font-size="10"
                  fill="#7a6850"
                >
                  {{ tick.value }}
                </text>
              }

              @for (tick of model.yTicks; track tick.value) {
                <line
                  class="embedding-grid__line"
                  [attr.x1]="MARGIN.left"
                  [attr.x2]="model.width - MARGIN.right"
                  [attr.y1]="tick.y"
                  [attr.y2]="tick.y"
                  stroke="rgba(126, 104, 62, 0.12)"
                  stroke-dasharray="4 6"
                />
                <text
                  [attr.x]="MARGIN.left - 10"
                  [attr.y]="(tick.y ?? 0) + 4"
                  text-anchor="end"
                  font-size="10"
                  fill="#7a6850"
                >
                  {{ tick.value }}
                </text>
              }

              @if (model.zeroX !== null) {
                <line
                  [attr.x1]="model.zeroX"
                  [attr.x2]="model.zeroX"
                  [attr.y1]="MARGIN.top"
                  [attr.y2]="model.height - MARGIN.bottom"
                  stroke="rgba(109, 90, 61, 0.2)"
                />
              }

              @if (model.zeroY !== null) {
                <line
                  [attr.x1]="MARGIN.left"
                  [attr.x2]="model.width - MARGIN.right"
                  [attr.y1]="model.zeroY"
                  [attr.y2]="model.zeroY"
                  stroke="rgba(109, 90, 61, 0.2)"
                />
              }
            </g>



            @for (target of model.currentTargets; track target.id) {
              <circle
                class="embedding-node embedding-node--target-bg"
                [attr.cx]="target.px"
                [attr.cy]="target.py"
                r="8"
                [attr.fill]="target.color"
                opacity="0.4"
              />
              <image
                class="embedding-node embedding-node--target"
                [attr.x]="target.px - 9"
                [attr.y]="target.py - 13"
                width="15"
                height="18"
                xlink:href="assets/map/target_1.png"
              />
            }


            <circle
              class="embedding-node--agent-shadow"
              [attr.cx]="model.currentAgent.px"
              [attr.cy]="model.currentAgent.py + 12"
              r="8"
            />
            <image
              class="embedding-node embedding-node--agent"
              [attr.x]="model.currentAgent.px - 16"
              [attr.y]="model.currentAgent.py - 18"
              width="32"
              height="32"
              xlink:href="assets/map/drone.png"
            />
          </svg>
        </div>


      </div>
    } @else {
      <div class="analysis-placeholder">
        No embedding snapshot is available for the selected episode.
      </div>
    }
  `,
  styles: [`
    .embedding-node {
      transition: cx 0.08s linear, cy 0.08s linear, x 0.08s linear, y 0.08s linear, transform 0.08s linear;
    }

    .embedding-node--agent {
      filter: drop-shadow(0 4px 5px rgba(0, 0, 0, 0.25));
    }

    .embedding-node--agent-shadow {
      fill: rgba(0, 0, 0, 0.1);
      filter: blur(2px);
      transition: cx 0.08s linear, cy 0.08s linear;
    }

    .embedding-panel__agent-list {
      display: flex;
      justify-content: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .embedding-panel__agent-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 6px 10px;
      border: 1px solid rgba(92, 77, 57, 0.15);
      border-radius: 8px;
      background: transparent;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s;
      color: #5c4d39;
      font-size: 11px;
      font-weight: 600;
    }

    .embedding-panel__agent-item:hover {
      background: rgba(52, 152, 219, 0.08);
      border-color: rgba(52, 152, 219, 0.3);
    }

    .embedding-panel__agent-item--active {
      background: rgba(52, 152, 219, 0.14);
      border-color: #3498db;
    }

    .embedding-panel__agent-icon {
      width: 32px;
      height: 32px;
      object-fit: contain;
    }

    .embedding-panel__agent-label {
      font-variant-numeric: tabular-nums;
    }

    .embedding-panel__legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-top: 8px;
      font-size: 12px;
      color: #5c4d39;
    }

    .embedding-panel__legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .embedding-panel__legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(92, 77, 57, 0.18);
    }
  `],
})
export class EmbeddingVisualizationPanel {
  protected readonly MARGIN = MARGIN;

  readonly snapshots = input<LearningStateEpisodeDto[]>([]);
  readonly currentSnapshot = input<LearningStateEpisodeDto | null>(null);
  readonly selectedAgent = input(0);
  readonly selectedAgentChange = output<number>();

  readonly agentIndices = computed(() => {
    const agents = this.currentSnapshot()?.learningState.episodeState?.agents ?? [];
    return Array.from({ length: agents.length }, (_, index) => index);
  });

  readonly plotModel = computed<PlotModel | null>(() => {
    const currentSnapshot = this.currentSnapshot();
    if (!currentSnapshot) {
      return null;
    }

    const currentState = currentSnapshot?.learningState.episodeState;
    const currentAgents = currentState?.agents ?? [];
    if (currentAgents.length === 0) {
      return null;
    }

    const selectedAgentIndex = Math.min(this.selectedAgent(), currentAgents.length - 1);
    const currentAgentState = currentAgents[selectedAgentIndex];
    const currentAgentPoint = this.readPoint(currentAgentState?.['agent_lv']);
    const currentTargetPoints = this.readPoints(currentAgentState?.['target_lv']);
    if (!currentAgentPoint || currentTargetPoints.length === 0) {
      return null;
    }
    const currentTargetClasses = this.readTargetClasses(
      currentState?.['target_classes'],
      currentTargetPoints.length,
    );
    const classColorMap = this.buildClassColorMap(this.snapshots());

    const domainPoints = [currentAgentPoint, ...currentTargetPoints];
    const xDomain = this.computeDomain(domainPoints.map((point) => point.x));
    const yDomain = this.computeDomain(domainPoints.map((point) => point.y));

    const xScale = scaleLinear().domain(xDomain).range([MARGIN.left, WIDTH - MARGIN.right]);
    const yScale = scaleLinear().domain(yDomain).range([HEIGHT - MARGIN.bottom, MARGIN.top]);

    const currentAgent = this.toNode(
      currentAgentPoint,
      xScale,
      yScale,
      `${selectedAgentIndex}`,
      `A${selectedAgentIndex}`,
      AGENT_COLOR,
    );

    const currentTargets = currentTargetPoints.map((point, index) =>
      this.toNode(
        point,
        xScale,
        yScale,
        `target-${index}`,
        `T${index} (${currentTargetClasses[index]})`,
        this.colorForClass(currentTargetClasses[index], classColorMap),
        currentTargetClasses[index],
      ),
    );

    const currentEpisodeNum = currentSnapshot.episode?.episodeNum ?? Infinity;


    return {
      width: WIDTH,
      height: HEIGHT,
      selectedAgent: selectedAgentIndex,
      agentCount: currentAgents.length,
      currentEpisodeNum: currentSnapshot?.episode.episodeNum ?? null,
      currentAgent,
      currentTargets,
      legendItems: Array.from(classColorMap.entries()).map(([className, color]) => ({
        className,
        color,
      })),
      xTicks: xScale.ticks(5).map((value: number) => ({ value: value.toFixed(2), x: xScale(value) })),
      yTicks: yScale.ticks(5).map((value: number) => ({ value: value.toFixed(2), y: yScale(value) })),
      zeroX: xDomain[0] <= 0 && xDomain[1] >= 0 ? xScale(0) : null,
      zeroY: yDomain[0] <= 0 && yDomain[1] >= 0 ? yScale(0) : null,
    };
  });

  private readPoint(value: unknown): PlotPoint | null {
    if (!Array.isArray(value) || value.length < 2) {
      return null;
    }

    const x = Number(value[0]);
    const y = Number(value[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return null;
    }

    return { x, y };
  }

  private readPoints(value: unknown): PlotPoint[] {
    if (!Array.isArray(value)) {
      return [];
    }

    return value
      .map((point) => this.readPoint(point))
      .filter((point): point is PlotPoint => point !== null);
  }

  private readTargetClasses(value: unknown, expectedLength: number): string[] {
    if (!Array.isArray(value)) {
      return Array.from({ length: expectedLength }, () => 'unknown');
    }

    return Array.from({ length: expectedLength }, (_unused, index) => {
      const className = value[index];
      return typeof className === 'string' && className.length > 0 ? className : 'unknown';
    });
  }

  private buildClassColorMap(snapshots: LearningStateEpisodeDto[]): Map<string, string> {
    const classNames = snapshots
      .flatMap((snapshot) => {
        const targetClasses = snapshot.learningState.episodeState?.target_classes;
        return Array.isArray(targetClasses) ? targetClasses : [];
      })
      .filter((className): className is string => typeof className === 'string' && className.length > 0);

    const uniqueClassNames = Array.from(new Set(classNames)).sort();
    if (uniqueClassNames.length === 0) {
      uniqueClassNames.push('unknown');
    }

    return new Map(
      uniqueClassNames.map((className, index) => [
        className,
        TARGET_CLASS_COLORS[index % TARGET_CLASS_COLORS.length],
      ]),
    );
  }

  private colorForClass(className: string, classColorMap: Map<string, string>): string {
    return classColorMap.get(className) ?? TARGET_CLASS_COLORS[0];
  }

  private computeDomain(values: number[]): [number, number] {
    const [minValue = -1, maxValue = 1] = extent(values);
    const spread = maxValue - minValue;
    const padding = spread === 0 ? 0.5 : spread * 0.18;
    return [minValue - padding, maxValue + padding];
  }

  private toNode(
    point: PlotPoint,
    xScale: ReturnType<typeof scaleLinear>,
    yScale: ReturnType<typeof scaleLinear>,
    id: string,
    label: string,
    color: string,
    className?: string,
  ): PlotNode {
    return {
      ...point,
      id,
      label,
      className,
      color,
      px: xScale(point.x),
      py: yScale(point.y),
    };
  }
}
