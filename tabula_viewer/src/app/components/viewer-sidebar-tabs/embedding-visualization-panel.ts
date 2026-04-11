import { CommonModule } from '@angular/common';
import { Component, computed, input, output, inject, signal } from '@angular/core';
import { extent } from 'd3-array';
import { scaleLinear } from 'd3-scale';
import { LearningStateEpisodeDto } from '../../services/policies.service';
import { LatentWorldService } from '../../services/latent-world.service';
import { Embedding3DRenderer } from './embedding-3d-renderer';

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

export type PlotPoint3D = {
  x: number;
  y: number;
  z: number;
};

export type PlotNode3D = PlotPoint3D & {
  id: string;
  label: string;
  className?: string;
  color: string;
};

export type PlotData3D = {
  agent: PlotNode3D;
  targets: PlotNode3D[];
  bounds: {
    min: PlotPoint3D;
    max: PlotPoint3D;
  };
};

const WIDTH = 560;
const HEIGHT = 360;
const MARGIN = { top: 20, right: 20, bottom: 40, left: 54 };
const AGENT_COLOR = '#3498db';
const TARGET_CLASS_COLORS = [
  '#ff3b30', // Vibrant Red
  '#007aff', // Vibrant Blue
  '#4cd964', // Vibrant Green
  '#ff9500', // Vibrant Orange
  '#5856d6', // Vibrant Indigo
  '#ff2d55', // Vibrant Pink
  '#5ac8fa', // Teal/Cyan
  '#af52de', // Vibrant Purple
];

const MODE_COLORS = [
  '#ff3b30',
  '#007aff',
  '#4cd964',
  '#ff9500',
  '#5856d6',
  '#ff2d55',
  '#5ac8fa',
  '#af52de',
];

@Component({
  selector: 'app-embedding-visualization-panel',
  standalone: true,
  imports: [CommonModule, Embedding3DRenderer],
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
                [style.border-color]="getDroneBorderColor(agentIndex)"
                [style.background]="getDroneBackground(agentIndex, agentIndex === model.selectedAgent)"
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

          <label class="embedding-panel__mode-toggle">
            <input 
              type="checkbox" 
              [checked]="is3DMode()" 
              (change)="is3DMode.set($any($event.target).checked)"
            />
            <span>3D View</span>
          </label>
        </div>

        <div class="embedding-panel__stage">
          @if (!is3DMode()) {
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
          } @else {
            @if (plotModel3D(); as data3D) {
              <app-embedding-3d-renderer [plotData]="data3D" />
            } @else {
              <div class="analysis-placeholder">
                No 3D embedding data available (requires at least 3 dimensions).
              </div>
            }
          }
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

    .embedding-panel__agent-item:hover {
      background: linear-gradient(135deg, rgba(52, 152, 219, 0.12) 0%, rgba(52, 152, 219, 0.05) 100%);
      border-color: rgba(52, 152, 219, 0.4);
      box-shadow: 0 3px 8px rgba(52, 152, 219, 0.15);
      transform: translateY(-1px);
    }

    .embedding-panel__agent-item--active {
      background: linear-gradient(135deg, rgba(52, 152, 219, 0.18) 0%, rgba(52, 152, 219, 0.08) 100%);
      border-color: rgba(52, 152, 219, 0.4);
      box-shadow: 0 4px 12px rgba(52, 152, 219, 0.25), inset 0 1px 2px rgba(255, 255, 255, 0.2), 0 0 0 1.6px rgba(52, 152, 219, 0.3);
      transform: translateY(-2px);
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

    .embedding-panel__mode-toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-left: auto;
      font-size: 12px;
      font-weight: 500;
      color: #5c4d39;
      cursor: pointer;
      user-select: none;
    }

    .embedding-panel__mode-toggle input[type="checkbox"] {
      width: 16px;
      height: 16px;
      cursor: pointer;
      accent-color: #3498db;
    }

    .embedding-panel__mode-toggle span {
      white-space: nowrap;
    }
  `],
})
export class EmbeddingVisualizationPanel {
  protected readonly MARGIN = MARGIN;
  private latentWorldService = inject(LatentWorldService);

  readonly snapshots = input<LearningStateEpisodeDto[]>([]);
  readonly currentSnapshot = input<LearningStateEpisodeDto | null>(null);
  readonly selectedAgent = input(0);
  readonly selectedAgentChange = output<number>();
  
  private latentVectorsSignal = signal<any>(null);
  readonly latentVectors = this.latentVectorsSignal.asReadonly();

  readonly is3DMode = signal(false);

  constructor() {
    this.latentWorldService.getLatentVectors().subscribe({
      next: (data) => this.latentVectorsSignal.set(data),
      error: (err) => {
        console.error('Failed to load latent vectors for embedding panel:', err);
        this.latentVectorsSignal.set(null);
      }
    });
  }

  readonly agentIndices = computed(() => {
    const agents = this.currentSnapshot()?.learningState.episodeState?.agents ?? [];
    return Array.from({ length: agents.length }, (_, index) => index);
  });

  protected getDroneBorderColor(agentIndex: number): string {
    const latentData = this.latentVectors();
    if (!latentData?.drones) {
      return 'rgba(92, 77, 57, 0.15)';
    }
    
    const drone = latentData.drones.find((d: any) => {
      const droneId = d.id;
      const match = droneId.match(/drone[_-]?(\d+)/);
      return match && parseInt(match[1]) === agentIndex;
    });
    
    if (!drone || drone.mode_id === undefined) {
      return 'rgba(92, 77, 57, 0.15)';
    }
    
    return MODE_COLORS[drone.mode_id % MODE_COLORS.length];
  }

  protected getDroneBackground(agentIndex: number, isActive: boolean): string {
    if (!isActive) {
      return 'linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0) 100%)';
    }

    const modeColor = this.getDroneBorderColor(agentIndex);
    if (modeColor === 'rgba(92, 77, 57, 0.15)') {
      return 'linear-gradient(135deg, rgba(52, 152, 219, 0.18) 0%, rgba(52, 152, 219, 0.08) 100%)';
    }

    // Convert hex to RGB and create light gradient
    const hex = modeColor.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    
    return `linear-gradient(135deg, rgba(${r}, ${g}, ${b}, 0.18) 0%, rgba(${r}, ${g}, ${b}, 0.08) 100%)`;
  }

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
    
    // Extract agent embedding: prefer enriched 2D, fallback to P matrix
    let agentEmbSource = currentAgentState?.['agent_emb_2d'];
    if (!agentEmbSource) {
      const P = currentAgentState?.['P'];
      const agentIdx = typeof currentAgentState?.['agent_idx'] === 'number' 
        ? currentAgentState['agent_idx'] 
        : selectedAgentIndex;
      agentEmbSource = (Array.isArray(P) && Array.isArray(P[agentIdx])) 
        ? P[agentIdx].slice(0, 2) 
        : undefined;
    }
    const currentAgentPoint = this.readPoint(agentEmbSource);
    
    // Extract target embeddings: prefer enriched 2D, fallback to U matrix
    let targetEmbSource = currentAgentState?.['target_emb_2d'];
    if (!targetEmbSource) {
      const U = currentAgentState?.['U'];
      targetEmbSource = (Array.isArray(U) && U.length > 0 && Array.isArray(U[0]))
        ? U[0].map((_, colIdx) => U.map(row => row[colIdx]).slice(0, 2)) // Transpose and take first 2 dims
        : undefined;
    }
    const currentTargetPoints = this.readPoints(targetEmbSource);
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

  readonly plotModel3D = computed<PlotData3D | null>(() => {
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
    
    // Extract agent embedding from P matrix
    const P = currentAgentState?.['P'];
    const agentIdx = typeof currentAgentState?.['agent_idx'] === 'number' 
      ? currentAgentState['agent_idx'] 
      : selectedAgentIndex;
    const agentEmbSource = (Array.isArray(P) && Array.isArray(P[agentIdx])) 
      ? P[agentIdx] 
      : undefined;
    const agentPoint = this.readPoint3D(agentEmbSource);
    
    // Extract target embeddings from U matrix
    const U = currentAgentState?.['U'];
    const targetEmbSource = (Array.isArray(U) && U.length > 0 && Array.isArray(U[0]))
      ? U[0].map((_, colIdx) => U.map(row => row[colIdx])) // Transpose U to get targets
      : undefined;
    const targetPoints = this.readPoints3D(targetEmbSource);
    
    if (!agentPoint || targetPoints.length === 0) {
      return null;
    }

    const currentTargetClasses = this.readTargetClasses(
      currentState?.['target_classes'],
      targetPoints.length,
    );
    const classColorMap = this.buildClassColorMap(this.snapshots());

    const agent: PlotNode3D = {
      ...agentPoint,
      id: `agent-${selectedAgentIndex}`,
      label: `A${selectedAgentIndex}`,
      color: AGENT_COLOR,
    };

    const targets: PlotNode3D[] = targetPoints.map((point, index) => ({
      ...point,
      id: `target-${index}`,
      label: `T${index} (${currentTargetClasses[index]})`,
      color: this.colorForClass(currentTargetClasses[index], classColorMap),
      className: currentTargetClasses[index],
    }));

    const allPoints = [agentPoint, ...targetPoints];
    const bounds = this.computeBounds3D(allPoints);

    return { agent, targets, bounds };
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

  private readPoint3D(value: unknown): PlotPoint3D | null {
    if (!Array.isArray(value) || value.length < 3) {
      return null;
    }

    const x = Number(value[0]);
    const y = Number(value[1]);
    const z = Number(value[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      return null;
    }

    return { x, y, z };
  }

  private readPoints3D(value: unknown): PlotPoint3D[] {
    if (!Array.isArray(value)) {
      return [];
    }

    return value
      .map((point) => this.readPoint3D(point))
      .filter((point): point is PlotPoint3D => point !== null);
  }

  private computeBounds3D(points: PlotPoint3D[]): { min: PlotPoint3D; max: PlotPoint3D } {
    if (points.length === 0) {
      return {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };
    }

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const zs = points.map((p) => p.z);

    const [minX = -1, maxX = 1] = extent(xs);
    const [minY = -1, maxY = 1] = extent(ys);
    const [minZ = -1, maxZ = 1] = extent(zs);

    return {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
    };
  }
}
