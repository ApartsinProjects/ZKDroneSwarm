import { CommonModule } from '@angular/common';
import { Component, computed, input } from '@angular/core';
import { extent } from 'd3-array';
import { scaleLinear } from 'd3-scale';
import { LatentVectorsData } from '../../services/latent-world.service';

type PlotPoint = {
  x: number;
  y: number;
};

type PlotNode = PlotPoint & {
  id: string;
  label: string;
  modeId: number;
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
  modeId: number;
  label: string;
  color: string;
};

type PlotModel = {
  width: number;
  height: number;
  drones: PlotNode[];
  targets: PlotNode[];
  legendItems: PlotLegendItem[];
  xTicks: PlotTick[];
  yTicks: PlotTick[];
  zeroX: number | null;
  zeroY: number | null;
};

const WIDTH = 560;
const HEIGHT = 360;
const MARGIN = { top: 20, right: 20, bottom: 40, left: 54 };
const DRONE_COLOR = '#3498db';
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
  selector: 'app-latent-world-visualization-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (plotModel(); as model) {
      <div class="latent-panel">
        <div class="latent-panel__header">
          <h3 class="latent-panel__title">Latent World Structure</h3>
          <p class="latent-panel__subtitle">Ground truth latent vectors (t-SNE projection)</p>
        </div>

        @if (model.legendItems.length > 0) {
          <div class="latent-panel__legend" aria-label="Mode legend">
            @for (item of model.legendItems; track item.modeId) {
              <span class="latent-panel__legend-item">
                <span
                  class="latent-panel__legend-swatch"
                  [style.background]="item.color"
                ></span>
                {{ item.label }}
              </span>
            }
          </div>
        }

        <div class="latent-panel__stage">
          <svg
            class="latent-panel__svg"
            [attr.viewBox]="'0 0 ' + model.width + ' ' + model.height"
            role="img"
            aria-label="Latent world visualization"
          >
            <g>
              @for (tick of model.xTicks; track tick.value) {
                <line
                  class="latent-grid__line"
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
                  class="latent-grid__line"
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

            @for (target of model.targets; track target.id) {
              <circle
                class="latent-node latent-node--target-bg"
                [attr.cx]="target.px"
                [attr.cy]="target.py"
                r="8"
                [attr.fill]="target.color"
                opacity="0.4"
              />
              <image
                class="latent-node latent-node--target"
                [attr.x]="target.px - 9"
                [attr.y]="target.py - 13"
                width="15"
                height="18"
                xlink:href="assets/map/target_1.png"
              />
            }

            @for (drone of model.drones; track drone.id) {
              <circle
                class="latent-node--drone-shadow"
                [attr.cx]="drone.px"
                [attr.cy]="drone.py + 12"
                r="8"
              />
              <image
                class="latent-node latent-node--drone"
                [attr.x]="drone.px - 16"
                [attr.y]="drone.py - 18"
                width="32"
                height="32"
                xlink:href="assets/map/drone.png"
              />
            }
          </svg>
        </div>
      </div>
    } @else {
      <div class="analysis-placeholder">
        No latent world data available. This visualization is only available for latent world scenarios.
      </div>
    }
  `,
  styleUrl: './latent-world-visualization-panel.scss',
})
export class LatentWorldVisualizationPanel {
  protected readonly MARGIN = MARGIN;

  readonly latentVectors = input<LatentVectorsData | null>(null);

  readonly plotModel = computed<PlotModel | null>(() => {
    const data = this.latentVectors();
    if (!data || !data.drones.length || !data.targets.length) {
      return null;
    }

    const dronePoints: PlotPoint[] = data.drones.map(d => ({
      x: d.tsne_coords[0],
      y: d.tsne_coords[1]
    }));

    const targetPoints: PlotPoint[] = data.targets.map(t => ({
      x: t.tsne_coords[0],
      y: t.tsne_coords[1]
    }));

    const allPoints = [...dronePoints, ...targetPoints];
    const xDomain = this.computeDomain(allPoints.map(p => p.x));
    const yDomain = this.computeDomain(allPoints.map(p => p.y));

    const xScale = scaleLinear().domain(xDomain).range([MARGIN.left, WIDTH - MARGIN.right]);
    const yScale = scaleLinear().domain(yDomain).range([HEIGHT - MARGIN.bottom, MARGIN.top]);

    const uniqueModes = new Set([
      ...data.drones.map(d => d.mode_id),
      ...data.targets.map(t => t.mode_id)
    ]);

    const modeColorMap = new Map(
      Array.from(uniqueModes).sort().map((modeId, index) => [
        modeId,
        MODE_COLORS[index % MODE_COLORS.length]
      ])
    );

    const drones: PlotNode[] = data.drones.map((drone, index) => ({
      x: dronePoints[index].x,
      y: dronePoints[index].y,
      id: drone.id,
      label: `${drone.id} (mode ${drone.mode_id})`,
      modeId: drone.mode_id,
      color: modeColorMap.get(drone.mode_id) ?? DRONE_COLOR,
      px: xScale(dronePoints[index].x),
      py: yScale(dronePoints[index].y)
    }));

    const targets: PlotNode[] = data.targets.map((target, index) => ({
      x: targetPoints[index].x,
      y: targetPoints[index].y,
      id: target.id,
      label: `${target.id} (mode ${target.mode_id})`,
      modeId: target.mode_id,
      color: modeColorMap.get(target.mode_id) ?? MODE_COLORS[0],
      px: xScale(targetPoints[index].x),
      py: yScale(targetPoints[index].y)
    }));

    return {
      width: WIDTH,
      height: HEIGHT,
      drones,
      targets,
      legendItems: Array.from(modeColorMap.entries()).map(([modeId, color]) => ({
        modeId,
        label: `Mode ${modeId}`,
        color
      })),
      xTicks: xScale.ticks(5).map((value: number) => ({ value: value.toFixed(2), x: xScale(value) })),
      yTicks: yScale.ticks(5).map((value: number) => ({ value: value.toFixed(2), y: yScale(value) })),
      zeroX: xDomain[0] <= 0 && xDomain[1] >= 0 ? xScale(0) : null,
      zeroY: yDomain[0] <= 0 && yDomain[1] >= 0 ? yScale(0) : null,
    };
  });

  private computeDomain(values: number[]): [number, number] {
    const [minValue = -1, maxValue = 1] = extent(values);
    const spread = maxValue - minValue;
    const padding = spread === 0 ? 0.5 : spread * 0.18;
    return [minValue - padding, maxValue + padding];
  }
}
