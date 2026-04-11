import { Component, input } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface DynamicEnvMetrics {
  currentStep: number;
  maxHp: number;
  currentHp: number;
  avgHpReductionPerStep: number;
}

@Component({
  selector: 'app-environment-info',
  imports: [CommonModule],
  templateUrl: './environment-info.html',
  styleUrl: './environment-info.scss',
})
export class EnvironmentInfo {
  metrics = input<DynamicEnvMetrics | null>(null);
}
