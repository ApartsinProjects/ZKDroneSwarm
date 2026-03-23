import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { PoliciesService } from '../../services/policies.service';

@Component({
  selector: 'app-policies',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './policies.component.html',
  styleUrl: './policies.component.scss'
})
export class PoliciesComponent {
  private policiesService = inject(PoliciesService);
  
  public policies = toSignal(this.policiesService.getPolicies());

  onPolicyClick(policyName: string): void {
    console.log(`Selected policy: ${policyName}`);
    this.policiesService.getBestEpisode(policyName).subscribe({
      next: (episode) => {
        console.log('Best episode retrieved:', episode);
      },
      error: (err) => {
        console.error('Error fetching best episode:', err);
      }
    });
  }
}
