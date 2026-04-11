import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

type PaperSection = 'abstract' | 'introduction' | 'literature-review' | 'model-problem-definition' | 'methods' | 'framework-description';

interface PaperSectionDefinition {
  id: PaperSection;
  label: string;
  htmlFile: string;
}

const PAPER_SECTIONS: ReadonlyArray<PaperSectionDefinition> = [
  { id: 'abstract', label: 'Abstract', htmlFile: 'abstract.html' },
  { id: 'introduction', label: 'Introduction', htmlFile: 'introduction.html' },
  { id: 'literature-review', label: 'Literature Review', htmlFile: 'literature-review.html' },
  { id: 'model-problem-definition', label: 'Model & Problem Def.', htmlFile: 'model-problem-definition.html' },
  { id: 'methods', label: 'Methods', htmlFile: 'methods.html' },
  { id: 'framework-description', label: 'Framework Desc.', htmlFile: 'framework-description.html' }
];

@Component({
  selector: 'app-paper-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="paper-panel">
      <div class="paper-panel__tabs" role="tablist" aria-label="Paper sections">
        @for (section of sections; track section.id) {
          <button
            type="button"
            class="paper-panel__tab"
            role="tab"
            [class.paper-panel__tab--active]="activeSection() === section.id"
            [attr.aria-selected]="activeSection() === section.id"
            [attr.aria-controls]="'paper-panel-' + section.id"
            (click)="selectSection(section.id)"
          >
            {{ section.label }}
          </button>
        }
      </div>
      
      <div class="paper-panel__content">
        @for (section of sections; track section.id) {
          <div
            class="paper-panel__section"
            role="tabpanel"
            [id]="'paper-panel-' + section.id"
            [hidden]="activeSection() !== section.id"
          >
            <iframe
              class="paper-panel__iframe"
              [src]="getSafeUrl(section.htmlFile)"
              [title]="section.label + ' section'"
              frameborder="0"
            ></iframe>
          </div>
        }
      </div>
    </div>
  `,
  styleUrl: './paper-panel.scss'
})
export class PaperPanel {
  protected readonly sections = PAPER_SECTIONS;
  protected readonly activeSection = signal<PaperSection>('abstract');

  constructor(private sanitizer: DomSanitizer) {}

  protected selectSection(sectionId: PaperSection): void {
    this.activeSection.set(sectionId);
  }

  protected getSafeUrl(htmlFile: string): SafeResourceUrl {
    return this.sanitizer.bypassSecurityTrustResourceUrl(`/html/${htmlFile}`);
  }
}
