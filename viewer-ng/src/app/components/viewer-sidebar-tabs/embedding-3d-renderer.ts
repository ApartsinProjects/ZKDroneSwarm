import { Component, ElementRef, ViewChild, input, AfterViewInit, OnDestroy, NgZone, inject, effect } from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PlotData3D } from './embedding-visualization-panel';

@Component({
  selector: 'app-embedding-3d-renderer',
  standalone: true,
  template: '<canvas #canvas></canvas>',
  styles: [`
    :host {
      display: block;
      width: 100%;
      height: 100%;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
  `],
})
export class Embedding3DRenderer implements AfterViewInit, OnDestroy {
  private ngZone = inject(NgZone);

  readonly plotData = input.required<PlotData3D>();

  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;

  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private animationFrameId: number | null = null;
  private sceneInitialized = false;
  private isFirstBuild = true;
  private textureLoader = new THREE.TextureLoader();

  constructor() {
    effect(() => {
      const data = this.plotData();
      if (this.sceneInitialized && data) {
        this.clearScene();
        this.buildScene();
      }
    });
  }

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf5f5dc);

    this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    this.camera.position.set(2, 2, 2);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight.position.set(5, 5, 5);
    this.scene.add(directionalLight);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    this.buildScene();
    this.sceneInitialized = true;
    this.animate();
  }

  private clearScene(): void {
    const objectsToRemove: THREE.Object3D[] = [];
    this.scene.traverse((object: THREE.Object3D) => {
      if (object instanceof THREE.Mesh) {
        objectsToRemove.push(object);
      } else if (object instanceof THREE.Sprite) {
        objectsToRemove.push(object);
      } else if (object instanceof THREE.GridHelper) {
        objectsToRemove.push(object);
      } else if (object instanceof THREE.Line) {
        objectsToRemove.push(object);
      }
    });

    objectsToRemove.forEach((object) => {
      this.scene.remove(object);
      if (object instanceof THREE.Mesh) {
        object.geometry.dispose();
        if (object.material instanceof THREE.Material) {
          object.material.dispose();
        }
      } else if (object instanceof THREE.Sprite) {
        if (object.material instanceof THREE.Material) {
          object.material.dispose();
        }
      } else if (object instanceof THREE.Line) {
        if (object instanceof THREE.Line) {
          (object as THREE.Line).geometry.dispose();
          if ((object as THREE.Line).material instanceof THREE.Material) {
            ((object as THREE.Line).material as THREE.Material).dispose();
          }
        }
      }
    });
  }

  private buildScene(): void {
    const data = this.plotData();

    const gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0xcccccc);
    this.scene.add(gridHelper);

    const bounds = data.bounds;
    const minZ = bounds.min.z;
    const maxZ = bounds.max.z;
    const rangeZ = maxZ - minZ;
    const minX = bounds.min.x;
    const maxX = bounds.max.x;
    const rangeX = maxX - minX;
    const minY = bounds.min.y;
    const maxY = bounds.max.y;
    const rangeY = maxY - minY;
    
    const numHorizontalGrids = 5;
    for (let i = 1; i <= numHorizontalGrids; i++) {
      const height = minZ + (rangeZ * i / (numHorizontalGrids + 1));
      const horizontalGrid = new THREE.GridHelper(10, 20, 0xaaaaaa, 0xdddddd);
      horizontalGrid.rotation.x = Math.PI / 2;
      horizontalGrid.position.y = height;
      horizontalGrid.material.opacity = 0.3;
      horizontalGrid.material.transparent = true;
      this.scene.add(horizontalGrid);
    }

    const numVerticalGridsX = 5;
    for (let i = 1; i <= numVerticalGridsX; i++) {
      const xPos = minX + (rangeX * i / (numVerticalGridsX + 1));
      const verticalGridX = new THREE.GridHelper(10, 20, 0xaaaaaa, 0xdddddd);
      verticalGridX.rotation.z = Math.PI / 2;
      verticalGridX.position.x = xPos;
      verticalGridX.material.opacity = 0.3;
      verticalGridX.material.transparent = true;
      this.scene.add(verticalGridX);
    }

    const numVerticalGridsY = 5;
    for (let i = 1; i <= numVerticalGridsY; i++) {
      const yPos = minY + (rangeY * i / (numVerticalGridsY + 1));
      const verticalGridY = new THREE.GridHelper(10, 20, 0xaaaaaa, 0xdddddd);
      verticalGridY.position.z = -yPos;
      verticalGridY.material.opacity = 0.3;
      verticalGridY.material.transparent = true;
      this.scene.add(verticalGridY);
    }

    const droneTexture = this.textureLoader.load('assets/map/drone.png');
    const agentMaterial = new THREE.SpriteMaterial({ map: droneTexture, sizeAttenuation: false });
    const agentSprite = new THREE.Sprite(agentMaterial);
    agentSprite.scale.set(0.18, 0.18, 1);
    agentSprite.position.set(data.agent.x, data.agent.z, -data.agent.y);
    this.scene.add(agentSprite);

    const targetTexture = this.textureLoader.load('assets/map/target_1.png');
    data.targets.forEach((target) => {
      const bgGeometry = new THREE.CircleGeometry(0.05, 32);
      const bgMaterial = new THREE.MeshBasicMaterial({ 
        color: target.color,
        transparent: true,
        opacity: 0.6,
        side: THREE.DoubleSide
      });
      const bgCircle = new THREE.Mesh(bgGeometry, bgMaterial);
      bgCircle.rotation.x = -Math.PI / 2;
      bgCircle.position.set(target.x, target.z - 0.04, -target.y);
      this.scene.add(bgCircle);

      const targetMaterial = new THREE.SpriteMaterial({ 
        map: targetTexture, 
        sizeAttenuation: false 
      });
      const targetSprite = new THREE.Sprite(targetMaterial);
      targetSprite.scale.set(0.12, 0.12, 1);
      targetSprite.position.set(target.x, target.z, -target.y);
      this.scene.add(targetSprite);
    });

    const centerX = (bounds.min.x + bounds.max.x) / 2;
    const centerY = (bounds.min.y + bounds.max.y) / 2;
    const centerZ = (bounds.min.z + bounds.max.z) / 2;

    if (this.isFirstBuild) {
      const rangeX = bounds.max.x - bounds.min.x;
      const rangeY = bounds.max.y - bounds.min.y;
      const maxRange = Math.max(rangeX, rangeY, rangeZ, 0.5);
      const distance = Math.max(maxRange * 4, 2);

      this.camera.position.set(centerX + distance, centerZ + distance, -(centerY + distance));
      this.camera.lookAt(centerX, centerZ, -centerY);
      this.isFirstBuild = false;
    }

    this.controls.target.set(centerX, centerZ, -centerY);
    this.controls.update();
  }

  private animate(): void {
    this.ngZone.runOutsideAngular(() => {
      const loop = () => {
        this.animationFrameId = requestAnimationFrame(loop);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      };
      loop();
    });
  }

  ngOnDestroy(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
    
    if (this.controls) {
      this.controls.dispose();
    }
    
    if (this.renderer) {
      this.renderer.dispose();
    }
    
    if (this.scene) {
      this.clearScene();
    }
  }
}
