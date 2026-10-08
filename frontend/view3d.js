// 3D flythrough: DSM mesh textured with the current map (overlay + arrows included).
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

export class Terrain3D {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xbfd3e0);
    this.camera = new THREE.PerspectiveCamera(50, 1, 1, 1e6);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445544, 1.6));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.6);
    this.scene.add(this.sun);
    this.mesh = null; this.exag = 1; this.visible = false;
    new ResizeObserver(() => this.resize()).observe(container);
    const loop = () => {
      requestAnimationFrame(loop);
      if (!this.visible || !this.mesh) return;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  resize() {
    const r = this.container.getBoundingClientRect();
    if (!r.width) return;
    this.renderer.setSize(r.width, r.height);
    this.camera.aspect = r.width / r.height;
    this.camera.updateProjectionMatrix();
  }

  setTerrain(heights, gw, gh, extentW, extentH, metric) {
    this.data = { heights, gw, gh, extentW, extentH, metric };
    let lo = Infinity, hi = -Infinity;
    for (const v of heights) { if (v < lo) lo = v; if (v > hi) hi = v; }
    this.lo = lo;
    // relative surfaces get a height range of 12% of the scene width
    this.zScale = metric ? 1 : (0.12 * Math.max(extentW, extentH)) / Math.max(hi - lo, 1e-6);
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); }
    const geo = new THREE.PlaneGeometry(extentW, extentH, gw - 1, gh - 1);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0 });
    this.mesh = new THREE.Mesh(geo, mat);
    this.scene.add(this.mesh);
    this.applyHeights();
    const span = Math.max(extentW, extentH);
    const mid = ((hi - lo) * this.zScale * this.exag) / 2;
    this.camera.near = span / 1000; this.camera.far = span * 20; this.camera.updateProjectionMatrix();
    this.camera.position.set(0, mid + span * 0.75, span * 0.85);
    this.controls.target.set(0, mid, 0);
    this.sun.position.set(-span, span, span * 0.5);
    this.resize();
  }

  applyHeights() {
    if (!this.mesh) return;
    const { heights } = this.data;
    const pos = this.mesh.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) pos.setY(i, (heights[i] - this.lo) * this.zScale * this.exag);
    pos.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
  }

  setExaggeration(x) { this.exag = x; this.applyHeights(); }

  setTexture(canvas) {
    if (!this.mesh) return;
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    if (this.mesh.material.map) this.mesh.material.map.dispose();
    this.mesh.material.map = tex;
    this.mesh.material.needsUpdate = true;
  }
}
