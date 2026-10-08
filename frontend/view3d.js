// 3D flythrough: DSM mesh textured with the current map (overlay + arrows included).
// Orbit by default; first-person "fly" mode and a "DEM only" comparison come from the
// original DepthWizard viewer (stratum patches: PointerLockControls, key B).
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { PointerLockControls } from './vendor/PointerLockControls.js';

const KEYS = { KeyW: 'f', ArrowUp: 'f', KeyS: 'b', ArrowDown: 'b', KeyA: 'l', ArrowLeft: 'l',
  KeyD: 'r', ArrowRight: 'r', KeyE: 'u', Space: 'u', KeyQ: 'd', KeyC: 'd' };

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
    this.fly = new PointerLockControls(this.camera, this.renderer.domElement);
    this.fly.addEventListener('unlock', () => this.onFlyChange && this.onFlyChange(false));
    this.fly.addEventListener('lock', () => this.onFlyChange && this.onFlyChange(true));
    this.held = new Set();
    window.addEventListener('keydown', (e) => { if (this.fly.isLocked && KEYS[e.code]) { this.held.add(KEYS[e.code]); if (e.shiftKey) this.held.add('fast'); e.preventDefault(); } });
    window.addEventListener('keyup', (e) => { if (KEYS[e.code]) this.held.delete(KEYS[e.code]); if (!e.shiftKey) this.held.delete('fast'); });
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445544, 1.6));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.6);
    this.scene.add(this.sun);
    this.mesh = null; this.exag = 1; this.visible = false;
    this.raycaster = new THREE.Raycaster();
    this.clock = new THREE.Clock();
    new ResizeObserver(() => this.resize()).observe(container);
    const loop = () => {
      requestAnimationFrame(loop);
      const dt = Math.min(this.clock.getDelta(), 0.1);
      if (!this.visible || !this.mesh) return;
      if (this.fly.isLocked) this.flyStep(dt);
      else this.controls.update();
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
    this.lo = lo; this.hi = hi;
    // relative surfaces get a height range of 12% of the scene width
    this.zScale = metric ? 1 : (0.12 * Math.max(extentW, extentH)) / Math.max(hi - lo, 1e-6);
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); }
    const geo = new THREE.PlaneGeometry(extentW, extentH, gw - 1, gh - 1);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0 });
    this.mesh = new THREE.Mesh(geo, mat);
    this.scene.add(this.mesh);
    this.applyHeights();
    this.mesh.scale.y = this.exag;
    this.span = Math.max(extentW, extentH);
    this.resetCamera();
    this.sun.position.set(-this.span, this.span, this.span * 0.5);
    this.resize();
  }

  resetCamera() {
    const mid = ((this.hi - this.lo) * this.zScale * this.exag) / 2;
    this.camera.near = this.span / 2000; this.camera.far = this.span * 20; this.camera.updateProjectionMatrix();
    this.camera.position.set(0, mid + this.span * 0.75, this.span * 0.85);
    this.controls.target.set(0, mid, 0);
    this.camera.lookAt(this.controls.target);
  }

  // swap the surface under the same camera (DSM <-> DEM only); same grid size, same height datum
  setHeights(heights) {
    if (!this.mesh || heights.length !== this.data.heights.length) return;
    this.data.heights = heights;
    this.applyHeights();
  }

  applyHeights() {
    if (!this.mesh) return;
    const { heights } = this.data;
    const pos = this.mesh.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) pos.setY(i, (heights[i] - this.lo) * this.zScale);
    pos.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
    this.mesh.geometry.computeBoundingSphere();
  }

  // vertical exaggeration is a scale on the mesh, so the 1M-vertex grid is never rebuilt
  setExaggeration(x) { this.exag = x; if (this.mesh) this.mesh.scale.y = x; }

  setTexture(canvas) {
    if (!this.mesh) return;
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    if (this.mesh.material.map) this.mesh.material.map.dispose();
    this.mesh.material.map = tex;
    this.mesh.material.needsUpdate = true;
  }

  // ---- fly mode: WASD / arrows to move, mouse to look, E/Space up, Q/C down, Shift fast, Esc to leave
  startFly() {
    if (!this.mesh) return;
    const ground = this.groundAt(this.camera.position.x, this.camera.position.z);
    if (ground !== null && this.camera.position.y > ground + this.span * 0.15) {
      this.camera.position.y = ground + Math.max(30, this.span * 0.04);   // drop to low-flight height
    }
    this.fly.lock();
  }

  stopFly() { this.fly.unlock(); }

  groundAt(x, z) {
    this.raycaster.set(new THREE.Vector3(x, 1e7, z), new THREE.Vector3(0, -1, 0));
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    return hit ? hit.point.y : null;
  }

  flyStep(dt) {
    const speed = this.span * (this.held.has('fast') ? 0.25 : 0.06) * dt;
    if (this.held.has('f')) this.fly.moveForward(speed);
    if (this.held.has('b')) this.fly.moveForward(-speed);
    if (this.held.has('r')) this.fly.moveRight(speed);
    if (this.held.has('l')) this.fly.moveRight(-speed);
    if (this.held.has('u')) this.camera.position.y += speed;
    if (this.held.has('d')) this.camera.position.y -= speed;
    const ground = this.groundAt(this.camera.position.x, this.camera.position.z);
    if (ground !== null) this.camera.position.y = Math.max(this.camera.position.y, ground + 2);   // never through the terrain
  }
}
