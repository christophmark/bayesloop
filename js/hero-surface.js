'use strict';

/* ============================================================
   bayesloop.com — animated hero background (three.js)

   Renders a time-varying probability density as a glossy,
   grid-lined 3D surface behind the hero copy — no axes, no box.
   The density is a mixture of wandering bivariate Gaussians
   whose means, covariances and weights drift smoothly, so
   probability mass visibly moves, spreads, splits and
   re-concentrates over a regular lattice: a picture of what
   bayesloop computes.

   three.js (~650 kB) is imported from jsDelivr only if WebGL is
   available. Until the first frame is rendered the hero shows
   its static fallback (centered copy, SVG waves); then the
   'hero--live' class fades the canvas in and slides the copy to
   the right (see the hero-animation block in css/style.css).
   If anything fails, the fallback simply remains.
   ============================================================ */

const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js';

/* ----------------------------- look ----------------------------- */
const SURFACE_COLOR = 0x2f8fc9;   // a touch lighter than --accent-strong for the dark bg
const SPECULAR_COLOR = 0x6c93b4;  // plastic sheen (kept subtle)
const SHININESS = 58;
const GRID_COLOR = 0x081c2e;
const GRID_OPACITY = 0.5;

const EXT = 3.2;        // world size of the sheet
const SEGMENTS = 114;   // surface resolution per side
const GRID_STEP = 6;    // gridline every GRID_STEP segments -> 19x19 cells
const GRID_LIFT = 0.0045;
const AMP = 0.39;       // pdf -> world-height scale

/* --------------------------- framing --------------------------- */
const CAM_DIR = { x: -3.4, y: 3.15, z: 4.8 };  // viewing direction (length = base distance)
const CAM_TARGET_Y = 0.50;
const ROTATION_SPEED = 0.0132;  // rad/s, continuous one-way orbit (~8 min per turn)
const SHEET_PX_PER_H = 1.42;   // projected sheet width ≈ this × canvas height at base distance
const MIN_SIDE_SPACE = 430;    // px left of the copy required for the side-by-side layout
const CONTAINER_MAX = 1080;    // keep in sync with .container in style.css
const CONTAINER_PAD = 24;

/* --------------------------- dynamics --------------------------- */
/* Mixture components: means wander on incommensurate sine pairs,
   covariances breathe, weights fade modes in and out. Densities are
   normalized, so narrow peaks are tall and wide ones flat (mass
   conservation — the "probability" look). */
const COMPONENTS = [
  {
    mx: t => 0.66 * Math.sin(0.31 * t + 1.4) + 0.21 * Math.sin(0.523 * t + 0.4),
    mz: t => 0.60 * Math.sin(0.27 * t + 4.2) + 0.23 * Math.sin(0.441 * t + 2.1),
    sx: t => 0.215 + 0.065 * Math.sin(0.37 * t + 0.8),
    sz: t => 0.215 + 0.065 * Math.sin(0.29 * t + 2.7),
    rho: t => 0.55 * Math.sin(0.23 * t + 1.0),
    w: t => 0.62 + 0.38 * Math.sin(0.157 * t + 0.9),
  },
  {
    mx: t => 0.58 * Math.sin(0.24 * t + 3.6) + 0.20 * Math.sin(0.39 * t + 5.1),
    mz: t => 0.63 * Math.sin(0.33 * t + 0.6) + 0.18 * Math.sin(0.47 * t + 3.9),
    sx: t => 0.23 + 0.06 * Math.sin(0.31 * t + 4.4),
    sz: t => 0.23 + 0.06 * Math.sin(0.41 * t + 1.6),
    rho: t => 0.50 * Math.sin(0.19 * t + 4.8),
    w: t => 0.50 + 0.50 * Math.sin(0.119 * t + 3.4),
  },
  {
    mx: t => 0.70 * Math.sin(0.21 * t + 5.5),
    mz: t => 0.68 * Math.sin(0.26 * t + 2.4),
    sx: t => 0.26 + 0.05 * Math.sin(0.27 * t + 1.1),
    sz: t => 0.26 + 0.05 * Math.sin(0.23 * t + 5.0),
    rho: t => 0.45 * Math.sin(0.17 * t + 2.2),
    w: t => 0.30 + 0.70 * Math.sin(0.087 * t + 5.8),
  },
];
const TIME_OFFSET = 6;  // start at a visually interesting moment

/* Snapshot of all component parameters at time t, with weights squared
   (sharper on/off transitions) and normalized. The small floor keeps the
   normalization away from zero when all raw weights happen to vanish. */
function sampleComponents(t) {
  const out = [];
  let wsum = 0;
  for (const c of COMPONENTS) {
    const raw = Math.max(0, c.w(t));
    const w = raw * raw + 0.012;
    const sx = c.sx(t);
    const sz = c.sz(t);
    const rho = c.rho(t);
    const omr = 1 - rho * rho;
    out.push({
      mx: c.mx(t), mz: c.mz(t),
      isx: 1 / sx, isz: 1 / sz,
      rho,
      expScale: -1 / (2 * omr),
      norm: w / (2 * Math.PI * sx * sz * Math.sqrt(omr)),
    });
    wsum += w;
  }
  for (const c of out) c.norm /= wsum;
  return out;
}

function initHeroSurface() {
  const hero = document.getElementById('hero');
  const holder = document.getElementById('hero-canvas');
  const copy = document.querySelector('.hero-copy');
  const surfaceSlot = document.querySelector('.hero-surface-slot');
  if (!hero || !holder || !copy) return;

  /* No WebGL -> keep the static fallback. */
  let gl = null;
  try {
    const probe = document.createElement('canvas');
    gl = probe.getContext('webgl2') || probe.getContext('webgl');
  } catch (e) { /* fall through */ }
  if (!gl) return;

  const reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  import(THREE_URL)
    .then((THREE) => start(THREE))
    .catch(() => { /* CDN unreachable -> static fallback */ });

  function start(THREE) {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    holder.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    const baseDist = Math.hypot(CAM_DIR.x, CAM_DIR.y, CAM_DIR.z);
    const camDir = {
      x: CAM_DIR.x / baseDist,
      y: CAM_DIR.y / baseDist,
      z: CAM_DIR.z / baseDist,
    };
    let camDist = baseDist;
    const camTarget = new THREE.Vector3(0, CAM_TARGET_Y, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.62));
    const key = new THREE.DirectionalLight(0xffffff, 1.85);
    key.position.set(-3.0, 4.2, 2.2);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(3.5, 1.5, -2.5);
    scene.add(fill);

    /* --- surface sheet */
    const geo = new THREE.PlaneGeometry(EXT, EXT, SEGMENTS, SEGMENTS);
    geo.rotateX(-Math.PI / 2);  // lie in the XZ plane, +Y up
    const pos = geo.attributes.position;
    const posArr = pos.array;
    const surface = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
      color: SURFACE_COLOR,
      specular: SPECULAR_COLOR,
      shininess: SHININESS,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    }));
    surface.frustumCulled = false;
    scene.add(surface);

    /* --- gridlines, sampled on the same vertices so they hug the sheet */
    const lineMap = [];
    for (let l = 0; l <= SEGMENTS; l += GRID_STEP) {
      for (let i = 0; i < SEGMENTS; i++) {
        lineMap.push(l * (SEGMENTS + 1) + i, l * (SEGMENTS + 1) + i + 1);      // along x
        lineMap.push(i * (SEGMENTS + 1) + l, (i + 1) * (SEGMENTS + 1) + l);    // along z
      }
    }
    const lineArr = new Float32Array(lineMap.length * 3);
    for (let i = 0; i < lineMap.length; i++) {
      lineArr[i * 3] = posArr[lineMap[i] * 3];
      lineArr[i * 3 + 2] = posArr[lineMap[i] * 3 + 2];
    }
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(lineArr, 3));
    const grid = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
      color: GRID_COLOR,
      transparent: true,
      opacity: GRID_OPACITY,
    }));
    grid.frustumCulled = false;
    scene.add(grid);

    function updateField(t) {
      const comps = sampleComponents(t);
      const nv = pos.count;
      for (let v = 0; v < nv; v++) {
        const x = posArr[v * 3];
        const z = posArr[v * 3 + 2];
        let h = 0;
        for (const c of comps) {
          const dx = (x - c.mx) * c.isx;
          const dz = (z - c.mz) * c.isz;
          const q = (dx * dx - 2 * c.rho * dx * dz + dz * dz) * c.expScale;
          if (q > -9) h += c.norm * Math.exp(q);
        }
        posArr[v * 3 + 1] = AMP * h;
      }
      pos.needsUpdate = true;
      geo.computeVertexNormals();
      for (let m = 0; m < lineMap.length; m++) {
        lineArr[m * 3 + 1] = posArr[lineMap[m] * 3 + 1] + GRID_LIFT;
      }
      lineGeo.attributes.position.needsUpdate = true;
    }

    function updateCamera(t) {
      const a = ROTATION_SPEED * t;
      const cos = Math.cos(a);
      const sin = Math.sin(a);
      camera.position.set(
        (camDir.x * cos - camDir.z * sin) * camDist,
        camDir.y * camDist + 0.06 * Math.sin(0.07 * t),
        (camDir.x * sin + camDir.z * cos) * camDist
      );
      camera.lookAt(camTarget);
    }

    /* Side-by-side only when there is enough room left of the copy; the
       same flag drives the CSS layout (hero--wide). The camera distance is
       chosen so the projected sheet fits that space, and a view offset
       shifts it into the middle of it. */
    function layout() {
      const w = hero.clientWidth;
      const h = hero.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;

      const contentLeft = Math.max(CONTAINER_PAD, (w - CONTAINER_MAX) / 2 + CONTAINER_PAD);
      const copyW = copy.getBoundingClientRect().width;
      const copyFinalLeft = (w - contentLeft) - copyW;  // copy ends up flush right in the container
      const sideSpace = copyFinalLeft;
      const wide = sideSpace >= MIN_SIDE_SPACE;
      hero.classList.toggle('hero--wide', wide);

      let targetW, centerX, yShift;
      if (wide) {
        targetW = Math.min(sideSpace * 1.08, 1.75 * h);
        centerX = sideSpace / 2;
        yShift = 0;
        camTarget.y = CAM_TARGET_Y;
      } else {
        const heroRect = hero.getBoundingClientRect();
        const slotRect = surfaceSlot ? surfaceSlot.getBoundingClientRect() : null;
        const hasSurfaceSlot = Boolean(slotRect && slotRect.height > 0);

        if (hasSurfaceSlot) {
          const slotCenterY = slotRect.top - heroRect.top + slotRect.height / 2;
          targetW = Math.min(w * 0.88, slotRect.width * 1.02, slotRect.height * 1.38);
          centerX = w / 2;
          yShift = Math.round(h / 2 - slotCenterY);
          camTarget.y = 0.35;
        } else {
          /* centered overlay: sheet sits in the lower part of the hero so the
             headline stays over dark background */
          targetW = Math.min(w * 0.95, 1.4 * h);
          centerX = w / 2;
          yShift = -Math.round(h * 0.2);
          camTarget.y = CAM_TARGET_Y;
        }
      }
      camDist = baseDist * Math.min(2.6, Math.max(0.8,
        (SHEET_PX_PER_H * h) / Math.max(targetW, 1) ));
      camera.setViewOffset(w, h, Math.round(w / 2 - centerX), yShift, w, h);
      camera.updateProjectionMatrix();
    }

    let running = false;
    let rafId = 0;

    function frame() {
      const t = TIME_OFFSET + performance.now() * 0.001;
      updateField(t);
      updateCamera(t);
      renderer.render(scene, camera);
      if (running) rafId = requestAnimationFrame(frame);
    }

    layout();
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(() => {
        layout();
        if (!running) frame();  // keep the paused frame undistorted
      }).observe(hero);
    } else {
      window.addEventListener('resize', layout);
    }

    /* First frame, then reveal (canvas fade + copy slide via CSS). */
    updateField(TIME_OFFSET);
    updateCamera(TIME_OFFSET);
    renderer.render(scene, camera);
    requestAnimationFrame(() => hero.classList.add('hero--live'));

    if (reduceMotion) return;  // static frame only

    /* Animate only while the hero is actually on screen. */
    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver((entries) => {
        const visible = entries[0].isIntersecting;
        if (visible && !running) {
          running = true;
          rafId = requestAnimationFrame(frame);
        } else if (!visible && running) {
          running = false;
          cancelAnimationFrame(rafId);
        }
      }, { threshold: 0.02 }).observe(hero);
    } else {
      running = true;
      rafId = requestAnimationFrame(frame);
    }
  }
}

if (document.readyState !== 'loading') initHeroSurface();
else document.addEventListener('DOMContentLoaded', initHeroSurface);
