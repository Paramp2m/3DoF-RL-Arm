/* Live demo: the trained hold policy steers the arm in MuJoCo (WebAssembly), drawn with three.js.
   You drag the target; the physics and the network run in this page, nothing is pre-recorded. */
import * as THREE from "../vendor/three.module.min.js";
import { createSim } from "./armsim.mjs";

export async function initLive({ canvas, status, fallback, roamBtn, randomBtn }) {
  const say = t => { if (status) status.textContent = t; };
  try {
    say("Loading the physics engine (10 MB, once)…");
    const here = p => new URL(p, import.meta.url);
    const [{ default: load }, xml, bin, cfg] = await Promise.all([import("../vendor/mujoco.js"), fetch(here("arm.xml")).then(r => r.text()), fetch(here("policy.bin")).then(r => r.arrayBuffer()), fetch(here("policy.json")).then(r => r.json())]);
    const mj = await load(), sim = await createSim(mj, xml, bin, cfg);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    // reachable targets: forward kinematics over the joint limits, so a dragged ball snaps to a place the arm can actually reach
    const pts = [];
    for (let y = -3; y <= 3; y += 0.3) for (let s = sim.lo[1]; s <= sim.hi[1]; s += 0.09) for (let e = sim.lo[2]; e <= sim.hi[2]; e += 0.1) pts.push(sim.fk(y, s, e));
    const nearest = p => { let b = pts[0], d = 1e9; for (const q of pts) { const k = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2; if (k < d) { d = k; b = q; } } return b; };
    const randomGoal = () => sim.fk((Math.random() - .5) * 3, sim.lo[1] + Math.random() * (sim.hi[1] - sim.lo[1]), sim.lo[2] + Math.random() * (sim.hi[2] - sim.lo[2]));

    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x24262d);
    const cam = new THREE.PerspectiveCamera(36, 1, 0.05, 10); cam.up.set(0, 0, 1); cam.position.set(0.95, -0.95, 0.6); cam.lookAt(0.17, 0, 0.22);
    scene.add(new THREE.AmbientLight(0xffffff, 1.1)); const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(0.6, -0.4, 1.4); scene.add(sun);
    const mat = c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.1 });
    const chk = document.createElement("canvas"); chk.width = chk.height = 256; const g = chk.getContext("2d");
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { g.fillStyle = (i + j) % 2 ? "#5a5d68" : "#43464f"; g.fillRect(i * 32, j * 32, 32, 32); }
    const tex = new THREE.CanvasTexture(chk); tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(3, 3); tex.colorSpace = THREE.SRGBColorSpace;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.MeshStandardMaterial({ map: tex, roughness: 1 })); scene.add(floor);
    const column = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.06, 32).rotateX(Math.PI / 2).translate(0, 0, 0.03), mat(0x9aa0ad)); scene.add(column);
    const link = (len, r, color) => { const grp = new THREE.Group(); grp.add(new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 8, 20).rotateZ(-Math.PI / 2).translate(len / 2, 0, 0), mat(color))); scene.add(grp); return grp; };
    const housing = new THREE.Mesh(new THREE.SphereGeometry(0.035, 24, 16), mat(0x9aa0ad)); scene.add(housing);
    const upper = link(0.30, 0.025, 0xeceef2), fore = link(0.25, 0.02, 0xb8bcc7);
    const tipM = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })); scene.add(tipM);
    const goalM = new THREE.Mesh(new THREE.SphereGeometry(0.022, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 })); scene.add(goalM);
    const halo = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.036, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.6 })); scene.add(halo);

    const pose = (m, body) => { const p = sim.data.xpos, q = sim.data.xquat; m.position.set(p[body * 3], p[body * 3 + 1], p[body * 3 + 2]); m.quaternion.set(q[body * 4 + 1], q[body * 4 + 2], q[body * 4 + 3], q[body * 4]); };
    const draw = () => {
      pose(housing, 2); pose(upper, 3); pose(fore, 4); tipM.position.set(...sim.tip());
      goalM.position.set(...sim.goal); halo.position.set(...sim.goal); halo.quaternion.copy(cam.quaternion); halo.scale.setScalar(1 + 0.15 * Math.sin(performance.now() / 260));
      const w = canvas.clientWidth, h = canvas.clientHeight; if (canvas.width !== Math.round(w * renderer.getPixelRatio())) { renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
      renderer.render(scene, cam);
    };

    // target: roams by itself until you take it (same idea as scripts/demo.py --headless)
    let roam = true, wp = { from: [0, 1, -.5], to: [0, 1, -.5], t0: 0, dur: 1 };
    const nextWp = now => { const q = [(Math.random() - .5) * 3, sim.lo[1] + Math.random() * (sim.hi[1] - sim.lo[1]), sim.lo[2] + Math.random() * (sim.hi[2] - sim.lo[2])]; const dist = Math.max(...q.map((v, k) => Math.abs(v - wp.to[k]))); wp = { from: wp.to, to: q, t0: now, dur: dist / 0.5 * 1000 + 1500, move: dist / 0.5 * 1000 }; };
    const roamStep = now => { if (now - wp.t0 > wp.dur) nextWp(now); const k = Math.min(1, (now - wp.t0) / wp.move), s = k * k * (3 - 2 * k); sim.setGoal(sim.fk(...wp.from.map((v, i) => v + s * (wp.to[i] - v)))); };
    sim.setGoal(sim.fk(0.4, 1.0, -0.2)); wp.to = [0.4, 1.0, -0.2]; wp.t0 = performance.now() - 5000;
    const setRoam = on => { roam = on; if (roamBtn) roamBtn.setAttribute("aria-pressed", on); if (on) { wp.to = [sim.data.qpos[0], sim.data.qpos[1], sim.data.qpos[2]]; wp.t0 = performance.now() - 5000; } };
    if (roamBtn) roamBtn.onclick = () => setRoam(!roam);
    if (randomBtn) randomBtn.onclick = () => { setRoam(false); sim.setGoal(randomGoal()); };

    // drag: ray from the pointer onto a plane through the target facing the camera, then snap to the reachable set
    const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(); let dragging = false;
    const place = ev => {
      const r = canvas.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ndc, cam);
      const n = new THREE.Vector3(); cam.getWorldDirection(n); const hit = new THREE.Vector3();
      if (ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(n, new THREE.Vector3(...sim.goal)), hit)) sim.setGoal(nearest([hit.x, hit.y, Math.max(hit.z, 0.03)]));
    };
    const near = ev => { const r = canvas.getBoundingClientRect(), p = new THREE.Vector3(...sim.goal).project(cam); return Math.hypot(ev.clientX - (p.x + 1) / 2 * r.width - r.left, ev.clientY - (1 - p.y) / 2 * r.height - r.top) < 56; };
    canvas.addEventListener("pointerdown", ev => { if (!near(ev)) return; dragging = true; setRoam(false); canvas.setPointerCapture(ev.pointerId); canvas.classList.add("grab"); });
    canvas.addEventListener("pointermove", ev => { if (dragging) place(ev); else canvas.style.cursor = near(ev) ? "grab" : "default"; });
    const end = () => { dragging = false; canvas.classList.remove("grab"); };
    canvas.addEventListener("pointerup", end); canvas.addEventListener("pointercancel", end);

    let acc = 0, sub = 0, last = performance.now(), seen = true, lastTxt = 0; const dts = sim.model.opt.timestep;
    new IntersectionObserver(e => { seen = e[0].isIntersecting; last = performance.now(); }).observe(canvas);
    (function frame(now) {
      requestAnimationFrame(frame); if (!seen) return;
      acc += Math.min(0.1, (now - last) / 1000); last = now; if (roam) roamStep(now);
      while (acc >= dts) { if (sub === 0) sim.control(); sim.substep(); sub = (sub + 1) % cfg.frame_skip; acc -= dts; }
      draw();
      if (now - lastTxt > 250) { lastTxt = now; const d = Math.hypot(...sim.tip().map((v, k) => v - sim.goal[k])) * 1000; say(`Tip ${d.toFixed(0)} mm from the target${d < 20 ? ", inside the 2 cm success zone" : ""}`); }
    })(last);
    canvas.dataset.ready = "1"; canvas.sim = sim; canvas.cam = cam;
  } catch (err) {
    console.warn("live demo unavailable:", err); canvas.hidden = true; if (fallback) fallback.hidden = false; say("The live demo needs WebGL and WebAssembly. Showing a recording instead.");
  }
}
