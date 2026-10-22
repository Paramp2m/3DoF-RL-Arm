/* Round, playful pieces shared by v5-v8: ring buttons, the pointer-driven arm toy, flower clip, copy button. */
function rings(root) {
  root.innerHTML = RL.tasks.map((t, i) =>
    `<button class="ring" data-task="${t.id}" style="--i:${i}"><svg viewBox="0 0 120 120" aria-hidden="true"><circle class="t" cx="60" cy="60" r="52"/><circle class="p" cx="60" cy="60" r="52" pathLength="100" stroke-dasharray="${t.success} 100" transform="rotate(-90 60 60)"/></svg><b>${Math.round(t.success)}<small>%</small></b><span>${t.name}</span></button>`).join("");
}

/* Two-link arm in the page: closed-form IK, not the learned policy. svg needs #base, #l1, #l2, #j, #tip, #goal. */
function armToy(svg, status) {
  const $ = s => svg.querySelector(s), B = { x: 400, y: 440 }, L1 = 230, L2 = 200;
  let g = { x: 580, y: 190 }, touched = false, t0 = performance.now();
  function draw() {
    const dx = g.x - B.x, dy = g.y - B.y, d = Math.hypot(dx, dy), far = d > L1 + L2 - 2;
    const dd = Math.min(Math.max(d, Math.abs(L1 - L2) + 2), L1 + L2 - 2), a = Math.atan2(dy, dx);
    const s = a - Math.acos((L1 * L1 + dd * dd - L2 * L2) / (2 * L1 * dd));
    const e = { x: B.x + L1 * Math.cos(s), y: B.y + L1 * Math.sin(s) }, p = { x: B.x + dd * Math.cos(a), y: B.y + dd * Math.sin(a) };
    $("#l1").setAttribute("d", `M${B.x} ${B.y}L${e.x} ${e.y}`); $("#l2").setAttribute("d", `M${e.x} ${e.y}L${p.x} ${p.y}`);
    $("#j").setAttribute("cx", e.x); $("#j").setAttribute("cy", e.y);
    $("#tip").setAttribute("cx", p.x); $("#tip").setAttribute("cy", p.y);
    $("#goal").setAttribute("cx", g.x); $("#goal").setAttribute("cy", g.y);
    if (status) status.textContent = far ? "Out of reach: the arm stretches as far as it can" : "Within reach: both joints solved exactly";
  }
  function put(ev) {
    const m = svg.getScreenCTM().inverse(), pt = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m);
    g = { x: Math.min(Math.max(pt.x, 20), 780), y: Math.min(Math.max(pt.y, 20), 440) }; touched = true; draw();
  }
  svg.addEventListener("pointermove", put); svg.addEventListener("pointerdown", put);
  (function idle(now) {
    if (!touched && !matchMedia("(prefers-reduced-motion: reduce)").matches) { const t = (now - t0) / 1800; g = { x: 400 + 190 * Math.cos(t), y: 250 + 120 * Math.sin(t * 1.3) }; draw(); }
    requestAnimationFrame(idle);
  })(t0);
  draw();
}

/* Scalloped clip for a picture: n petals of depth amp (0..1 of the radius). */
function flower(el, n, amp) {
  const pts = []; for (let i = 0; i < 180; i++) { const a = i / 180 * Math.PI * 2, r = 50 - amp * 50 * (0.5 + 0.5 * Math.cos(n * a)); pts.push((50 + r * Math.cos(a)).toFixed(2) + "% " + (50 + r * Math.sin(a)).toFixed(2) + "%"); }
  el.style.clipPath = "polygon(" + pts.join(",") + ")";
}
