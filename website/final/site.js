/* Shared data + behaviour for every variant. Numbers: docs/results.md (3 seeds, 100 fixed eval goals). */
const RL = {
  media: (document.currentScript && document.currentScript.src.replace(/site\.js.*$/, "media/")) || "../shared/media/",
  tasks: [
    { id: "reach", name: "Reach", sentence: "Move the tip to a random goal and stop within 2 cm.", success: 96.0, std: 0.0, seeds: [96, 96, 96], median: 5.7, steps: "1M", clip: "phase3_reach" },
    { id: "hold", name: "Hold", sentence: "Reach, then stay within 2 cm for the last 50 of 250 steps.", success: 95.3, std: 2.5, seeds: [93, 95, 98], median: 5.1, steps: "1M", clip: "phase3_hold" },
    { id: "track", name: "Track", sentence: "Follow a target circling at 8 cm radius, one lap every 8 seconds.", success: 100.0, std: 0.0, seeds: [100, 100, 100], median: 3.5, extra: "tracking error 3.7 mm", steps: "300k", clip: "phase5_track" },
    { id: "obstacle", name: "Obstacle", sentence: "Reach around a box that sits across the straight line to the goal.", success: 88.7, std: 3.1, seeds: [86, 88, 92], median: 8.1, extra: "collisions 1.7%", steps: "400-450k", clip: "phase5_obstacle" },
    { id: "push", name: "Push", sentence: "Push a free 5 cm cube to a goal with a pusher pad.", success: 92.7, std: 2.3, seeds: [90, 94, 94], median: 6.5, steps: "1M", clip: "phase5_push" }
  ],
  exps: [
    { id: "e1", name: "Action modes", q: "Which action interface learns reach best?", img: "e1_action_modes_curves.png", hi: 0,
      bars: [["delta_pos", 70.0, 11.5], ["torque_gravcomp", 66.0, 9.6], ["torque", 45.7, 14.0], ["abs_pos", 31.7, 1.5]],
      note: "SAC, 300k steps, 3 seeds. Torque modes reach the goal in 8 to 10 steps against 36 for delta_pos, but move more jerkily. Every curve was still rising." },
    { id: "e2", name: "SAC or PPO", q: "Same environment, reward and network: which algorithm?", img: "e2_algorithms_curves.png", hi: 0,
      bars: [["SAC", 70.0, 11.5], ["PPO", 49.3, 2.3]],
      note: "PPO is smoother (mean action change 0.050 against 0.180) but learns more slowly in this budget." },
    { id: "e3", name: "Sparse reward", q: "Can a pure did-it-reach reward match the shaped one?", img: "e3_her_sparse_curves.png", hi: 0,
      bars: [["SAC + HER, sparse reward", 85.0, 5.3], ["SAC, shaped reward", 70.0, 11.5]],
      note: "With hindsight experience replay the sparse reward won by 15 points and reached the goal closer (8.4 mm against 13.5 mm median)." },
    { id: "e4", name: "Versus IK", q: "How close is the learned policy to closed-form inverse kinematics?", img: "", hi: 0,
      bars: [["Closed-form IK, no training", 100.0, 0], ["Learned policy, 1M steps", 96.0, 0]],
      note: "IK needs no training and is exact (0.0 mm). The learned policy lands within 4 points, with a median miss of 5.7 mm." }
  ]
};

/* Task explorer: root holds [data-task] buttons, a <video data-vid> and [data-f=...] readouts. */
function explorer(root, first) {
  const vid = root.querySelector("[data-vid]");
  const btns = [...root.querySelectorAll("[data-task]")];
  function pick(id) {
    const t = RL.tasks.find(x => x.id === id) || RL.tasks[0];
    btns.forEach(b => { const on = b.dataset.task === t.id; b.setAttribute("aria-pressed", on); b.classList.toggle("on", on); });
    if (vid) { vid.poster = RL.media + t.clip + ".jpg"; vid.src = RL.media + t.clip + (vid.canPlayType("video/mp4; codecs=\"avc1.42E01E\"") ? ".mp4" : ".webm"); vid.play().catch(() => {}); }
    const f = {
      name: t.name, sentence: t.sentence, success: t.success.toFixed(1), std: "± " + t.std.toFixed(1),
      seeds: t.seeds.join("%, ") + "%", median: t.median.toFixed(1), extra: t.extra ? " \u00b7 " + t.extra : "", steps: t.steps + " steps"
    };
    root.querySelectorAll("[data-f]").forEach(el => { el.textContent = f[el.dataset.f] ?? ""; });
    root.querySelectorAll("[data-bar]").forEach(el => { el.style.setProperty("--w", t.success + "%"); });
    root.dispatchEvent(new CustomEvent("pick", { detail: t }));
  }
  btns.forEach(b => b.addEventListener("click", () => pick(b.dataset.task)));
  btns.forEach(b => b.addEventListener("mouseenter", () => { if (matchMedia("(hover:hover)").matches) pick(b.dataset.task); }));
  pick(first || RL.tasks[0].id);
}

/* Experiment sheet: [data-exp] buttons, [data-exp-bars], [data-exp-img], [data-exp-title], [data-exp-q], [data-exp-note]. */
function experiments(root) {
  const btns = [...root.querySelectorAll("[data-exp]")];
  function pick(id) {
    const e = RL.exps.find(x => x.id === id) || RL.exps[0];
    btns.forEach(b => { const on = b.dataset.exp === e.id; b.setAttribute("aria-pressed", on); b.classList.toggle("on", on); });
    const set = (k, v) => root.querySelectorAll(k).forEach(el => { el.textContent = v; });
    set("[data-exp-title]", e.name); set("[data-exp-q]", e.q); set("[data-exp-note]", e.note);
    const bars = root.querySelector("[data-exp-bars]");
    if (bars) bars.innerHTML = e.bars.map((b, i) =>
      `<div class="bar${i === e.hi ? " hi" : ""}"><span class="bl">${b[0]}</span><span class="bt"><i style="--w:${b[1]}%"></i></span><span class="bv">${b[1].toFixed(1)}${b[2] ? " ± " + b[2].toFixed(1) : ""}%</span></div>`).join("");
    const img = root.querySelector("[data-exp-img]");
    if (img) { img.hidden = !e.img; if (e.img) { img.src = RL.media + e.img; img.alt = "Learning curves for " + e.name + ": validation success against environment steps, mean and spread over three seeds"; } }
    root.classList.remove("swap"); void root.offsetWidth; root.classList.add("swap");
  }
  btns.forEach(b => b.addEventListener("click", () => pick(b.dataset.exp)));
  pick(RL.exps[0].id);
}

/* Entrance: elements with .rv rise and fade once as they enter, staggered inside a parent by data-i. */
function reveal() {
  const els = [...document.querySelectorAll(".rv")];
  if (!("IntersectionObserver" in window) || matchMedia("(prefers-reduced-motion: reduce)").matches) { els.forEach(e => e.classList.add("in")); return; }
  const io = new IntersectionObserver(es => es.forEach(en => {
    if (!en.isIntersecting) return;
    const sibs = [...en.target.parentElement.children].filter(c => c.classList.contains("rv"));
    en.target.style.transitionDelay = Math.min(sibs.indexOf(en.target), 5) * 75 + "ms";
    en.target.classList.add("in"); io.unobserve(en.target);
  }), { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  els.forEach(e => io.observe(e));
}

/* Copy button for the run command. */
function copier(btn, text) {
  btn.addEventListener("click", () => {
    const done = () => { const o = btn.textContent; btn.textContent = "Copied"; setTimeout(() => { btn.textContent = o; }, 1400); };
    const t = typeof text === "function" ? text() : text;
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(done, done);
  });
}

document.documentElement.classList.add("js");
