/* The rl-arm hold policy running on MuJoCo (WebAssembly). Mirrors rlarm/arm.py (load_model) and
   rlarm/envs/base.py (obs + delta_pos step); scripts/export_web.py writes policy.* and ref.json. */
export async function createSim(mj, xml, bin, cfg) {
  mj.FS.writeFile("/arm.xml", xml);
  const model = mj.MjModel.from_xml_path("/arm.xml"), data = new mj.MjData(model);
  const kp = 400, kv = 20, tau = 12;
  model.opt.disableactuator = 1 << 1; // motor group off, position group on
  for (let i = 0; i < model.nu; i++) {
    model.actuator_ctrllimited[i] = 1;
    if (model.actuator_group[i] === 0) {
      const j = model.actuator_trnid[i * 2];
      model.actuator_gainprm[i * 10] = kp;
      model.actuator_biasprm[i * 10 + 1] = -kp; model.actuator_biasprm[i * 10 + 2] = -kv;
      model.actuator_ctrlrange[i * 2] = model.jnt_range[j * 2]; model.actuator_ctrlrange[i * 2 + 1] = model.jnt_range[j * 2 + 1];
      model.actuator_forcerange[i * 2] = -tau; model.actuator_forcerange[i * 2 + 1] = tau; model.actuator_forcelimited[i] = 1;
    } else { model.actuator_ctrlrange[i * 2] = -tau; model.actuator_ctrlrange[i * 2 + 1] = tau; }
  }
  for (let d = 0; d < 3; d++) { model.dof_damping[d] = 0.5; model.dof_armature[d] = 0.01; model.dof_frictionloss[d] = 0; }

  const f32 = new Float32Array(bin), layers = [];
  let off = 0;
  for (const [o, i] of cfg.layers) { layers.push({ o, i, w: f32.subarray(off, off + o * i), b: f32.subarray(off + o * i, off + o * i + o) }); off += o * i + o; }
  const lo = [0, 1, 2].map(j => model.jnt_range[j * 2]), hi = [0, 1, 2].map(j => model.jnt_range[j * 2 + 1]);
  const sim = { model, data, goal: [0.3, 0, 0.3], targetQ: [0, 0, 0], lo, hi, dt: cfg.frame_skip * model.opt.timestep };

  sim.tip = () => [...data.site_xpos.subarray(0, 3)];
  sim.setGoal = g => { sim.goal = g; data.mocap_pos[0] = g[0]; data.mocap_pos[1] = g[1]; data.mocap_pos[2] = g[2]; };
  sim.reset = (q = cfg.home) => { mj.mj_resetData(model, data); for (let k = 0; k < 3; k++) data.qpos[k] = q[k]; sim.targetQ = [...q]; mj.mj_forward(model, data); };
  sim.features = () => { // [sin q, cos q, scaled qdot, tip, tip, goal, (goal - tip) * scale] as GoalDiffExtractor builds it
    const q = data.qpos, tip = sim.tip(), g = sim.goal, x = new Float32Array(21);
    for (let k = 0; k < 3; k++) {
      x[k] = Math.sin(q[k]); x[3 + k] = Math.cos(q[k]);
      x[6 + k] = Math.max(-cfg.vel_clip, Math.min(cfg.vel_clip, data.qvel[k] * cfg.vel_scale));
      x[9 + k] = tip[k]; x[12 + k] = tip[k]; x[15 + k] = g[k]; x[18 + k] = (g[k] - tip[k]) * cfg.scale;
    }
    return x;
  };
  sim.act = x => { // actor MLP: ReLU hidden layers, tanh of the mean
    for (let n = 0; n < layers.length; n++) {
      const L = layers[n], y = new Float32Array(L.o);
      for (let r = 0; r < L.o; r++) { let s = L.b[r]; const row = r * L.i; for (let c = 0; c < L.i; c++) s += L.w[row + c] * x[c]; y[r] = n < layers.length - 1 ? Math.max(0, s) : Math.tanh(s); }
      x = y;
    }
    return x;
  };
  sim.control = () => { // the 25 Hz policy decision: move each joint setpoint by at most max_delta
    const a = sim.act(sim.features());
    sim.targetQ = sim.targetQ.map((t, k) => Math.max(lo[k], Math.min(hi[k], t + a[k] * cfg.max_delta)));
    return a;
  };
  sim.substep = () => { for (let k = 0; k < 3; k++) data.ctrl[k] = sim.targetQ[k]; mj.mj_step(model, data); };
  sim.step = () => { const a = sim.control(); for (let s = 0; s < cfg.frame_skip; s++) sim.substep(); return a; };
  sim.fk = (yaw, sh, el) => { // same chain as arm.xml: pitch axis -y raises the link
    const r = 0.30 * Math.cos(sh) + 0.25 * Math.cos(sh + el);
    return [r * Math.cos(yaw), r * Math.sin(yaw), 0.10 + 0.30 * Math.sin(sh) + 0.25 * Math.sin(sh + el)];
  };
  sim.reset();
  return sim;
}
