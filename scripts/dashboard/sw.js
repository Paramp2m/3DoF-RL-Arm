// rl-arm dashboard service worker: keeps the last copy of the page and of every reply this browser saw,
// so a phone that can't reach the laptop still shows the latest numbers, marked as old (header x-saved).
// It always asks the laptop first; nothing here is ever sent anywhere.
const CACHE = "rlarm-v1", WAIT = 5000, MEDIA_MAX = 60, EVERY = 15000;
const last = {}; // cache key -> when it was last saved (saving the 120 kB state every 2 s is wasted work)

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin || u.pathname === "/sw.js") return;
  e.respondWith(answer(e, u));
});

// one saved copy per page/reply; /api/state's ?v=.. only asks "changed?", so it shares one key
const keyOf = (u) => (u.pathname === "/" || u.pathname === "/index.html" ? "/" : u.pathname === "/api/state" ? "/api/state" : u.pathname + u.search);

async function answer(e, u) {
  const c = await caches.open(CACHE), key = keyOf(u);
  const net = fetch(e.request).then((res) => {
    if (res.ok) { const p = save(c, key, res.clone(), u).catch(() => {}); try { e.waitUntil(p); } catch { /* already answered from the saved copy */ } }
    return res;
  });
  const old = await c.match(key);
  if (!old) return net;
  // laptop asleep, off, or off Tailscale: answer with the saved copy after a few seconds instead of hanging
  const late = new Promise((r) => setTimeout(r, WAIT));
  return Promise.race([net, late]).then((res) => res || old).catch(() => old);
}

async function save(c, key, res, u) {
  const now = Date.now(), media = u.pathname.startsWith("/file/");
  if (!media && key !== "/" && now - (last[key] || 0) < EVERY && (await c.match(key))) return;
  last[key] = now;
  const head = { "content-type": res.headers.get("content-type") || "", "x-saved": String(now) };
  let body;
  if (key === "/api/state") {
    const d = await res.json();
    if (d.same || d.error) return; // "nothing changed" replies carry no data
    // a changed reply leaves out the decision log when the page already has it: keep the saved one
    if (!d.decisions) { const o = await c.match(key); d.decisions = o ? (await o.json()).decisions || [] : []; }
    body = JSON.stringify(d);
  } else body = await res.blob();
  await c.put(key, new Response(body, { headers: head }));
  if (media) { const ks = (await c.keys()).filter((r) => new URL(r.url).pathname.startsWith("/file/")); for (const r of ks.slice(0, -MEDIA_MAX)) await c.delete(r); }
}
