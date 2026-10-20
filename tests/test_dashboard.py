import gzip
import json
import os
import sys
import threading
import urllib.error
import urllib.request

import pytest

from rlarm.common import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import dashboard


@pytest.fixture(scope="module")
def url():
    srv = dashboard.serve(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}))
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_state_lists_phases_runs_docs_and_unchanged_poll_is_tiny(url):
    code, body = get(url + "/api/state")
    s = json.loads(body)
    assert code == 200 and len(s["phase"]["list"]) == 8
    assert {"runs", "machine", "decisions", "docs", "experiments"} <= s.keys()
    assert all(dashboard.NAME.match(r["id"]) for r in s["runs"])
    assert {"val_kind", "stopped_early", "usable"} <= s["runs"][0].keys()
    assert {"cpu", "cpu_temp", "chip_w", "mem_used", "battery"} <= s["machine"].keys()
    same = json.loads(get(url + "/api/state?v=" + s["v"])[1])
    assert same["same"] and same["v"] == s["v"] and "runs" not in same


def test_unchanged_poll_still_carries_fresh_machine_stats(url, monkeypatch):
    cpu = iter([1, 99])
    monkeypatch.setattr(dashboard, "machine", lambda: {"cpu": next(cpu)})
    dashboard._state.update(t=0.0)  # expire the 1 s cache
    v = json.loads(get(url + "/api/state")[1])["v"]
    dashboard._state.update(t=0.0)
    r = json.loads(get(url + "/api/state?v=" + v)[1])
    assert r["same"] and r["machine"]["cpu"] == 99


def test_a_crash_while_reading_the_project_returns_the_error(url, monkeypatch):
    def boom():
        raise KeyError("success")

    monkeypatch.setattr(dashboard, "state", boom)
    dashboard._state.update(t=0.0)
    code, body = get(url + "/api/state")
    assert code == 500 and "KeyError" in json.loads(body)["error"]
    dashboard._state.update(t=0.0, body=None)


def test_page_and_bundled_markdown_renderer_are_served(url):
    assert get(url + "/")[0] == 200
    assert get(url + "/marked.min.js")[0] == 200


def test_only_project_docs_and_media_are_readable(url):
    assert get(url + "/api/doc?path=CLAUDE.md")[0] == 200
    assert get(url + "/api/doc?path=../../../etc/passwd")[0] == 404
    assert get(url + "/api/doc?path=pyproject.toml")[0] == 404
    assert get(url + "/file/runs/../pyproject.toml")[0] == 404
    assert get(url + "/file/docs/media/../../CLAUDE.md")[0] == 404


def test_search_finds_docs_and_shows_plain_text_snippets(url):
    code, body = get(url + "/api/search?q=reach")
    hits = json.loads(body)
    assert code == 200 and any(h["kind"] == "doc" for h in hits)
    assert not any("**" in h["text"] for h in hits if h["kind"] == "doc")


def test_requests_from_other_sites_are_refused(url):
    assert get(url + "/api/state", {"Host": "evil.example"})[0] == 403
    req = urllib.request.Request(
        url + "/api/action",
        data=b'{"kind": "view"}',
        headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403


def test_actions_are_whitelisted():
    assert dashboard.action({"kind": "train"}) == {"error": "unknown action"}
    assert dashboard.action({"kind": "watch", "run": "../../etc"}) == {"error": "unknown run"}
    assert dashboard.action({"kind": "gate", "phase": "1; rm"}) == {"error": "bad phase"}
    for bad in (-1, 17, "2; rm", None):
        assert "error" in dashboard.action({"kind": "view", "speed": bad})


def test_config_diff_names_changed_settings_only():
    base = {"env": {"action_mode": "delta_pos", "max_delta": 0.05}, "seed": 0}
    cfg = {"env": {"action_mode": "torque", "max_delta": 0.05}, "seed": 2}
    rows = dashboard.diff(cfg, base)
    assert [r["key"] for r in rows] == ["env.action_mode"]
    assert rows[0]["name"] == "How the network drives the motors"


def test_dismiss_only_takes_real_run_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "DISMISSED", tmp_path / "dismissed.json")
    rid = next(r.name for r in dashboard.RUNS.iterdir() if dashboard.NAME.match(r.name))
    for bad in (None, [], ["../../etc"], [rid, "2026-01-01_nope_s0"]):
        assert "error" in dashboard.action({"kind": "dismiss", "runs": bad})
    assert "ok" in dashboard.action({"kind": "dismiss", "runs": [rid]})
    assert "ok" in dashboard.action({"kind": "dismiss", "runs": [rid]})
    assert json.loads((tmp_path / "dismissed.json").read_text()) == [rid]


def test_phone_via_tailscale_sees_everything_but_cannot_open_laptop_windows(url, monkeypatch):
    ts = "laptop.tail0000.ts.net"
    monkeypatch.setattr(dashboard, "HOSTS", {"127.0.0.1", "localhost", ts})
    assert get(url + "/api/state", {"Host": ts})[0] == 200
    req = urllib.request.Request(
        url + "/api/action",
        data=b'{"kind": "watch", "run": "nope"}',
        headers={"Content-Type": "application/json", "Host": ts, "Origin": f"https://{ts}"},
    )
    assert "laptop" in json.loads(urllib.request.urlopen(req).read())["error"]
    bad = urllib.request.Request(
        url + "/api/action",
        data=b'{"kind": ["watch"]}',  # not a string: refused, not a dropped connection
        headers={"Content-Type": "application/json", "Host": ts, "Origin": f"https://{ts}"},
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(bad)
    assert e.value.code == 400


def test_changed_polls_skip_the_decision_log_the_page_already_has(url):
    s = json.loads(get(url + "/api/state")[1])
    again = json.loads(get(url + f"/api/state?v=stale&dv={s['dv']}")[1])
    assert "decisions" in s and "decisions" not in again and again["runs"]


def test_laptop_page_data_covers_every_thread_and_names_this_dashboard(url):
    code, body = get(url + "/api/system")
    v = json.loads(body)
    assert code == 200 and len(v["cores"]) == os.cpu_count() and 0 <= v["cpu"] <= 100
    assert v["mem_total"] > 0 and v["disk_free"] > 0 and isinstance(v["hist"], list)
    assert any(p["kind"] == "self" for p in v["procs"])
    assert v["sleep_blockers"] is None or isinstance(v["sleep_blockers"], list)


def test_text_is_gzipped_for_browsers_that_accept_it(url):
    r = urllib.request.urlopen(
        urllib.request.Request(url + "/", headers={"Accept-Encoding": "gzip"})
    )
    assert r.headers["Content-Encoding"] == "gzip"
    assert gzip.decompress(r.read()).startswith(b"<!doctype html>")
    assert get(url + "/")[1].startswith(b"<!doctype html>")  # and plain for clients that don't


def test_page_carries_its_build_and_the_offline_worker_is_served(url):
    page = get(url + "/")[1].decode()
    build = json.loads(get(url + "/api/state")[1])["build"]
    assert f'BUILD = "{build}"' in page  # the page reloads itself when the state's build differs
    r = urllib.request.urlopen(url + "/sw.js")
    assert (
        r.headers["Content-Type"] == "text/javascript" and r.headers["Cache-Control"] == "no-cache"
    )


@pytest.mark.slow
def test_server_restarts_itself_on_new_code_but_not_on_broken_code(tmp_path):
    import shutil
    import socket
    import subprocess
    import time

    shutil.copytree(ROOT / "scripts" / "dashboard", tmp_path / "scripts" / "dashboard")
    src = tmp_path / "scripts" / "dashboard.py"
    shutil.copy(ROOT / "scripts" / "dashboard.py", src)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = tmp_path / "log.txt"
    cmd = [sys.executable, str(src), "--port", str(port), "--no-open"]
    p = subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT)

    def build() -> str:
        for _ in range(100):
            try:
                page = get(f"http://127.0.0.1:{port}/")[1].decode()  # an empty project: no state
                return page.split('BUILD = "')[1].split('"')[0]
            except (OSError, IndexError):
                time.sleep(0.2)
        raise AssertionError(log.read_text())

    try:
        b0 = build()
        src.write_text(src.read_text() + "\nthis is not python\n")
        time.sleep(7)
        assert build() == b0 and "doesn't load" in log.read_text()
        src.write_text(src.read_text().replace("\nthis is not python\n", "\n# new code\n"))
        time.sleep(7)
        assert build() != b0 and "restarting on the new code" in log.read_text()
        assert p.poll() is None  # same process (exec), still serving
    finally:
        p.kill()


def test_other_model_jobs_show_with_their_stage(tmp_path):
    # a python process whose argv names a project script counts, whatever it really runs
    p = __import__("subprocess").Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "scripts/fake_job.py", "--x", "1"],
        cwd=ROOT,
    )
    try:
        for _ in range(50):
            jobs = [j for j in dashboard.work() if j["pid"] == p.pid]
            if jobs:
                break
            __import__("time").sleep(0.1)
        assert jobs and jobs[0]["what"] == "Running fake_job.py"
        assert jobs[0]["cmd"] == "scripts/fake_job.py --x 1"
    finally:
        p.kill()
    cfg = {"imitate": {"expert_episodes": 10, "betas": [0.5, 0.2, 0.0], "dagger_iters": 2}}
    run = tmp_path / "2026-10-20_x_imitate"
    run.mkdir()
    assert dashboard.imitate_progress(run, cfg)["step"] == 0  # collecting expert examples
    (run / "expert_eps.pkl").write_bytes(b"")
    assert "studying" in dashboard.imitate_progress(run, cfg)["stage"]
    (run / "imitate.json").write_text(json.dumps([{"round": "bc", "success": 0.2}]))
    pr = dashboard.imitate_progress(run, cfg)
    assert "round 1 of 2" in pr["stage"] and (pr["step"], pr["steps"]) == (2, 4)
    assert pr["rounds"][0]["success"] == 0.2


def test_queued_launcher_is_listed_until_its_file_exists(tmp_path):
    import subprocess
    import time

    sh = ROOT / "runs/_logs" / f"launch_test{os.getpid()}.sh"
    flag = tmp_path / "done"
    sh.parent.mkdir(parents=True, exist_ok=True)
    sh.write_text(
        f"until [ -f {flag} ]; do sleep 0.2; done\n"
        "for s in 0 1; do echo --config configs/obstacle_h5.yaml > /dev/null; done\n"  # never trains
    )
    p = subprocess.Popen(["bash", str(sh.relative_to(ROOT))], cwd=ROOT)
    try:
        time.sleep(0.3)
        mine = [j for j in dashboard.queued() if j["pid"] == p.pid]
        assert mine and "obstacle_h5 (2 seeds)" in mine[0]["what"] and mine[0]["queued"]
        flag.write_text("")
        assert not [j for j in dashboard.queued() if j["pid"] == p.pid]
    finally:
        p.kill()
        sh.unlink()
