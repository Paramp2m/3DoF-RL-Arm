import json
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
        return e.code, b""


def test_state_lists_phases_runs_docs_and_unchanged_poll_is_tiny(url):
    code, body = get(url + "/api/state")
    s = json.loads(body)
    assert code == 200 and len(s["phase"]["list"]) == 8
    assert {"runs", "machine", "decisions", "docs", "experiments"} <= s.keys()
    assert all(dashboard.NAME.match(r["id"]) for r in s["runs"])
    assert json.loads(get(url + "/api/state?v=" + s["v"])[1]) == {"v": s["v"], "same": True}


def test_page_and_bundled_markdown_renderer_are_served(url):
    assert get(url + "/")[0] == 200
    assert get(url + "/marked.min.js")[0] == 200


def test_only_project_docs_and_media_are_readable(url):
    assert get(url + "/api/doc?path=CLAUDE.md")[0] == 200
    assert get(url + "/api/doc?path=../../../etc/passwd")[0] == 404
    assert get(url + "/api/doc?path=pyproject.toml")[0] == 404
    assert get(url + "/file/runs/../pyproject.toml")[0] == 404
    assert get(url + "/file/docs/media/../../CLAUDE.md")[0] == 404


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
