from __future__ import annotations

import json
import threading
import time
import webbrowser
from datetime import datetime
from typing import Any

from flask import Flask, Response, jsonify, stream_with_context


class ProgressStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._next_id = 1
        self.task = ""
        self.provider = ""
        self.workspace = ""
        self.status = "idle"

    def set_context(self, task: str, provider: str, workspace: str) -> None:
        with self._lock:
            self.task = task
            self.provider = provider
            self.workspace = workspace
            self.status = "running"

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            payload = dict(event)
            payload["id"] = self._next_id
            payload["timestamp"] = datetime.now().strftime("%H:%M:%S")
            self._next_id += 1
            self._events.append(payload)
            if payload.get("kind") == "finish":
                self.status = "completed" if payload.get("status") == "success" else "failed"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task": self.task,
                "provider": self.provider,
                "workspace": self.workspace,
                "status": self.status,
                "events": list(self._events),
            }

    def events_after(self, event_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [event for event in self._events if int(event["id"]) > event_id]


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>PTA Agent Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #11182a;
      --panel2: #151f35;
      --text: #edf2ff;
      --muted: #91a0bd;
      --line: #26344f;
      --accent: #7c9cff;
      --ok: #54d69a;
      --warn: #ffc857;
      --bad: #ff6b7a;
      --run: #70c2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, #172342 0, var(--bg) 38%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell { max-width: 1120px; margin: 0 auto; padding: 28px 20px 60px; }
    .header { display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 22px; }
    h1 { margin: 0 0 6px; font-size: 26px; }
    .subtitle { color: var(--muted); font-size: 14px; }
    .status {
      padding: 8px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;
      background: rgba(112,194,255,.13); color: var(--run); border: 1px solid rgba(112,194,255,.25);
    }
    .grid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 12px; margin-bottom: 18px; }
    .card { background: rgba(17,24,42,.92); border: 1px solid var(--line); border-radius: 15px; padding: 15px 16px; }
    .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 7px; }
    .value { font-size: 14px; word-break: break-word; }
    .progress-wrap { margin: 16px 0 22px; }
    .progress-head { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-bottom:8px; }
    .progress { height: 8px; border-radius: 999px; background:#0c1324; overflow:hidden; border:1px solid #202d46; }
    .bar { height:100%; width:0%; background: linear-gradient(90deg, #6f8cff, #6bd5ff); transition: width .35s ease; }
    .timeline { position: relative; margin-top: 8px; }
    .timeline:before { content:""; position:absolute; left:17px; top:8px; bottom:8px; width:2px; background:var(--line); }
    .event { position:relative; display:grid; grid-template-columns:36px 1fr; gap:12px; margin-bottom:14px; }
    .dot { width:14px; height:14px; margin:7px 0 0 11px; border-radius:50%; border:3px solid var(--bg); box-shadow:0 0 0 2px var(--line); background:var(--muted); z-index:2; }
    .event.success .dot { background:var(--ok); }
    .event.error .dot { background:var(--bad); }
    .event.warning .dot { background:var(--warn); }
    .event.running .dot { background:var(--run); animation:pulse 1.2s infinite; }
    @keyframes pulse { 0%,100%{box-shadow:0 0 0 2px var(--line),0 0 0 0 rgba(112,194,255,.35)} 50%{box-shadow:0 0 0 2px var(--line),0 0 0 8px rgba(112,194,255,0)} }
    .event-card { background:rgba(21,31,53,.9); border:1px solid var(--line); border-radius:13px; padding:12px 14px; }
    .event-top { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
    .event-title { font-size:14px; font-weight:700; }
    .meta { color:var(--muted); font-size:11px; white-space:nowrap; }
    .detail { color:#bdc8dc; font-size:12px; line-height:1.5; margin-top:8px; white-space:pre-wrap; max-height:220px; overflow:auto; }
    .chips { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
    .chip { font-size:10px; color:#aebbd1; border:1px solid #31415f; border-radius:999px; padding:3px 7px; }
    .empty { text-align:center; color:var(--muted); padding:38px 10px; border:1px dashed var(--line); border-radius:14px; }
    @media (max-width: 760px) { .grid { grid-template-columns:1fr; } .header { flex-direction:column; } }
  </style>
</head>
<body>
<div class="shell">
  <div class="header">
    <div>
      <h1>PTA Agent Monitor</h1>
      <div class="subtitle">Live view of what the autonomous developer is doing.</div>
    </div>
    <div id="status" class="status">CONNECTING</div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">Task</div><div id="task" class="value">—</div></div>
    <div class="card"><div class="label">Provider</div><div id="provider" class="value">—</div></div>
    <div class="card"><div class="label">Workspace</div><div id="workspace" class="value">—</div></div>
  </div>

  <div class="progress-wrap">
    <div class="progress-head"><span>Agent progress</span><span id="stepText">Waiting...</span></div>
    <div class="progress"><div id="bar" class="bar"></div></div>
  </div>

  <div id="timeline" class="timeline"><div class="empty">Waiting for agent activity...</div></div>
</div>
<script>
  const timeline = document.getElementById('timeline');
  const seen = new Set();
  let lastStep = 0;
  let maxSteps = 0;

  function esc(v) {
    return String(v ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
  }

  function updateProgress(event) {
    if (event.max_steps) maxSteps = event.max_steps;
    if (event.step) lastStep = Math.max(lastStep, event.step);
    if (maxSteps) {
      const pct = Math.min(100, Math.max(2, (lastStep / maxSteps) * 100));
      document.getElementById('bar').style.width = pct + '%';
      document.getElementById('stepText').textContent = lastStep ? `Step ${lastStep} of ${maxSteps}` : 'Planning';
    }
    if (event.kind === 'finish' && event.status === 'success') {
      document.getElementById('bar').style.width = '100%';
      document.getElementById('stepText').textContent = 'Completed';
    }
  }

  function addEvent(event) {
    if (seen.has(event.id)) return;
    seen.add(event.id);
    if (timeline.querySelector('.empty')) timeline.innerHTML = '';
    updateProgress(event);
    const el = document.createElement('div');
    el.className = `event ${event.status || 'info'}`;
    const chips = [event.kind, event.action, event.target].filter(Boolean)
      .map(v => `<span class="chip">${esc(v)}</span>`).join('');
    el.innerHTML = `
      <div class="dot"></div>
      <div class="event-card">
        <div class="event-top">
          <div class="event-title">${esc(event.title)}</div>
          <div class="meta">${event.step ? `Step ${event.step} · ` : ''}${esc(event.timestamp)}</div>
        </div>
        ${event.detail ? `<div class="detail">${esc(event.detail)}</div>` : ''}
        ${chips ? `<div class="chips">${chips}</div>` : ''}
      </div>`;
    timeline.appendChild(el);
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function paintState(state) {
    document.getElementById('task').textContent = state.task || '—';
    document.getElementById('provider').textContent = state.provider || '—';
    document.getElementById('workspace').textContent = state.workspace || '—';
    const s = document.getElementById('status');
    s.textContent = (state.status || 'running').toUpperCase();
    if (state.status === 'completed') { s.style.color = '#54d69a'; s.style.borderColor = 'rgba(84,214,154,.3)'; }
    if (state.status === 'failed') { s.style.color = '#ff6b7a'; s.style.borderColor = 'rgba(255,107,122,.3)'; }
    (state.events || []).forEach(addEvent);
  }

  fetch('/api/state').then(r => r.json()).then(paintState);
  const stream = new EventSource('/events');
  stream.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    addEvent(event);
    if (event.kind === 'finish') {
      fetch('/api/state').then(r => r.json()).then(paintState);
    }
  };
  stream.onerror = () => {
    document.getElementById('status').textContent = 'RECONNECTING';
  };
</script>
</body>
</html>
"""


def create_app(store: ProgressStore) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/state")
    def state() -> Response:
        return jsonify(store.snapshot())

    @app.get("/events")
    def events() -> Response:
        @stream_with_context
        def stream():
            last_id = 0
            while True:
                pending = store.events_after(last_id)
                if pending:
                    for event in pending:
                        last_id = int(event["id"])
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                else:
                    yield ": keep-alive\n\n"
                time.sleep(0.5)

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def start_dashboard(
    store: ProgressStore,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> threading.Thread:
    app = create_app(store)

    def run() -> None:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=run, name="pta-agent-dashboard", daemon=True)
    thread.start()
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    return thread
