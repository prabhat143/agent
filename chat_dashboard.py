from __future__ import annotations

import json
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context


IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", "target", ".idea", ".vscode"}


class ChatProgressStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._instructions: list[str] = []
        self._next_id = 1
        self._stop_requested = False
        self.task = ""
        self.provider = ""
        self.workspace = ""
        self.current_directory = "."
        self.status = "idle"

    def set_context(self, task: str, provider: str, workspace: str) -> None:
        with self._lock:
            self.task = task
            self.provider = provider
            self.workspace = workspace
            self.current_directory = "."
            self.status = "running"
        self.publish({"kind": "chat", "role": "user", "title": "You", "status": "info", "detail": task})

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            payload = dict(event)
            payload["id"] = self._next_id
            payload["timestamp"] = datetime.now().strftime("%H:%M:%S")
            self._next_id += 1
            self._events.append(payload)
            if payload.get("cwd"):
                self.current_directory = str(payload["cwd"])
            if payload.get("kind") == "finish":
                self.status = "completed" if payload.get("status") == "success" else "failed"

    def add_instruction(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        with self._lock:
            self._instructions.append(cleaned)
        self.publish({"kind": "chat", "role": "user", "title": "You", "status": "info", "detail": cleaned})
        self.publish({
            "kind": "guidance",
            "title": "Instruction queued for the agent",
            "status": "warning",
            "detail": "This will be applied before the next AI reasoning step.",
        })

    def drain_instructions(self) -> list[str]:
        with self._lock:
            items = list(self._instructions)
            self._instructions.clear()
            return items

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True
        self.publish({
            "kind": "control",
            "title": "Stop requested",
            "status": "warning",
            "detail": "The agent will stop safely before its next step.",
        })

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task": self.task,
                "provider": self.provider,
                "workspace": self.workspace,
                "current_directory": self.current_directory,
                "status": self.status,
                "events": list(self._events),
                "queued_instructions": len(self._instructions),
                "stop_requested": self._stop_requested,
            }

    def events_after(self, event_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [event for event in self._events if int(event["id"]) > event_id]

    def _workspace_root(self) -> Path:
        if not self.workspace:
            raise ValueError("Workspace is not configured yet.")
        return Path(self.workspace).expanduser().resolve()

    def _resolve_workspace_path(self, relative_path: str) -> Path:
        root = self._workspace_root()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Path escapes the workspace.") from exc
        return candidate

    def workspace_tree(self) -> list[dict[str, Any]]:
        root = self._workspace_root()
        if not root.exists():
            return []

        def build(directory: Path, depth: int = 0) -> list[dict[str, Any]]:
            if depth > 12:
                return []
            nodes: list[dict[str, Any]] = []
            try:
                children = sorted(
                    directory.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except (OSError, PermissionError):
                return nodes
            for child in children:
                if child.name in IGNORED_DIRS or child.name.startswith(".DS_Store"):
                    continue
                rel = child.relative_to(root).as_posix()
                if child.is_dir():
                    nodes.append({
                        "name": child.name,
                        "path": rel,
                        "type": "directory",
                        "children": build(child, depth + 1),
                    })
                elif child.is_file():
                    try:
                        size = child.stat().st_size
                    except OSError:
                        size = 0
                    nodes.append({"name": child.name, "path": rel, "type": "file", "size": size})
            return nodes

        return build(root)

    def read_workspace_file(self, relative_path: str) -> dict[str, Any]:
        path = self._resolve_workspace_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        size = path.stat().st_size
        if size > 500_000:
            return {"path": relative_path, "content": "File is too large to preview (>500 KB).", "truncated": True}
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"path": relative_path, "content": "Binary/non-UTF-8 file. Preview unavailable.", "truncated": True}
        truncated = len(content) > 150_000
        if truncated:
            content = content[:150_000] + "\n\n... preview truncated ..."
        return {"path": relative_path, "content": content, "truncated": truncated}


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>PTA Agent</title>
<style>
:root{color-scheme:dark;--bg:#0c0d10;--panel:#15171c;--panel2:#1d2026;--line:#2b2f38;--text:#f3f5f7;--muted:#9aa3b2;--accent:#8b5cf6;--ok:#41c98e;--warn:#f4bd50;--bad:#ff6675;--blue:#62b5ff}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}body{overflow:hidden}
.app{display:grid;grid-template-columns:280px minmax(0,1fr) 390px;width:100%;height:100dvh;min-height:0;overflow:hidden}
.explorer{background:#0f1115;border-right:1px solid var(--line);display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}.explorerHead{height:64px;flex:0 0 64px;padding:13px 12px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:8px}.explorerTitle{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.07em}.explorerRoot{font-size:10px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:205px}.refresh{padding:6px 8px;background:var(--panel2);color:#ccd3dd;border:1px solid var(--line);border-radius:7px;font-size:11px}.tree{flex:1;min-height:0;overflow:auto;padding:8px 5px 20px;font-size:12px;user-select:none}.treeRow{height:26px;display:flex;align-items:center;gap:5px;border-radius:6px;padding-right:5px;cursor:pointer;white-space:nowrap;min-width:max-content}.treeRow:hover{background:#191c22}.treeRow.active{background:#232834;color:#fff}.indent{display:inline-block;flex:0 0 auto}.twisty{width:14px;text-align:center;color:#7e8795;font-size:10px}.fileIcon{width:15px;text-align:center;color:#9aa3b2}.treeName{overflow:hidden;text-overflow:ellipsis;max-width:210px}.treeEmpty{padding:18px 10px;color:var(--muted);font-size:11px}
.chat{display:flex;flex-direction:column;min-width:0;min-height:0;border-right:1px solid var(--line);overflow:hidden}.top{height:64px;flex:0 0 64px;display:flex;align-items:center;justify-content:space-between;padding:0 22px;border-bottom:1px solid var(--line);background:rgba(12,13,16,.94)}.brand{font-weight:750}.sub{font-size:12px;color:var(--muted);margin-top:3px}.status{font-size:11px;font-weight:800;padding:6px 9px;border:1px solid var(--line);border-radius:999px;color:var(--blue)}
.mainArea{position:relative;flex:1 1 auto;min-height:0;overflow:hidden}.messages{position:absolute;inset:0;overflow-y:auto;overflow-x:hidden;padding:26px 22px 28px;overscroll-behavior:contain;scrollbar-gutter:stable}.msg{max-width:780px;margin:0 auto 18px;display:flex;gap:12px}.avatar{width:28px;height:28px;border-radius:7px;display:grid;place-items:center;flex:0 0 auto;background:var(--panel2);font-size:12px;font-weight:800}.msg.user .avatar{background:#2b2440}.bubble{min-width:0;flex:1}.who{font-size:12px;font-weight:750;margin-bottom:6px}.body{font-size:14px;line-height:1.6;white-space:pre-wrap;color:#e7eaf0;overflow-wrap:anywhere}.msg.system .body{color:#c3cad5;font-size:13px}.meta{font-size:10px;color:var(--muted);margin-top:6px}
.preview{position:absolute;inset:0;background:#0c0d10;display:none;flex-direction:column;z-index:4}.preview.open{display:flex}.previewHead{height:46px;flex:0 0 46px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 14px;gap:12px}.previewPath{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#cbd6ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.closePreview{background:#22262d;color:#d7dce4;border:1px solid #343a45;border-radius:7px;padding:6px 9px;font-size:11px}.code{flex:1;min-height:0;overflow:auto;margin:0;padding:16px 18px;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;white-space:pre;tab-size:4;color:#d9dee7;background:#101216}
.composerWrap{position:relative;flex:0 0 auto;padding:14px 22px 20px;background:linear-gradient(180deg,rgba(12,13,16,.35),var(--bg) 28%);z-index:5}.composer{max-width:820px;margin:auto;background:var(--panel);border:1px solid #363b46;border-radius:18px;padding:12px 12px 10px;box-shadow:0 14px 35px rgba(0,0,0,.35)}textarea{width:100%;min-height:58px;max-height:150px;resize:vertical;background:transparent;border:0;outline:0;color:var(--text);font:inherit;padding:4px 6px}.actions{display:flex;justify-content:space-between;align-items:center;gap:10px}.hint{font-size:11px;color:var(--muted)}button{border:0;border-radius:10px;padding:9px 13px;font-weight:750;cursor:pointer}.send{background:#f0f2f5;color:#111}.stop{background:#2a1b1f;color:#ff9aa5;border:1px solid #58313a}.send:disabled{opacity:.5;cursor:not-allowed}
.side{background:#101216;display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}.sideTop{flex:0 0 auto;padding:18px;border-bottom:1px solid var(--line)}.sideTitle{font-weight:800;font-size:14px}.info{margin-top:12px;display:grid;gap:10px}.infoRow{font-size:11px;color:var(--muted)}.infoRow b{display:block;color:#dfe4ec;font-size:12px;margin-top:2px;font-weight:650;white-space:normal;overflow-wrap:anywhere}.path{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#b8c8ff!important}.progress{height:5px;background:#22262d;border-radius:999px;overflow:hidden;margin-top:14px}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--blue));transition:.25s}.stepText{font-size:10px;color:var(--muted);margin-top:6px}.timeline{flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;padding:16px 14px;overscroll-behavior:contain;scrollbar-gutter:stable}.event{position:relative;border-left:2px solid var(--line);padding:0 0 18px 14px;margin-left:6px}.event:last-child{padding-bottom:4px}.event:before{content:"";position:absolute;left:-6px;top:2px;width:10px;height:10px;border-radius:50%;background:#6f7785;box-shadow:0 0 0 3px #101216}.event.success:before{background:var(--ok)}.event.error:before{background:var(--bad)}.event.warning:before{background:var(--warn)}.event.running:before{background:var(--blue)}.et{font-size:12px;font-weight:700;line-height:1.35;overflow-wrap:anywhere}.ed{font-size:11px;color:var(--muted);margin-top:5px;white-space:pre-wrap;max-height:160px;overflow:auto;overflow-wrap:anywhere}.em{font-size:9px;color:#727b89;margin-top:5px}.cwd{font-size:10px;color:#9eb6ff;margin-top:5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.empty{font-size:12px;color:var(--muted);padding:22px 8px;text-align:center}
.messages::-webkit-scrollbar,.timeline::-webkit-scrollbar,.tree::-webkit-scrollbar,.code::-webkit-scrollbar,.ed::-webkit-scrollbar{width:10px;height:10px}.messages::-webkit-scrollbar-thumb,.timeline::-webkit-scrollbar-thumb,.tree::-webkit-scrollbar-thumb,.code::-webkit-scrollbar-thumb,.ed::-webkit-scrollbar-thumb{background:#343945;border-radius:999px;border:2px solid transparent;background-clip:padding-box}
@media(max-width:1150px){.app{grid-template-columns:230px minmax(0,1fr) 340px}}@media(max-width:900px){body{overflow:auto}.app{display:flex;flex-direction:column;height:auto;min-height:100dvh;overflow:visible}.explorer{height:300px;flex:0 0 300px;border-right:0;border-bottom:1px solid var(--line)}.chat{height:100dvh;min-height:620px;border-right:0}.side{display:flex;min-height:420px;max-height:70dvh;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<div class="app">
<aside class="explorer">
  <div class="explorerHead"><div><div class="explorerTitle">Workspace</div><div id="explorerRoot" class="explorerRoot">Loading...</div></div><button id="refreshFiles" class="refresh">Refresh</button></div>
  <div id="tree" class="tree"><div class="treeEmpty">Loading workspace...</div></div>
</aside>
<section class="chat">
  <div class="top"><div><div class="brand">PTA Autonomous Developer</div><div class="sub">Chat with the agent while it builds</div></div><div id="status" class="status">CONNECTING</div></div>
  <div class="mainArea"><div id="messages" class="messages"></div><div id="preview" class="preview"><div class="previewHead"><div id="previewPath" class="previewPath"></div><button id="closePreview" class="closePreview">Close</button></div><pre id="code" class="code"></pre></div></div>
  <div class="composerWrap"><div class="composer"><textarea id="input" placeholder="Tell the agent what to change, correct, or focus on..."></textarea><div class="actions"><div class="hint">Your message is applied before the next reasoning step.</div><div><button id="stop" class="stop">Stop</button> <button id="send" class="send">Send</button></div></div></div></div>
</section>
<aside class="side"><div class="sideTop"><div class="sideTitle">Live execution</div><div class="info"><div class="infoRow">Provider<b id="provider">—</b></div><div class="infoRow">Workspace root<b id="workspace" class="path">—</b></div><div class="infoRow">Current execution directory<b id="currentDir" class="path">.</b></div></div><div class="progress"><div id="bar" class="bar"></div></div><div id="stepText" class="stepText">Waiting...</div></div><div id="timeline" class="timeline"><div class="empty">Waiting for agent activity...</div></div></aside>
</div>
<script>
const seen=new Set(),messages=document.getElementById('messages'),timeline=document.getElementById('timeline'),input=document.getElementById('input'),tree=document.getElementById('tree');let maxSteps=0,lastStep=0,activeFile='',lastTreeSignature='';const expanded=new Set();
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}function nearBottom(el,threshold=90){return el.scrollHeight-el.scrollTop-el.clientHeight<threshold}
function updateProgress(e){if(e.max_steps)maxSteps=e.max_steps;if(e.step)lastStep=Math.max(lastStep,e.step);if(e.cwd)document.getElementById('currentDir').textContent=e.cwd;if(maxSteps){document.getElementById('bar').style.width=Math.min(100,(lastStep/maxSteps)*100)+'%';document.getElementById('stepText').textContent=`Step ${lastStep} of ${maxSteps}`}if(e.kind==='finish'&&e.status==='success'){document.getElementById('bar').style.width='100%';document.getElementById('stepText').textContent='Completed'}}
function addChat(e){const follow=nearBottom(messages);const wrap=document.createElement('div');wrap.className='msg '+(e.role==='user'?'user':e.kind==='chat'?'assistant':'system');const who=e.role==='user'?'You':'Agent';wrap.innerHTML=`<div class="avatar">${e.role==='user'?'Y':'AI'}</div><div class="bubble"><div class="who">${who}</div><div class="body">${esc(e.detail||e.title||'')}</div><div class="meta">${esc(e.timestamp||'')}</div></div>`;messages.appendChild(wrap);if(follow)messages.scrollTo({top:messages.scrollHeight,behavior:'smooth'})}
function addTimeline(e){if(e.kind==='chat')return;const follow=nearBottom(timeline);if(timeline.querySelector('.empty'))timeline.innerHTML='';const el=document.createElement('div');el.className='event '+(e.status||'info');el.innerHTML=`<div class="et">${esc(e.title||e.kind)}</div>${e.detail?`<div class="ed">${esc(e.detail)}</div>`:''}${e.cwd?`<div class="cwd">cwd: ${esc(e.cwd)}</div>`:''}<div class="em">${e.step?`Step ${e.step} · `:''}${esc(e.timestamp||'')}</div>`;timeline.appendChild(el);if(follow)timeline.scrollTo({top:timeline.scrollHeight,behavior:'smooth'})}
function addEvent(e){if(seen.has(e.id))return;seen.add(e.id);updateProgress(e);if(e.kind==='chat'||e.kind==='guidance'||e.kind==='finish')addChat(e);addTimeline(e);if(e.kind==='result'&&(e.action==='write_file'||e.action==='make_directory'))loadFiles(false)}
function paint(s){document.getElementById('provider').textContent=s.provider||'—';document.getElementById('workspace').textContent=s.workspace||'—';document.getElementById('explorerRoot').textContent=s.workspace||'Workspace';document.getElementById('currentDir').textContent=s.current_directory||'.';const st=document.getElementById('status');st.textContent=(s.status||'running').toUpperCase();(s.events||[]).forEach(addEvent)}
function iconFor(name){const ext=(name.split('.').pop()||'').toLowerCase();if(['java','kt'].includes(ext))return 'J';if(['xml','html'].includes(ext))return '<>';if(['json','yaml','yml','properties','toml'].includes(ext))return '{}';if(['py'].includes(ext))return 'Py';if(['md','txt'].includes(ext))return 'T';return '·'}
function renderNodes(nodes,depth=0){let html='';for(const n of nodes){if(n.type==='directory'){if(depth<2&&!expanded.has(n.path))expanded.add(n.path);const open=expanded.has(n.path);html+=`<div class="treeRow folder" data-path="${esc(n.path)}" style="padding-left:${depth*14+5}px"><span class="twisty">${open?'▼':'▶'}</span><span class="fileIcon">▰</span><span class="treeName">${esc(n.name)}</span></div>`;if(open)html+=renderNodes(n.children||[],depth+1)}else{html+=`<div class="treeRow file ${activeFile===n.path?'active':''}" data-path="${esc(n.path)}" style="padding-left:${depth*14+5}px"><span class="twisty"></span><span class="fileIcon">${esc(iconFor(n.name))}</span><span class="treeName">${esc(n.name)}</span></div>`}}return html}
async function loadFiles(force=true){try{const r=await fetch('/api/files',{cache:'no-store'});if(!r.ok)throw new Error(await r.text());const data=await r.json(),sig=JSON.stringify(data.tree);if(!force&&sig===lastTreeSignature)return;lastTreeSignature=sig;tree.innerHTML=data.tree.length?renderNodes(data.tree):'<div class="treeEmpty">Workspace is empty.</div>';bindTree()}catch(e){tree.innerHTML=`<div class="treeEmpty">Could not load workspace: ${esc(e.message)}</div>`}}
function bindTree(){tree.querySelectorAll('.folder').forEach(el=>el.onclick=()=>{const p=el.dataset.path;if(expanded.has(p))expanded.delete(p);else expanded.add(p);loadFiles(true)});tree.querySelectorAll('.file').forEach(el=>el.onclick=()=>openFile(el.dataset.path))}
async function openFile(path){try{const r=await fetch('/api/file?path='+encodeURIComponent(path),{cache:'no-store'});if(!r.ok)throw new Error(await r.text());const data=await r.json();activeFile=path;document.getElementById('previewPath').textContent=data.path;document.getElementById('code').textContent=data.content;document.getElementById('preview').classList.add('open');loadFiles(true)}catch(e){alert('Could not open file: '+e.message)}}
document.getElementById('closePreview').onclick=()=>{document.getElementById('preview').classList.remove('open');activeFile='';loadFiles(true)};document.getElementById('refreshFiles').onclick=()=>loadFiles(true);
async function send(){const text=input.value.trim();if(!text)return;document.getElementById('send').disabled=true;try{const r=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});if(!r.ok)throw new Error(await r.text());input.value=''}catch(e){alert('Could not send instruction: '+e.message)}finally{document.getElementById('send').disabled=false;input.focus()}}
document.getElementById('send').onclick=send;input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});document.getElementById('stop').onclick=async()=>{if(confirm('Stop the agent after the current operation?'))await fetch('/api/stop',{method:'POST'})};fetch('/api/state').then(r=>r.json()).then(s=>{paint(s);loadFiles(true)});const stream=new EventSource('/events');stream.onmessage=m=>addEvent(JSON.parse(m.data));stream.onerror=()=>document.getElementById('status').textContent='RECONNECTING';setInterval(()=>loadFiles(false),1800);
</script>
</body>
</html>
"""


def create_app(store: ChatProgressStore) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/state")
    def state() -> Response:
        return jsonify(store.snapshot())

    @app.get("/api/files")
    def files() -> Response:
        try:
            return jsonify({"tree": store.workspace_tree()})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/file")
    def file_preview() -> Response:
        relative_path = str(request.args.get("path", "")).strip()
        if not relative_path:
            return jsonify({"error": "path is required"}), 400
        try:
            return jsonify(store.read_workspace_file(relative_path))
        except FileNotFoundError:
            return jsonify({"error": "file not found"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/message")
    def message() -> Response:
        data = request.get_json(silent=True) or {}
        text = str(data.get("message", "")).strip()
        if not text:
            return jsonify({"error": "message is required"}), 400
        store.add_instruction(text)
        return jsonify({"ok": True})

    @app.post("/api/stop")
    def stop() -> Response:
        store.request_stop()
        return jsonify({"ok": True})

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
                time.sleep(0.35)

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def start_chat_dashboard(
    store: ChatProgressStore,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> threading.Thread:
    app = create_app(store)

    def run() -> None:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=run, name="pta-agent-chat-dashboard", daemon=True)
    thread.start()
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    return thread
