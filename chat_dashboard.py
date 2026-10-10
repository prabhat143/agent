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
            "title": "Instruction queued",
            "status": "warning",
            "detail": "Your instruction will be applied before the next reasoning step.",
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
                children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
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
<title>PTA Agent Workspace</title>
<style>
:root{
  color-scheme:dark;
  --bg:#0b0d11;--surface:#101319;--surface2:#151922;--surface3:#1a1f29;
  --line:#242a35;--line2:#303846;--text:#eef2f7;--muted:#8e98a8;
  --accent:#8b7cff;--accent2:#64b5ff;--ok:#43d49b;--warn:#f4bd62;--bad:#ff6f7d;
  --shadow:0 16px 44px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
html,body{margin:0;width:100%;height:100%;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body{overflow:hidden}
button,input,textarea{font:inherit}
button{cursor:pointer}
.app{display:grid;grid-template-columns:260px minmax(0,1fr) 350px;width:100%;height:100dvh;min-height:0;overflow:hidden}

/* Explorer */
.explorer{display:flex;flex-direction:column;min-width:0;min-height:0;background:#0e1116;border-right:1px solid var(--line)}
.explorerTop{padding:14px 13px 11px;border-bottom:1px solid var(--line);background:#0f1218}
.eyebrow{font-size:10px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#697485}
.workspaceTitle{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:7px}
.workspaceTitle strong{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.iconBtn{width:28px;height:28px;display:grid;place-items:center;background:transparent;border:1px solid transparent;border-radius:8px;color:#9da7b7}
.iconBtn:hover{background:var(--surface2);border-color:var(--line)}
.rootPath{font-size:10px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fileSearch{padding:9px 10px;border-bottom:1px solid var(--line)}
.fileSearch input{width:100%;height:31px;border:1px solid var(--line);border-radius:8px;background:#0b0e13;color:var(--text);outline:0;padding:0 10px;font-size:11px}
.fileSearch input:focus{border-color:#4b5568}
.tree{flex:1;min-height:0;overflow:auto;padding:7px 5px 18px;font-size:12px;user-select:none;scrollbar-gutter:stable}
.treeSection{padding:7px 9px 4px;color:#667184;font-size:9px;font-weight:800;letter-spacing:.11em;text-transform:uppercase}
.treeRow{height:28px;display:flex;align-items:center;gap:5px;padding-right:6px;border-radius:7px;cursor:pointer;white-space:nowrap;min-width:max-content;color:#cbd2dc}
.treeRow:hover{background:#151922}.treeRow.active{background:#202631;color:#fff}.treeRow.current{box-shadow:inset 2px 0 0 var(--accent2)}
.indent{display:inline-block;flex:0 0 auto}.twisty{width:14px;text-align:center;color:#798495;font-size:9px}.fileIcon{width:16px;text-align:center;color:#8792a3}.treeName{overflow:hidden;text-overflow:ellipsis;max-width:185px}.treeEmpty{padding:20px 10px;color:var(--muted);font-size:11px;text-align:center}

/* Main */
.main{display:flex;flex-direction:column;min-width:0;min-height:0;background:var(--bg);border-right:1px solid var(--line)}
.topbar{height:60px;flex:0 0 60px;display:flex;align-items:center;justify-content:space-between;gap:14px;padding:0 18px;border-bottom:1px solid var(--line);background:rgba(11,13,17,.96)}
.brandWrap{min-width:0}.brand{font-size:14px;font-weight:800}.taskLine{font-size:11px;color:var(--muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:650px}
.topActions{display:flex;align-items:center;gap:8px}.providerChip{font-size:10px;color:#b9c4d3;border:1px solid var(--line);background:#10141b;padding:6px 8px;border-radius:999px}.status{font-size:10px;font-weight:850;padding:6px 9px;border:1px solid rgba(100,181,255,.26);border-radius:999px;color:var(--accent2);background:rgba(100,181,255,.08)}
.tabs{height:38px;flex:0 0 38px;display:flex;align-items:end;padding:0 12px;border-bottom:1px solid var(--line);background:#0e1116;overflow-x:auto}
.tab{height:33px;display:flex;align-items:center;gap:7px;padding:0 11px;border:0;border-bottom:2px solid transparent;background:transparent;color:#8f99a8;font-size:11px;white-space:nowrap}.tab.active{color:#eef2f7;border-bottom-color:var(--accent)}.tabClose{opacity:.6;font-size:12px}.tabClose:hover{opacity:1}
.content{position:relative;flex:1 1 auto;min-height:0;overflow:hidden}
.messages{position:absolute;inset:0;overflow-y:auto;overflow-x:hidden;padding:24px 28px 30px;scrollbar-gutter:stable}
.msg{max-width:820px;margin:0 auto 19px;display:grid;grid-template-columns:28px minmax(0,1fr);gap:11px}.avatar{width:28px;height:28px;border-radius:9px;display:grid;place-items:center;background:#202630;color:#dfe6f0;font-size:10px;font-weight:800}.msg.user .avatar{background:#2b2545;color:#ddd5ff}.bubble{min-width:0}.who{font-size:11px;font-weight:800;margin:1px 0 6px}.body{font-size:13px;line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere;color:#dfe5ed}.msg.user .body{color:#f0f2f6}.msg.system .body{color:#b4becc}.meta{font-size:9px;color:#667184;margin-top:5px}
.preview{position:absolute;inset:0;display:none;flex-direction:column;background:#0d1015}.preview.open{display:flex}.previewMeta{height:34px;flex:0 0 34px;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 13px;border-bottom:1px solid var(--line);background:#10141a}.breadcrumb{font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#9ba8bb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.previewBadge{font-size:9px;color:#7d899a}.code{flex:1;min-height:0;overflow:auto;margin:0;padding:18px 20px 30px;background:#0d1015;color:#d6dde8;font:12px/1.62 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;white-space:pre;tab-size:4;scrollbar-gutter:stable}
.composerWrap{flex:0 0 auto;padding:12px 18px 16px;background:linear-gradient(180deg,rgba(11,13,17,.3),#0b0d11 32%)}
.composer{max-width:860px;margin:auto;background:#12161d;border:1px solid #2e3541;border-radius:16px;padding:10px 10px 9px;box-shadow:var(--shadow)}
textarea{width:100%;min-height:54px;max-height:150px;resize:vertical;background:transparent;border:0;outline:0;color:var(--text);padding:5px 7px;font-size:13px;line-height:1.5}.composerFoot{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:4px 3px 0}.hint{font-size:10px;color:#707b8c}.buttons{display:flex;gap:7px}.send,.stop{border-radius:9px;padding:7px 11px;font-size:11px;font-weight:800}.send{border:0;background:#eef1f5;color:#101216}.send:hover{background:#fff}.stop{background:#22171b;color:#ff9ca6;border:1px solid #4d2931}.stop:hover{background:#2b1b20}

/* Activity */
.activity{display:flex;flex-direction:column;min-width:0;min-height:0;background:#0e1116}
.activityTop{padding:14px;border-bottom:1px solid var(--line)}.activityHeader{display:flex;align-items:center;justify-content:space-between;gap:10px}.activityHeader strong{font-size:13px}.liveDot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent2);margin-right:6px;box-shadow:0 0 0 4px rgba(100,181,255,.08)}
.summaryGrid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:12px}.summary{min-width:0;padding:9px;background:#11151c;border:1px solid var(--line);border-radius:9px}.summary span{display:block;font-size:9px;color:#697486;text-transform:uppercase;letter-spacing:.06em}.summary b{display:block;margin-top:3px;font-size:11px;color:#dce3ec;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.summary.wide{grid-column:1/-1}
.progressHead{display:flex;align-items:center;justify-content:space-between;margin-top:11px;font-size:9px;color:#788394}.progress{height:5px;background:#1b2029;border-radius:999px;overflow:hidden;margin-top:6px}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:.25s}
.filterBar{display:flex;gap:5px;padding:9px 11px;border-bottom:1px solid var(--line)}.filterBtn{border:1px solid transparent;background:transparent;color:#7f8998;border-radius:7px;padding:5px 8px;font-size:10px}.filterBtn.active{background:#181d25;border-color:#2b323e;color:#dfe5ee}
.timeline{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;padding:10px 10px 18px;scrollbar-gutter:stable}.event{position:relative;padding:10px 10px 10px 29px;border-radius:10px;margin-bottom:5px}.event:hover{background:#12161d}.event:before{content:"";position:absolute;left:12px;top:15px;width:7px;height:7px;border-radius:50%;background:#697486}.event:after{content:"";position:absolute;left:15px;top:23px;bottom:-7px;width:1px;background:#252b35}.event:last-child:after{display:none}.event.success:before{background:var(--ok)}.event.error:before{background:var(--bad)}.event.warning:before{background:var(--warn)}.event.running:before{background:var(--accent2)}
.eventTitle{font-size:11px;font-weight:750;line-height:1.4;color:#dce2eb}.eventMeta{font-size:9px;color:#687384;margin-top:4px}.cwd{font:9px ui-monospace,SFMono-Regular,Menlo,monospace;color:#91a8d4;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.event details{margin-top:6px}.event summary{font-size:9px;color:#778395;cursor:pointer;list-style:none}.event summary::-webkit-details-marker{display:none}.eventDetail{font-size:10px;color:#9ca7b6;line-height:1.45;margin-top:6px;white-space:pre-wrap;max-height:170px;overflow:auto;border-left:2px solid #252c37;padding-left:8px}.empty{padding:28px 12px;text-align:center;color:#6e7989;font-size:11px}

::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-thumb{background:#2e3541;border-radius:999px;border:2px solid transparent;background-clip:padding-box}::-webkit-scrollbar-track{background:transparent}
@media(max-width:1180px){.app{grid-template-columns:225px minmax(0,1fr) 310px}.treeName{max-width:155px}}
@media(max-width:900px){body{overflow:auto}.app{display:flex;flex-direction:column;height:auto;min-height:100dvh;overflow:visible}.explorer{height:300px;flex:0 0 300px;border-right:0;border-bottom:1px solid var(--line)}.main{height:100dvh;min-height:650px;border-right:0}.activity{height:520px;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<div class="app">
  <aside class="explorer">
    <div class="explorerTop">
      <div class="eyebrow">Explorer</div>
      <div class="workspaceTitle"><strong>Workspace</strong><button id="refreshFiles" class="iconBtn" title="Refresh files">↻</button></div>
      <div id="explorerRoot" class="rootPath">Loading…</div>
    </div>
    <div class="fileSearch"><input id="fileFilter" placeholder="Filter files…" /></div>
    <div id="tree" class="tree"><div class="treeEmpty">Loading workspace…</div></div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div class="brandWrap"><div class="brand">PTA Autonomous Developer</div><div id="taskLine" class="taskLine">Waiting for task…</div></div>
      <div class="topActions"><span id="providerChip" class="providerChip">ollama</span><span id="status" class="status">CONNECTING</span></div>
    </div>
    <div id="tabs" class="tabs"><button class="tab active" data-tab="chat">Chat</button></div>
    <section class="content">
      <div id="messages" class="messages"></div>
      <div id="preview" class="preview">
        <div class="previewMeta"><div id="previewPath" class="breadcrumb"></div><div id="previewBadge" class="previewBadge">read-only</div></div>
        <pre id="code" class="code"></pre>
      </div>
    </section>
    <div class="composerWrap">
      <div class="composer">
        <textarea id="input" placeholder="Guide the agent, correct it, or ask it to change direction…"></textarea>
        <div class="composerFoot"><div class="hint">Enter to send · Shift+Enter for a new line</div><div class="buttons"><button id="stop" class="stop">Stop</button><button id="send" class="send">Send</button></div></div>
      </div>
    </div>
  </main>

  <aside class="activity">
    <div class="activityTop">
      <div class="activityHeader"><strong><span class="liveDot"></span>Live activity</strong><span id="stepText" style="font-size:10px;color:#788394">Waiting</span></div>
      <div class="summaryGrid">
        <div class="summary"><span>Provider</span><b id="provider">—</b></div>
        <div class="summary"><span>Status</span><b id="statusText">—</b></div>
        <div class="summary wide"><span>Current directory</span><b id="currentDir">.</b></div>
      </div>
      <div class="progressHead"><span>Run progress</span><span id="progressPct">0%</span></div>
      <div class="progress"><div id="bar" class="bar"></div></div>
    </div>
    <div class="filterBar"><button class="filterBtn active" data-filter="all">All</button><button class="filterBtn" data-filter="error">Errors</button><button class="filterBtn" data-filter="action">Actions</button></div>
    <div id="timeline" class="timeline"><div class="empty">No activity yet.</div></div>
  </aside>
</div>
<script>
const seen=new Set();
const expanded=new Set();
let selectedFile=null, currentTree=[], currentFilter='all', maxSteps=0, lastStep=0;
const messages=document.getElementById('messages');
const timeline=document.getElementById('timeline');
const input=document.getElementById('input');

function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
function nearBottom(el,t=90){return el.scrollHeight-el.scrollTop-el.clientHeight<t}
function basename(p){const parts=String(p||'').split('/');return parts[parts.length-1]||p}
function extIcon(name){
  const n=name.toLowerCase();
  if(n.endsWith('.java'))return 'J'; if(n.endsWith('.xml'))return '<>'; if(n.endsWith('.json'))return '{}';
  if(n.endsWith('.js')||n.endsWith('.ts'))return 'JS'; if(n.endsWith('.py'))return 'Py'; if(n.endsWith('.md'))return 'M';
  if(n.endsWith('.properties')||n.endsWith('.yml')||n.endsWith('.yaml'))return '⚙'; return '·';
}
function isCurrentPath(path){const cwd=document.getElementById('currentDir').textContent||'.';return cwd!=='.'&&(path===cwd||path.startsWith(cwd+'/'))}

function updateProgress(e){
  if(e.max_steps)maxSteps=e.max_steps;
  if(e.step)lastStep=Math.max(lastStep,e.step);
  if(e.cwd)document.getElementById('currentDir').textContent=e.cwd;
  if(maxSteps){const pct=Math.min(100,Math.round((lastStep/maxSteps)*100));document.getElementById('bar').style.width=pct+'%';document.getElementById('progressPct').textContent=pct+'%';document.getElementById('stepText').textContent=lastStep?`Step ${lastStep} / ${maxSteps}`:'Planning';}
  if(e.kind==='finish'&&e.status==='success'){document.getElementById('bar').style.width='100%';document.getElementById('progressPct').textContent='100%';document.getElementById('stepText').textContent='Completed';}
}

function addChat(e){
  const follow=nearBottom(messages);
  const wrap=document.createElement('div');
  wrap.className='msg '+(e.role==='user'?'user':e.kind==='chat'?'assistant':'system');
  const who=e.role==='user'?'You':'Agent';
  wrap.innerHTML=`<div class="avatar">${e.role==='user'?'Y':'AI'}</div><div class="bubble"><div class="who">${who}</div><div class="body">${esc(e.detail||e.title||'')}</div><div class="meta">${esc(e.timestamp||'')}</div></div>`;
  messages.appendChild(wrap); if(follow)messages.scrollTo({top:messages.scrollHeight,behavior:'smooth'});
}

function eventMatches(e){if(currentFilter==='all')return true;if(currentFilter==='error')return e.status==='error'||e.status==='warning';if(currentFilter==='action')return ['action','result','loop','recovery','web_search'].includes(e.kind);return true}
function renderTimelineFromState(events){timeline.innerHTML='';const filtered=events.filter(eventMatches);if(!filtered.length){timeline.innerHTML='<div class="empty">Nothing to show for this filter.</div>';return;}filtered.forEach(renderTimelineEvent)}
function renderTimelineEvent(e){
  const el=document.createElement('div');el.className='event '+(e.status||'info');
  const detail=e.detail?`<details><summary>Show details</summary><div class="eventDetail">${esc(e.detail)}</div></details>`:'';
  el.innerHTML=`<div class="eventTitle">${esc(e.title||e.kind)}</div>${e.cwd?`<div class="cwd">${esc(e.cwd)}</div>`:''}<div class="eventMeta">${e.step?`Step ${e.step} · `:''}${esc(e.timestamp||'')}</div>${detail}`;
  timeline.appendChild(el);
}
function addTimeline(e){if(e.kind==='chat'||!eventMatches(e))return;const follow=nearBottom(timeline);if(timeline.querySelector('.empty'))timeline.innerHTML='';renderTimelineEvent(e);if(follow)timeline.scrollTo({top:timeline.scrollHeight,behavior:'smooth'});}
function addEvent(e){if(seen.has(e.id))return;seen.add(e.id);updateProgress(e);if(e.kind==='chat'||e.kind==='guidance'||e.kind==='finish')addChat(e);addTimeline(e);if(['write_file','make_directory'].includes(e.action)||e.kind==='workspace')scheduleTreeRefresh();}

function paint(s){
  document.getElementById('provider').textContent=s.provider||'—';
  document.getElementById('providerChip').textContent=s.provider||'—';
  document.getElementById('currentDir').textContent=s.current_directory||'.';
  document.getElementById('taskLine').textContent=s.task||'No task';
  document.getElementById('explorerRoot').textContent=s.workspace||'Workspace';
  const st=document.getElementById('status');const label=(s.status||'running').toUpperCase();st.textContent=label;document.getElementById('statusText').textContent=label;
  if(s.status==='completed'){st.style.color='#43d49b';st.style.borderColor='rgba(67,212,155,.28)'}
  if(s.status==='failed'){st.style.color='#ff6f7d';st.style.borderColor='rgba(255,111,125,.28)'}
  (s.events||[]).forEach(addEvent);
}

function renderTree(nodes,depth=0,filter=''){
  let html='';
  for(const n of nodes){
    const matches=!filter||n.name.toLowerCase().includes(filter)||n.path.toLowerCase().includes(filter);
    const childMatch=n.type==='directory'&&containsMatch(n.children||[],filter);
    if(filter&&!matches&&!childMatch)continue;
    if(n.type==='directory'){
      const open=filter?true:expanded.has(n.path);
      html+=`<div class="treeRow ${isCurrentPath(n.path)?'current':''}" data-kind="dir" data-path="${esc(n.path)}"><span class="indent" style="width:${depth*14}px"></span><span class="twisty">${open?'▼':'▶'}</span><span class="fileIcon">▰</span><span class="treeName">${esc(n.name)}</span></div>`;
      if(open)html+=renderTree(n.children||[],depth+1,filter);
    }else{
      html+=`<div class="treeRow ${selectedFile===n.path?'active':''} ${isCurrentPath(n.path)?'current':''}" data-kind="file" data-path="${esc(n.path)}"><span class="indent" style="width:${depth*14}px"></span><span class="twisty"></span><span class="fileIcon">${esc(extIcon(n.name))}</span><span class="treeName">${esc(n.name)}</span></div>`;
    }
  }
  return html;
}
function containsMatch(nodes,filter){if(!filter)return true;for(const n of nodes){if(n.name.toLowerCase().includes(filter)||n.path.toLowerCase().includes(filter))return true;if(n.children&&containsMatch(n.children,filter))return true}return false}
function bindTree(){document.querySelectorAll('.treeRow').forEach(row=>row.onclick=()=>{const path=row.dataset.path;if(row.dataset.kind==='dir'){expanded.has(path)?expanded.delete(path):expanded.add(path);drawTree()}else openFile(path)});}
function drawTree(){const q=document.getElementById('fileFilter').value.trim().toLowerCase();const tree=document.getElementById('tree');tree.innerHTML='<div class="treeSection">Files</div>'+renderTree(currentTree,0,q);if(!currentTree.length)tree.innerHTML='<div class="treeEmpty">Workspace is empty.</div>';bindTree();}
async function refreshTree(){try{const r=await fetch('/api/files/tree');if(!r.ok)throw new Error(await r.text());const data=await r.json();currentTree=data.tree||[];document.getElementById('explorerRoot').textContent=data.workspace||'Workspace';drawTree()}catch(e){document.getElementById('tree').innerHTML=`<div class="treeEmpty">${esc(e.message)}</div>`}}
let treeTimer=null;function scheduleTreeRefresh(){clearTimeout(treeTimer);treeTimer=setTimeout(refreshTree,450)}

function activateTab(name){document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));document.getElementById('preview').classList.toggle('open',name!=='chat');}
function addFileTab(path){
  let tab=document.querySelector(`.tab[data-tab="file:${CSS.escape(path)}"]`);
  if(!tab){tab=document.createElement('button');tab.className='tab';tab.dataset.tab='file:'+path;tab.innerHTML=`${esc(basename(path))}<span class="tabClose">×</span>`;document.getElementById('tabs').appendChild(tab);tab.onclick=(ev)=>{if(ev.target.classList.contains('tabClose')){ev.stopPropagation();tab.remove();selectedFile=null;activateTab('chat');drawTree();return}openFile(path)}}
}
async function openFile(path){selectedFile=path;drawTree();addFileTab(path);activateTab('file:'+path);document.getElementById('previewPath').textContent=path;document.getElementById('code').textContent='Loading…';try{const r=await fetch('/api/files/content?path='+encodeURIComponent(path));if(!r.ok)throw new Error(await r.text());const data=await r.json();document.getElementById('code').textContent=data.content||'';document.getElementById('previewBadge').textContent=data.truncated?'read-only · truncated':'read-only'}catch(e){document.getElementById('code').textContent='Could not load file: '+e.message}}
document.querySelector('.tab[data-tab="chat"]').onclick=()=>activateTab('chat');

document.getElementById('fileFilter').addEventListener('input',drawTree);
document.getElementById('refreshFiles').onclick=refreshTree;
document.querySelectorAll('.filterBtn').forEach(btn=>btn.onclick=async()=>{currentFilter=btn.dataset.filter;document.querySelectorAll('.filterBtn').forEach(b=>b.classList.toggle('active',b===btn));const s=await fetch('/api/state').then(r=>r.json());renderTimelineFromState(s.events||[])});

async function send(){const text=input.value.trim();if(!text)return;const b=document.getElementById('send');b.disabled=true;try{const r=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});if(!r.ok)throw new Error(await r.text());input.value=''}catch(e){alert('Could not send instruction: '+e.message)}finally{b.disabled=false;input.focus()}}
document.getElementById('send').onclick=send;input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
document.getElementById('stop').onclick=async()=>{if(confirm('Stop the agent after its current operation?'))await fetch('/api/stop',{method:'POST'})};

fetch('/api/state').then(r=>r.json()).then(paint);refreshTree();setInterval(refreshTree,3000);
const stream=new EventSource('/events');stream.onmessage=m=>addEvent(JSON.parse(m.data));stream.onerror=()=>document.getElementById('status').textContent='RECONNECTING';
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

    @app.get("/api/files/tree")
    def files_tree() -> Response:
        try:
            return jsonify({"workspace": store.workspace, "tree": store.workspace_tree()})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.get("/api/files/content")
    def file_content() -> Response:
        relative_path = str(request.args.get("path", "")).strip()
        if not relative_path:
            return jsonify({"error": "path is required"}), 400
        try:
            return jsonify(store.read_workspace_file(relative_path))
        except FileNotFoundError:
            return jsonify({"error": "file not found"}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

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
