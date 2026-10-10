from __future__ import annotations

import json
import logging
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
            "detail": "Your guidance will be applied before the next reasoning step.",
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
<title>PTA Autonomous Developer</title>
<style>
:root{
  color-scheme:dark;--bg:#07101d;--bg2:#0a1322;--panel:#0d1726;--panel2:#101c2d;--panel3:#132137;
  --line:#22324a;--line2:#2e4362;--text:#f4f7fb;--muted:#91a0b7;--blue:#2d8cff;--blue2:#5eb8ff;
  --purple:#8a5cf6;--green:#3dd598;--orange:#f4ad47;--red:#ff6577;--shadow:0 12px 35px rgba(0,0,0,.26)
}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}body{overflow:hidden}
button,input,textarea,select{font:inherit}.app{height:100dvh;display:grid;grid-template-rows:76px minmax(0,1fr);background:linear-gradient(180deg,#08111f 0,#07101d 100%)}
.topbar{display:flex;align-items:center;gap:18px;padding:0 22px;border-bottom:1px solid var(--line);background:rgba(8,17,31,.96)}
.brandBlock{display:flex;align-items:center;min-width:360px;gap:12px}.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--purple),#6f3df0);display:grid;place-items:center;font-weight:900;box-shadow:0 0 26px rgba(138,92,246,.28)}.brand{font-size:17px;font-weight:800}.tagline{font-size:11px;color:var(--muted);margin-top:2px}.topSpacer{flex:1}
.topControl{height:38px;border:1px solid var(--line);background:#0b1625;color:#dfe7f3;border-radius:9px;padding:0 13px;display:flex;align-items:center;gap:7px;font-size:12px}.topControl b{color:#fff}.statusPill{color:var(--green);font-weight:800}.statusDot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 10px rgba(61,213,152,.5)}
.primary{background:linear-gradient(180deg,#258cff,#1978e8);border:1px solid #429eff;color:#fff;border-radius:9px;height:38px;padding:0 15px;font-weight:750;cursor:pointer}.secondary{background:#0f1a2a;border:1px solid var(--line);color:#e4eaf3;border-radius:9px;height:38px;padding:0 14px;font-weight:700;cursor:pointer}.iconBtn{width:38px;height:38px;border-radius:9px;border:1px solid var(--line);background:#0f1a2a;color:#bac7d9;cursor:pointer}
.workspaceGrid{min-height:0;display:grid;grid-template-columns:390px minmax(0,1fr) 400px}.left,.right,.center{min-height:0;overflow:hidden}.left{border-right:1px solid var(--line);background:#091321;display:flex;flex-direction:column}.center{border-right:1px solid var(--line);background:#08111e;display:flex;flex-direction:column}.right{background:#091321;display:flex;flex-direction:column}
.leftTop{padding:14px;border-bottom:1px solid var(--line)}.sectionTitle{font-size:12px;font-weight:800;letter-spacing:.04em;color:#eaf0f8;margin-bottom:10px}.projectRow{display:flex;gap:8px}.projectSelect{flex:1;min-width:0;border:1px solid var(--line2);background:#0d1a2a;border-radius:9px;padding:10px 12px;color:#eef3fb}.miniBtn{width:38px;border-radius:8px;border:1px solid var(--line2);background:#122137;color:#c9d4e5;cursor:pointer}
.navList{padding:10px 12px;border-bottom:1px solid var(--line)}.navItem{height:36px;border-radius:7px;padding:0 12px;display:flex;align-items:center;gap:10px;color:#b5c1d3;font-size:13px}.navItem.active{background:linear-gradient(90deg,rgba(45,140,255,.18),rgba(45,140,255,.08));color:#64b2ff;border-left:2px solid var(--blue)}
.filesHead{display:flex;align-items:center;justify-content:space-between;padding:13px 15px 8px}.filesTitle{font-size:12px;font-weight:800;color:#dbe4f1}.filesActions{display:flex;gap:8px}.smallIcon{border:0;background:transparent;color:#8fa1b8;cursor:pointer;font-size:15px}.filterWrap{padding:0 12px 10px}.filter{width:100%;height:36px;border-radius:8px;border:1px solid var(--line);background:#0a1524;color:#e8eef7;padding:0 11px;outline:none}.filter::placeholder{color:#687b95}.tree{flex:1;min-height:0;overflow:auto;padding:3px 7px 18px;font-size:12px}.treeRow{height:28px;display:flex;align-items:center;gap:6px;border-radius:6px;cursor:pointer;white-space:nowrap;min-width:max-content;color:#cbd5e5}.treeRow:hover{background:#101e31}.treeRow.active{background:linear-gradient(90deg,#153f78,#102d52);color:#fff}.indent{display:inline-block;flex:0 0 auto}.twisty{width:13px;text-align:center;color:#8292aa;font-size:10px}.fileIcon{width:16px;text-align:center;color:#7ea8ff}.javaIcon{color:#d36bff}.treeName{max-width:260px;overflow:hidden;text-overflow:ellipsis}.treeEmpty{padding:18px;color:var(--muted)}
.tabs{height:48px;display:flex;align-items:stretch;border-bottom:1px solid var(--line);background:#091321;overflow-x:auto;overflow-y:hidden}.tab{display:flex;align-items:center;gap:8px;min-width:max-content;padding:0 18px;border-right:1px solid var(--line);color:#9fb0c6;font-size:12px;cursor:pointer;position:relative}.tab.active{color:#fff;background:#0c1726}.tab.active:before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:var(--blue)}.tab .x{color:#6c7f99}.chatTab{min-width:74px;justify-content:center}.plusTab{width:44px;justify-content:center}
.breadcrumb{height:40px;display:flex;align-items:center;padding:0 14px;border-bottom:1px solid var(--line);font-size:11px;color:#8295ae;background:#0b1523}.crumb{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.readOnly{margin-left:auto;border:1px solid var(--line);background:#101d2f;border-radius:7px;padding:5px 8px;color:#aab8cb}
.mainViewport{flex:1;min-height:0;position:relative;overflow:hidden}.chatPane,.codePane{position:absolute;inset:0;overflow:auto}.chatPane{padding:28px 24px 130px}.chatPane.hidden,.codePane.hidden{display:none}.codePane{background:#08111e}.codeWrap{display:grid;grid-template-columns:48px minmax(0,1fr);min-height:100%;font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}.lineNums{padding:18px 8px;text-align:right;color:#4d617c;border-right:1px solid #17263a;white-space:pre;user-select:none;background:#091421}.code{padding:18px 20px;margin:0;white-space:pre;overflow:auto;color:#e1e8f4}.chatMsg{max-width:820px;margin:0 auto 20px;display:grid;grid-template-columns:36px minmax(0,1fr);gap:12px}.avatar{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font-size:12px;font-weight:800;background:#18253a;color:#e8eff9}.chatMsg.user .avatar{background:linear-gradient(135deg,#6d48f3,#8e60ff)}.msgCard{border:1px solid var(--line);background:#0d1828;border-radius:12px;padding:13px 15px;line-height:1.55;font-size:13px;color:#dde5f1}.chatMsg.user .msgCard{background:linear-gradient(135deg,rgba(92,64,190,.48),rgba(53,39,116,.5));border-color:#4b3c85}.who{font-size:11px;color:#8fa0b6;font-weight:750;margin-bottom:7px}.msgTime{font-size:9px;color:#60738e;margin-top:8px}.planCard{margin-top:10px;padding-top:10px;border-top:1px solid #22334c}
.composerWrap{position:absolute;left:50%;transform:translateX(-50%);bottom:16px;width:min(860px,calc(100% - 42px));z-index:8}.composer{background:#0d1828;border:1px solid #2c3d57;border-radius:13px;box-shadow:var(--shadow);padding:11px}.composer textarea{width:100%;min-height:58px;max-height:150px;resize:none;border:0;outline:0;background:transparent;color:#eef4fb;padding:4px;font-size:13px}.composerActions{display:flex;align-items:center;justify-content:space-between;margin-top:7px}.composerMeta{font-size:10px;color:#70839d}.sendRow{display:flex;align-items:center;gap:8px}.mode{border:1px solid var(--line);background:#101d2f;color:#b9c7d8;border-radius:8px;height:34px;padding:0 10px}.stopBtn{height:34px;border-radius:8px;border:1px solid #633240;background:#301a23;color:#ff9bac;font-weight:750;padding:0 12px;cursor:pointer}.sendBtn{height:34px;border-radius:8px;border:1px solid #439cff;background:#2489ff;color:#fff;font-weight:800;padding:0 15px;cursor:pointer}
.rightTabs{height:48px;border-bottom:1px solid var(--line);display:flex;background:#091321}.rTab{height:48px;padding:0 17px;display:flex;align-items:center;color:#8fa0b7;font-size:12px;cursor:pointer;position:relative}.rTab.active{color:#62b2ff}.rTab.active:after{content:"";position:absolute;left:12px;right:12px;bottom:0;height:2px;background:#3c9aff}.rightBody{flex:1;min-height:0;overflow:hidden;position:relative}.activityPane,.terminalPane,.logsPane{position:absolute;inset:0;overflow:auto}.activityPane.hidden,.terminalPane.hidden,.logsPane.hidden{display:none}.activityTop{padding:14px;border-bottom:1px solid var(--line)}.runCard{border:1px solid var(--line);background:#0d1928;border-radius:10px;padding:12px}.runLine{display:flex;align-items:center;gap:8px}.runText{font-size:13px;font-weight:800;color:var(--green)}.runStep{margin-left:auto;font-size:10px;color:#8799b0}.progress{height:5px;border-radius:999px;background:#17243a;overflow:hidden;margin-top:10px}.bar{height:100%;width:0;background:linear-gradient(90deg,#29d4b3,#28b7ff);transition:.3s}.cwdBox{margin-top:11px;padding-top:10px;border-top:1px solid #20314a;font-size:10px;color:#73869f}.cwdValue{color:#53abff;font:11px ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:4px;overflow-wrap:anywhere}
.filters{display:flex;gap:7px;padding:12px 14px 7px}.filterBtn{border:1px solid var(--line);background:#0e1a2a;color:#8295ae;border-radius:8px;padding:6px 10px;font-size:10px;cursor:pointer}.filterBtn.active{background:#187ee8;color:#fff;border-color:#278fff}.timeline{padding:5px 14px 18px}.event{display:grid;grid-template-columns:24px minmax(0,1fr);gap:9px;padding:8px 0;position:relative}.event:not(:last-child):before{content:"";position:absolute;left:11px;top:28px;bottom:-4px;width:1px;background:#1e314a}.eventIcon{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;font-size:10px;background:#23354d;color:#a8b8cb;z-index:1}.event.success .eventIcon{background:#173c35;color:#55e4b8}.event.error .eventIcon{background:#44202a;color:#ff8292}.event.warning .eventIcon{background:#43351d;color:#ffca6f}.event.running .eventIcon{background:#173c66;color:#62b6ff}.eventTitle{font-size:11px;font-weight:750;color:#e5ebf5}.eventDetail{font-size:10px;color:#788ba3;line-height:1.45;margin-top:3px;white-space:pre-wrap;max-height:90px;overflow:auto}.eventMeta{font-size:9px;color:#4f6580;margin-top:4px}.terminalPane,.logsPane{padding:14px;background:#07101c}.terminalHeader{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.terminalTitle{font-size:12px;font-weight:800}.terminalOut,.logsOut{border:1px solid var(--line);background:#050b13;border-radius:8px;padding:12px;min-height:240px;color:#9ed0ff;font:10px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;overflow:auto}.logsOut{color:#a9b8ca}
::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-thumb{background:#263a55;border-radius:999px;border:2px solid transparent;background-clip:padding-box}::-webkit-scrollbar-track{background:transparent}
@media(max-width:1350px){.workspaceGrid{grid-template-columns:320px minmax(0,1fr) 350px}.brandBlock{min-width:300px}}@media(max-width:1050px){.workspaceGrid{grid-template-columns:280px minmax(0,1fr)}.right{display:none}.topControl:nth-of-type(2){display:none}}@media(max-width:760px){body{overflow:auto}.app{height:auto;min-height:100dvh}.topbar{flex-wrap:wrap;height:auto;padding:12px}.brandBlock{min-width:0}.workspaceGrid{display:flex;flex-direction:column}.left{height:360px}.center{height:780px}.right{display:flex;height:620px}.topControl,.secondary,.iconBtn{display:none}}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brandBlock"><div class="logo">AI</div><div><div class="brand">PTA Autonomous Developer</div><div class="tagline">Build, test and ship with AI</div></div></div>
    <div class="topSpacer"></div>
    <div class="topControl">Provider: <b id="providerTop">—</b></div>
    <div class="topControl">Model: <b id="modelTop">local</b> ▾</div>
    <div class="topControl statusPill"><span class="statusDot"></span><span id="statusTop">RUNNING</span></div>
    <button class="primary" title="Project creation wiring comes next">＋ New Project</button>
    <button class="secondary" title="Project opening wiring comes next">Open Project</button>
    <button class="iconBtn" title="Settings">⚙</button>
  </header>

  <div class="workspaceGrid">
    <aside class="left">
      <div class="leftTop"><div class="sectionTitle">Projects</div><div class="projectRow"><select id="projectSelect" class="projectSelect"><option>workspace</option></select><button class="miniBtn">＋</button></div></div>
      <div class="navList"><div class="navItem active">▣ Workspace</div><div class="navItem">⌕ Search</div><div class="navItem">⑂ Git <span style="color:#5d708a;font-size:10px">(soon)</span></div></div>
      <div class="filesHead"><div class="filesTitle">Files</div><div class="filesActions"><button id="refreshFiles" class="smallIcon">↻</button><button class="smallIcon">▽</button></div></div>
      <div class="filterWrap"><input id="fileFilter" class="filter" placeholder="Search files..." /></div>
      <div id="tree" class="tree"><div class="treeEmpty">Loading workspace...</div></div>
    </aside>

    <main class="center">
      <div id="tabs" class="tabs"><div class="tab chatTab active" data-chat="1">Chat</div><div class="tab plusTab">＋</div></div>
      <div id="breadcrumb" class="breadcrumb"><span class="crumb">Chat with the agent while it builds</span></div>
      <div class="mainViewport">
        <div id="chatPane" class="chatPane"></div>
        <div id="codePane" class="codePane hidden"><div class="codeWrap"><div id="lineNums" class="lineNums"></div><pre id="code" class="code"></pre></div></div>
        <div class="composerWrap"><div class="composer"><textarea id="input" placeholder="Type a message to the agent..."></textarea><div class="composerActions"><div class="composerMeta">Your guidance is applied before the next reasoning step.</div><div class="sendRow"><select class="mode"><option>Auto</option></select><button id="stop" class="stopBtn">Stop</button><button id="send" class="sendBtn">Send</button></div></div></div></div>
      </div>
    </main>

    <aside class="right">
      <div class="rightTabs"><div class="rTab active" data-pane="activity">Live Activity</div><div class="rTab" data-pane="terminal">Terminal</div><div class="rTab" data-pane="logs">Logs</div></div>
      <div class="rightBody">
        <div id="activityPane" class="activityPane">
          <div class="activityTop"><div class="runCard"><div class="runLine"><span class="statusDot"></span><span id="runStatus" class="runText">Running</span><span id="runStep" class="runStep">Waiting...</span></div><div class="progress"><div id="bar" class="bar"></div></div><div class="cwdBox">Current directory<div id="currentDir" class="cwdValue">.</div></div></div></div>
          <div class="filters"><button class="filterBtn active" data-filter="all">All</button><button class="filterBtn" data-filter="action">Actions</button><button class="filterBtn" data-filter="error">Errors</button><button class="filterBtn" data-filter="web_search">Search</button></div>
          <div id="timeline" class="timeline"></div>
        </div>
        <div id="terminalPane" class="terminalPane hidden"><div class="terminalHeader"><div class="terminalTitle">Latest Command Output</div><button id="clearTerminal" class="filterBtn">Clear</button></div><pre id="terminalOut" class="terminalOut">No command output yet.</pre></div>
        <div id="logsPane" class="logsPane hidden"><div class="terminalHeader"><div class="terminalTitle">Agent Logs</div></div><pre id="logsOut" class="logsOut">Waiting for events...</pre></div>
      </div>
    </aside>
  </div>
</div>
<script>
const seen=new Set();
const chatPane=document.getElementById('chatPane'),tree=document.getElementById('tree'),timeline=document.getElementById('timeline');
const codePane=document.getElementById('codePane'),code=document.getElementById('code'),lineNums=document.getElementById('lineNums'),input=document.getElementById('input');
const tabs=document.getElementById('tabs'),breadcrumb=document.getElementById('breadcrumb');
let maxSteps=0,lastStep=0,currentFilter='all',latestCommandOutput='No command output yet.',allEvents=[],treeData=[],openFiles=[],activeFile=null;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
function nearBottom(el,t=90){return el.scrollHeight-el.scrollTop-el.clientHeight<t}
function statusSymbol(s,k){if(s==='success')return '✓';if(s==='error')return '!';if(s==='warning')return '↻';if(k==='web_search')return '⌕';if(k==='thinking')return '◇';return '▶'}
function updateProgress(e){if(e.max_steps)maxSteps=e.max_steps;if(e.step)lastStep=Math.max(lastStep,e.step);if(e.cwd)document.getElementById('currentDir').textContent=e.cwd;if(maxSteps){document.getElementById('bar').style.width=Math.min(100,(lastStep/maxSteps)*100)+'%';document.getElementById('runStep').textContent=`Step ${lastStep} of ${maxSteps}`}if(e.kind==='finish'){document.getElementById('bar').style.width=e.status==='success'?'100%':document.getElementById('bar').style.width;document.getElementById('runStatus').textContent=e.status==='success'?'Completed':'Stopped'}}
function addChat(e){const follow=nearBottom(chatPane);const wrap=document.createElement('div');wrap.className='chatMsg '+(e.role==='user'?'user':'assistant');const who=e.role==='user'?'You':'Agent';wrap.innerHTML=`<div class="avatar">${e.role==='user'?'Y':'AI'}</div><div><div class="who">${who}</div><div class="msgCard">${esc(e.detail||e.title||'')}</div><div class="msgTime">${esc(e.timestamp||'')}</div></div>`;chatPane.appendChild(wrap);if(follow)chatPane.scrollTo({top:chatPane.scrollHeight,behavior:'smooth'})}
function eventMatches(e){if(currentFilter==='all')return true;if(currentFilter==='error')return e.status==='error';if(currentFilter==='action')return ['action','result','workspace','finish','guidance'].includes(e.kind);return e.kind===currentFilter||e.action===currentFilter}
function renderTimeline(){timeline.innerHTML='';allEvents.filter(eventMatches).forEach(e=>{if(e.kind==='chat')return;const el=document.createElement('div');el.className='event '+(e.status||'info');el.innerHTML=`<div class="eventIcon">${statusSymbol(e.status,e.kind)}</div><div><div class="eventTitle">${esc(e.title||e.kind)}</div>${e.detail?`<div class="eventDetail">${esc(e.detail)}</div>`:''}<div class="eventMeta">${e.step?`Step ${e.step} · `:''}${e.cwd?`cwd: ${esc(e.cwd)} · `:''}${esc(e.timestamp||'')}</div></div>`;timeline.appendChild(el)});timeline.scrollTop=timeline.scrollHeight}
function addEvent(e){if(seen.has(e.id))return;seen.add(e.id);allEvents.push(e);updateProgress(e);if(e.kind==='chat'||e.kind==='guidance'||e.kind==='finish')addChat(e);if(e.action==='run_command'&&e.kind==='result'){latestCommandOutput=e.detail||'';document.getElementById('terminalOut').textContent=latestCommandOutput}document.getElementById('logsOut').textContent=allEvents.map(x=>`[${x.timestamp||''}] ${x.title||x.kind}${x.detail?`\n${x.detail}`:''}`).join('\n\n');renderTimeline();if(['write_file','make_directory'].includes(e.action)&&e.kind==='result')loadFiles(false)}
function paint(s){document.getElementById('providerTop').textContent=s.provider||'—';document.getElementById('statusTop').textContent=(s.status||'running').toUpperCase();document.getElementById('runStatus').textContent=(s.status||'running').replace(/^./,c=>c.toUpperCase());document.getElementById('currentDir').textContent=s.current_directory||'.';(s.events||[]).forEach(addEvent)}
function fileIcon(name){if(name.endsWith('.java'))return '<span class="fileIcon javaIcon">J</span>';if(name.endsWith('.xml'))return '<span class="fileIcon">◇</span>';if(name.endsWith('.properties')||name.endsWith('.yml')||name.endsWith('.yaml'))return '<span class="fileIcon">⚙</span>';return '<span class="fileIcon">·</span>'}
function projectNameFromTree(nodes){const first=(nodes||[]).find(n=>n.type==='directory');return first?first.name:'workspace'}
function renderProjectSelect(nodes){const sel=document.getElementById('projectSelect');const dirs=(nodes||[]).filter(n=>n.type==='directory');sel.innerHTML=dirs.length?dirs.map(d=>`<option>${esc(d.name)}</option>`).join(''):'<option>workspace</option>'}
function renderTree(nodes,filter=''){tree.innerHTML='';const term=filter.trim().toLowerCase();function walk(list,depth){for(const n of list){const childMatch=n.type==='directory'&&n.children?.some(c=>JSON.stringify(c).toLowerCase().includes(term));const match=!term||n.name.toLowerCase().includes(term)||childMatch;if(!match)continue;const row=document.createElement('div');row.className='treeRow'+(activeFile===n.path?' active':'');row.dataset.path=n.path;row.dataset.type=n.type;row.innerHTML=`<span class="indent" style="width:${depth*15}px"></span>${n.type==='directory'?`<span class="twisty">${term?'▾':'▸'}</span><span class="fileIcon">▰</span>`:fileIcon(n.name)}<span class="treeName">${esc(n.name)}</span>`;tree.appendChild(row);if(n.type==='directory'){let expanded=term?true:row.dataset.expanded==='1';row.onclick=()=>{expanded=!expanded;row.dataset.expanded=expanded?'1':'0';row.querySelector('.twisty').textContent=expanded?'▾':'▸';let p=row.nextSibling;while(p&&Number(p.dataset.depth||0)>depth){p.style.display=expanded?'flex':'none';p=p.nextSibling}};const startCount=tree.children.length;walk(n.children||[],depth+1);for(let i=startCount;i<tree.children.length;i++){tree.children[i].dataset.depth=depth+1;tree.children[i].style.display=term?'flex':'none'}}else row.onclick=()=>openFile(n.path)}}walk(nodes,0);if(!tree.children.length)tree.innerHTML='<div class="treeEmpty">No matching files.</div>'}
async function loadFiles(showLoading=true){if(showLoading)tree.innerHTML='<div class="treeEmpty">Loading workspace...</div>';try{const r=await fetch('/api/files');const data=await r.json();treeData=data.files||[];renderProjectSelect(treeData);renderTree(treeData,document.getElementById('fileFilter').value)}catch(e){tree.innerHTML='<div class="treeEmpty">Could not load workspace.</div>'}}
function renderTabs(){tabs.innerHTML='<div class="tab chatTab" data-chat="1">Chat</div>';for(const f of openFiles){const t=document.createElement('div');t.className='tab'+(activeFile===f.path?' active':'');t.innerHTML=`<span class="javaIcon">J</span>${esc(f.name)} <span class="x">×</span>`;t.onclick=(ev)=>{if(ev.target.classList.contains('x')){ev.stopPropagation();closeFile(f.path)}else showFile(f.path)};tabs.appendChild(t)}const plus=document.createElement('div');plus.className='tab plusTab';plus.textContent='＋';tabs.appendChild(plus);const chat=tabs.querySelector('[data-chat="1"]');if(!activeFile)chat.classList.add('active');chat.onclick=showChat}
function showChat(){activeFile=null;chatPane.classList.remove('hidden');codePane.classList.add('hidden');breadcrumb.innerHTML='<span class="crumb">Chat with the agent while it builds</span>';renderTabs();renderTree(treeData,document.getElementById('fileFilter').value)}
async function openFile(path){if(!openFiles.find(f=>f.path===path))openFiles.push({path,name:path.split('/').pop()});await showFile(path)}
async function showFile(path){activeFile=path;chatPane.classList.add('hidden');codePane.classList.remove('hidden');code.textContent='Loading...';lineNums.textContent='';renderTabs();renderTree(treeData,document.getElementById('fileFilter').value);try{const r=await fetch('/api/file?path='+encodeURIComponent(path));const data=await r.json();if(!r.ok)throw new Error(data.error||'Unable to read file');code.textContent=data.content||'';const lines=(data.content||'').split('\n').length;lineNums.textContent=Array.from({length:lines},(_,i)=>i+1).join('\n');breadcrumb.innerHTML=`<span class="crumb">${esc(path.split('/').join(' › '))}</span><span class="readOnly">Read only</span>`}catch(e){code.textContent='Unable to preview file: '+e.message}}
function closeFile(path){openFiles=openFiles.filter(f=>f.path!==path);if(activeFile===path){if(openFiles.length)showFile(openFiles[openFiles.length-1].path);else showChat()}else renderTabs()}
async function send(){const text=input.value.trim();if(!text)return;document.getElementById('send').disabled=true;try{const r=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});if(!r.ok)throw new Error(await r.text());input.value='';showChat()}catch(e){alert('Could not send instruction: '+e.message)}finally{document.getElementById('send').disabled=false;input.focus()}}
document.getElementById('send').onclick=send;input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});document.getElementById('stop').onclick=async()=>{if(confirm('Stop the agent after the current operation?'))await fetch('/api/stop',{method:'POST'})};document.getElementById('refreshFiles').onclick=()=>loadFiles();document.getElementById('fileFilter').oninput=e=>renderTree(treeData,e.target.value);document.getElementById('clearTerminal').onclick=()=>{latestCommandOutput='';document.getElementById('terminalOut').textContent=''};
document.querySelectorAll('.filterBtn[data-filter]').forEach(b=>b.onclick=()=>{document.querySelectorAll('.filterBtn[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');currentFilter=b.dataset.filter;renderTimeline()});document.querySelectorAll('.rTab').forEach(t=>t.onclick=()=>{document.querySelectorAll('.rTab').forEach(x=>x.classList.remove('active'));t.classList.add('active');['activity','terminal','logs'].forEach(p=>document.getElementById(p+'Pane').classList.toggle('hidden',p!==t.dataset.pane))});
renderTabs();fetch('/api/state').then(r=>r.json()).then(paint);loadFiles();setInterval(()=>loadFiles(false),3000);const stream=new EventSource('/events');stream.onmessage=m=>addEvent(JSON.parse(m.data));stream.onerror=()=>document.getElementById('statusTop').textContent='RECONNECTING';
</script>
</body>
</html>
"""


def create_app(store: ChatProgressStore) -> Flask:
    app = Flask(__name__)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/state")
    def state() -> Response:
        return jsonify(store.snapshot())

    @app.get("/api/files")
    def files() -> Response:
        try:
            return jsonify({"files": store.workspace_tree(), "workspace": store.workspace})
        except Exception as exc:
            return jsonify({"error": str(exc), "files": []}), 500

    @app.get("/api/file")
    def file_preview() -> Response:
        path = str(request.args.get("path", "")).strip()
        if not path:
            return jsonify({"error": "path is required"}), 400
        try:
            return jsonify(store.read_workspace_file(path))
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


def start_chat_dashboard(store: ChatProgressStore, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> threading.Thread:
    app = create_app(store)

    def run() -> None:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=run, name="pta-agent-chat-dashboard", daemon=True)
    thread.start()
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    return thread
