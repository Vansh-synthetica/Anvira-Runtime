"""Run a generated browser script in a fake browser (Node + stubs) and report what a real page would trip over.

A static check cannot see that ``startGameBtn`` is used but never declared, or that a click handler throws on the third tick.
Loading the script against a stub DOM/canvas/timers/localStorage, clicking the buttons, sending keys and touches, and advancing
the game clock does. Only used when ``node`` is on PATH; otherwise the caller falls back to the static checks.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import List

_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
process.on('unhandledRejection', () => {}); process.on('uncaughtException', () => {});
const code = fs.readFileSync(process.argv[2], 'utf8');
const html = fs.readFileSync(process.argv[3], 'utf8');
const out = { loadError: null, tickErrors: [], missingIds: [], draws: 0, ticks: 0, clicked: [], listeners: {}, touched: false, keys: false };
const ids = [...html.matchAll(/\bid\s*=\s*["']([\w-]+)["']/g)].map((m) => m[1]);
const handlers = { document: {}, window: {} };
const timers = [], frames = [];
function on(bucket, t, fn) { (bucket[t] = bucket[t] || []).push(fn); out.listeners[t] = (out.listeners[t] || 0) + 1; }
function mkCtx() {
  const ctx = new Proxy({}, { get(_, k) {
    if (k === 'canvas') return null;
    if (/^(fill|stroke|clear|draw|arc|rect|line|move|begin|close|save|restore|scale|translate|rotate|set|put|measure|create|quadratic|bezier|ellipse|round)/.test(String(k)) || k === 'fill') return (...a) => { out.draws++; return { addColorStop() {}, width: 10 }; };
    return undefined; }, set() { return true; } });
  return ctx;
}
function mkEl(id) {
  const store = { style: {}, dataset: {}, children: [], _h: {}, textContent: '', innerHTML: '', innerText: '', value: '', width: 600, height: 400, clientWidth: 600, clientHeight: 400,
    offsetWidth: 600, offsetHeight: 400, disabled: false, hidden: false, id };
  store.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } };
  store.addEventListener = (t, fn) => { (store._h[t] = store._h[t] || []).push(fn); out.listeners[t] = (out.listeners[t] || 0) + 1; };
  store.removeEventListener = () => {};
  store.getContext = () => store._ctx || (store._ctx = mkCtx());
  store.getBoundingClientRect = () => ({ left: 0, top: 0, width: 600, height: 400, right: 600, bottom: 400 });
  store.appendChild = (c) => { store.children.push(c); return c; };
  return new Proxy(store, { get(t, k) { if (k in t) return t[k]; if (typeof k === 'symbol') return undefined; return (...a) => undefined; }, set(t, k, v) { t[k] = v; return true; } });
}
const els = {};
ids.forEach((i) => { els[i] = mkEl(i); });
const document = { readyState: 'loading', body: mkEl('body'), documentElement: mkEl('html'),
  getElementById: (i) => { if (!els[i]) { out.missingIds.push(i); return null; } return els[i]; },
  querySelector: (s) => { const m = /^#([\w-]+)$/.exec(s); if (m) return document.getElementById(m[1]); return mkEl('q'); },
  querySelectorAll: (s) => { const m = /^#([\w-]+)$/.exec(s); const e = m ? els[m[1]] : null; return e ? [e] : []; },
  getElementsByClassName: () => [], createElement: () => mkEl('new'), addEventListener: (t, fn) => on(handlers.document, t, fn), removeEventListener() {} };
const store = {};
const localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; } };
const noop = () => new Proxy({ connect() {}, start() {}, stop() {}, disconnect() {}, buffer: null, gain: { value: 1, setValueAtTime() {}, exponentialRampToValueAtTime() {}, linearRampToValueAtTime() {} }, frequency: { value: 0, setValueAtTime() {} }, type: '' }, { get: (t, k) => (k in t ? t[k] : (typeof k === 'symbol' ? undefined : () => {})) });
class AudioCtx { constructor() { this.destination = {}; this.currentTime = 0; this.state = 'running'; this.sampleRate = 44100;
    return new Proxy(this, { get: (t, k) => (k in t ? t[k] : (typeof k === 'symbol' ? undefined : (...a) => noop())) }); }
  resume() { return Promise.resolve(); } }
class AudioEl { constructor() { this.currentTime = 0; this.volume = 1; } play() { return Promise.resolve(); } pause() {} load() {} addEventListener() {} }
const window = { document, localStorage, AudioContext: AudioCtx, webkitAudioContext: AudioCtx, Audio: AudioEl, innerWidth: 800, innerHeight: 600, devicePixelRatio: 1,
  addEventListener: (t, fn) => on(handlers.window, t, fn), removeEventListener() {}, requestAnimationFrame: (fn) => { frames.push(fn); return frames.length; },
  cancelAnimationFrame() {}, setInterval: (fn, ms) => { timers.push({ fn, ms }); return timers.length; }, clearInterval: (id) => { if (timers[id - 1]) timers[id - 1].dead = true; },
  setTimeout: (fn) => { timers.push({ fn, once: true }); return timers.length; }, clearTimeout: (id) => { if (timers[id - 1]) timers[id - 1].dead = true; },
  location: { reload() {}, href: 'http://localhost/', assign() {}, replace() {}, search: '', hash: '', pathname: '/' }, history: { pushState() {}, replaceState() {} }, screen: { width: 800, height: 600 },
  matchMedia: () => ({ matches: false, addEventListener() {} }), navigator: { vibrate() {}, userAgent: 'node' }, fetch: () => Promise.resolve({ ok: true, json: () => Promise.resolve({}), text: () => Promise.resolve(''), arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)), blob: () => Promise.resolve({}) }),
  console: { log() {}, warn() {}, error() {}, info() {} }, Math, JSON, Date, Number, String, Array, Object, Promise, Set, Map, parseInt, parseFloat, isNaN, Image: class { }, alert() {}, confirm: () => true, prompt: () => null };
window.window = window; window.self = window; window.globalThis = window;
Object.defineProperty(window, 'Audio', { value: AudioEl });
const sandbox = vm.createContext(window);
function fire(bucket, type, ev) { for (const fn of (bucket[type] || [])) { try { fn(ev); } catch (e) { out.tickErrors.push(`${type} handler: ${e && e.message}`); } } }
function keyEv(key) { return { key, code: /^[a-z]$/i.test(key) ? 'Key' + key.toUpperCase() : key, keyCode: 0, preventDefault() {}, stopPropagation() {}, target: document.body }; }
try { vm.runInContext(code, sandbox, { timeout: 3000, filename: 'game.js' }); } catch (e) { out.loadError = `${e && e.name}: ${e && e.message}`; }
if (!out.loadError) {
  document.readyState = 'complete';
  for (const t of ['DOMContentLoaded', 'load']) { fire(handlers.document, t, {}); fire(handlers.window, t, {}); }
  const startNames = Object.keys(els).filter((i) => /start|play|begin/i.test(i) && !/re-?start|resume|again|pause/i.test(i));
  const clickAll = (names, label) => { for (const b of names) for (const fn of (els[b]._h.click || [])) { try { fn({ preventDefault() {}, target: els[b] }); if (label === 'start') out.clicked.push(b); } catch (e) { out.tickErrors.push(`click ${b}: ${e && e.message}`); } } };
  const step = (n) => {
    let ran = 0;
    for (let i = 0; i < n; i++) {
      let any = false;
      for (const t of timers.slice()) { if (t.dead) continue; any = true; try { t.fn(); } catch (e) { out.tickErrors.push(`tick: ${e && e.name}: ${e && e.message}`); t.dead = true; } if (t.once) t.dead = true; }
      const fr = frames.splice(0); for (const fn of fr) { any = true; try { fn(i * 16); } catch (e) { out.tickErrors.push(`frame: ${e && e.name}: ${e && e.message}`); } }
      if (any) ran++; else break;
    }
    return ran;
  };
  // Phase 1: a player presses Start. Did a loop begin, and does it draw?
  const before = out.draws;
  clickAll(startNames, 'start');
  out.ticks = step(60);
  out.drawsAfterStart = out.draws - before;
  // Phase 2: everything else a player does. Only crashes matter here (Space may legitimately pause).
  const others = Object.keys(els).filter((i) => /btn|button|pause|restart|reset|resume|again/i.test(i) && !startNames.includes(i));
  clickAll(others, 'other'); clickAll(others.filter((i) => /pause/i.test(i)), 'other');
  for (const k of ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd', ' ', 'Enter']) { fire(handlers.document, 'keydown', keyEv(k)); fire(handlers.window, 'keydown', keyEv(k)); out.keys = true; }
  const touch = (x, y) => ({ touches: [{ clientX: x, clientY: y }], changedTouches: [{ clientX: x, clientY: y }], preventDefault() {}, target: els[Object.keys(els)[0]] });
  for (const t of ['touchstart', 'touchmove', 'touchend']) for (const bucket of [handlers.document, handlers.window, ...Object.values(els).map((e) => e._h)]) { if ((bucket[t] || []).length) out.touched = true; fire(bucket, t, touch(300, 200)); }
  clickAll(startNames, 'restart');
  step(300);
}
out.tickErrors = [...new Set(out.tickErrors)].slice(0, 5);
out.missingIds = [...new Set(out.missingIds)];
console.log(JSON.stringify(out));
"""


def available() -> bool:
    return shutil.which("node") is not None


def run(code: str, html: str, needs_canvas: bool = True) -> List[str]:
    """Problems found by executing ``code`` against a fake page built from ``html`` (empty list = it ran cleanly)."""
    node = shutil.which("node")
    if not node:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        cpath, hpath, hh = os.path.join(tmp, "game.js"), os.path.join(tmp, "page.html"), os.path.join(tmp, "h.js")
        for p, text in ((cpath, code), (hpath, html), (hh, _HARNESS)):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
        try:
            r = subprocess.run([node, hh, cpath, hpath], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return []
    try:
        res = json.loads((r.stdout.strip().splitlines() or ["{}"])[-1])
    except ValueError:
        return []
    problems: List[str] = []
    if res.get("loadError"):
        problems.append(f"Running the script in a browser-like page failed immediately with {res['loadError']}. "
                        "Declare every variable and function before you use it.")
    for e in res.get("tickErrors", []):
        problems.append(f"While playing (clicking buttons, pressing keys, advancing the game) it threw: {e}")
    if res.get("missingIds") and not res.get("loadError"):
        problems.append("It asked the page for ids that do not exist: " + ", ".join(res["missingIds"]) + ".")
    if needs_canvas and not res.get("loadError") and not res.get("tickErrors") and res.get("ticks", 0) > 0 and res.get("drawsAfterStart", 0) == 0:
        problems.append("After pressing Start and running the game clock nothing was ever drawn on the canvas.")
    if needs_canvas and not res.get("loadError") and res.get("ticks", 0) == 0:
        problems.append("Pressing Start never began the game loop (no setInterval/setTimeout/requestAnimationFrame ran).")
    return problems
