"""Web server for the Car Care Agent chat. Runs locally or on Render.

Local:   python3 server.py          (opens http://localhost:8000)
Render:  start command `python server.py`; set GEMINI_API_KEY as an environment variable.
"""
import json
import os
import re
import threading
import time
import webbrowser
from collections import defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import requests

import agents
import llm

PORT = int(os.environ.get("PORT", "8000"))
HOSTED = bool(os.environ.get("RENDER") or os.environ.get("HOSTED"))
HOST = os.environ.get("HOST") or ("0.0.0.0" if HOSTED else "127.0.0.1")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

MAX_CONCURRENT_TURNS = int(os.environ.get("MAX_CONCURRENT_TURNS", "2"))
MSGS_PER_VISITOR_PER_MIN = int(os.environ.get("MSGS_PER_VISITOR_PER_MIN", "5"))
EVAL_TOKEN = os.environ.get("EVAL_TOKEN", "")  # optional: lets your eval run skip the per-visitor limit

SESSIONS = agents.Sessions()
_slots = threading.BoundedSemaphore(MAX_CONCURRENT_TURNS)
_recent, _recent_lock = defaultdict(deque), threading.Lock()
_llm = None
_llm_lock = threading.Lock()


def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = llm.Gemini()
        return _llm


def visitor_allowed(ip, now=None, limit=None):
    now, limit = now or time.time(), limit or MSGS_PER_VISITOR_PER_MIN
    with _recent_lock:
        q = _recent[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True


VPIC = "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/{make}/modelyear/{year}{vtype}"
_models_cache, _models_lock = {}, threading.Lock()


def lookup_models(make, year, get=None):
    """Model names for a make and year from NHTSA's vehicle database (cars, SUVs, trucks).

    Feeds the Model search box. Returns [] on any problem, so the visitor can still type a model.
    """
    get = get or requests.get
    make = str(make or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9 .&'-]{1,40}", make):
        return []
    try:
        year = int(year)
    except (TypeError, ValueError):
        return []
    if not 1995 <= year <= time.gmtime().tm_year + 1:
        return []
    key = (make.lower(), year)
    with _models_lock:
        if key in _models_cache:
            return _models_cache[key]
    names = set()
    for vtype in ("/vehicletype/car", "/vehicletype/multipurpose", "/vehicletype/truck"):
        try:
            r = get(VPIC.format(make=quote(make), year=year, vtype=vtype), params={"format": "json"}, timeout=8)
            r.raise_for_status()
            for row in r.json().get("Results") or []:
                name = str(row.get("Model_Name") or "").strip()
                if name:
                    names.add(name)
        except Exception:
            continue
    out = sorted(names, key=str.lower)[:300]
    if out:
        with _models_lock:
            _models_cache[key] = out
    return out


def session_view(s):
    return {"session": s["id"], "profile": agents.public_profile(s), "pending": list(s["pending"].values()),
            "turns_left": agents.MAX_TURNS_PER_SESSION - s["turns"],
            "limits": {"max_turns": agents.MAX_TURNS_PER_SESSION, "max_steps": agents.MAX_AGENT_STEPS,
                       "max_tool_calls": agents.MAX_TOOL_CALLS_PER_TURN, "max_output_tokens": llm.MAX_OUTPUT_TOKENS}}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=STATIC, **k)

    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            print("  (the browser stopped waiting before the answer was ready)", flush=True)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n > 50_000:
            raise ValueError("too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _ip(self):
        return (self.headers.get("X-Forwarded-For", "").split(",")[0].strip() or self.client_address[0])

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/healthz":
            return self._json(200, {"ok": True})
        if path == "/api/session":
            sid = self.path.split("sid=")[-1] if "sid=" in self.path else ""
            return self._json(200, session_view(SESSIONS.get(sid)))
        if path == "/api/models":
            q = parse_qs(urlparse(self.path).query)
            return self._json(200, {"models": lookup_models(q.get("make", [""])[0], q.get("year", [""])[0])})
        return super().do_GET()

    def do_POST(self):
        try:
            data = self._body()
        except ValueError:
            return self._json(400, {"ok": False, "error": "Bad request."})
        s = SESSIONS.get(str(data.get("session", "")))
        path = self.path.split("?")[0]

        if path == "/api/profile":
            try:
                return self._json(200, {"ok": True, "profile": agents.set_profile(s, data.get("profile") or {})})
            except ValueError as e:
                return self._json(400, {"ok": False, "error": str(e)})
        if path == "/api/approve":
            with s["lock"]:
                res = agents.decide(s, str(data.get("id", "")), data.get("decision") == "approve")
            return self._json(200 if res["ok"] else 409, res)
        if path == "/api/reset":
            s.update(history=[], turns=0, pending={}, last_agent=None, last_reply_asked=False)
            return self._json(200, {"ok": True, **session_view(s)})
        if path != "/api/chat":
            return self._json(404, {"ok": False, "error": "Not found"})

        trusted = EVAL_TOKEN and self.headers.get("X-Eval-Token") == EVAL_TOKEN
        if not trusted and not visitor_allowed(self._ip()):
            return self._json(429, {"ok": False, "error": "You're sending messages quickly. Wait a minute, then try again."})
        if not _slots.acquire(timeout=60):
            return self._json(503, {"ok": False, "error": "A lot of people are using the agent right now. Try again in a moment."})
        t0 = time.time()
        print(f"New message (chat {s['id'][:6]}...)", flush=True)
        try:
            client = get_llm()
            with s["lock"]:
                res = agents.handle_turn(s, data.get("message", ""), client, secrets=[client.api_key])
        except llm.LLMError as e:
            res = {"ok": False, "error": str(e)}
        except Exception as e:  # never leave the browser waiting forever
            res = {"ok": False, "error": f"Server error: {type(e).__name__}: {e}"}
        finally:
            _slots.release()
        print(f"  done in {time.time() - t0:.0f}s: {'ok' if res.get('ok') else 'ERROR - ' + str(res.get('error'))[:200]}",
              flush=True)
        res["session"] = s["id"]
        res["turns_left"] = agents.MAX_TURNS_PER_SESSION - s["turns"]
        self._json(200 if res.get("ok") else 502, res)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    if HOSTED:
        print(f"Car Care Agent listening on {HOST}:{PORT}")
    else:
        url = f"http://localhost:{PORT}"
        print(f"Car Care Agent is running at {url}  (Ctrl+C to stop)")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
