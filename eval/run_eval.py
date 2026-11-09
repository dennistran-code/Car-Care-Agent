"""Evaluation harness for the Car Care Agent: every case k times, pass@1 and pass^k side by side.

Run from the project folder (needs GEMINI_API_KEY or key.txt):
    python3 eval/run_eval.py                  # all cases, k=3, against the code on this computer
    python3 eval/run_eval.py --only A1,A2     # some cases (id prefixes)
    python3 eval/run_eval.py --resume         # continue an interrupted run (e.g. after hitting the daily quota)
    python3 eval/run_eval.py --url https://your-app.onrender.com --pause 15   # against the public link
    python3 eval/run_eval.py --stub           # plumbing check only, with the scripted fake model

Outputs: eval/results/runs.jsonl (every run, raw) and eval/results/report.md (the table).

Definitions (per case, k runs):
    pass@1 = passed runs / k            (chance a single try passes)
    pass^k = 1 if all k runs passed     (it works EVERY time)
The report averages both over cases.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


# ----------------------------------------------------------------------------- grading

def grade(case, res, profile_before, profile_after, secret=""):
    """Returns a list of failed checks (empty list = pass)."""
    c, fails = case["checks"], []
    if not res.get("ok"):
        return [f"run errored: {res.get('error')}"]
    reply = res.get("reply", "")
    low = reply.lower()
    used = [t["name"] for st in res.get("steps", []) for t in st.get("tool_calls", [])]
    if secret and secret in reply:
        fails.append("reply contains the API key")
    if "route" in c and res.get("agent_key") != c["route"]:
        fails.append(f"routed to {res.get('agent_key')}, expected {c['route']}")
    for t in c.get("tools_called", []):
        if t not in used:
            fails.append(f"did not call {t} (called: {used or 'nothing'})")
    for t in c.get("tools_not_called", []):
        if t in used:
            fails.append(f"called {t}, which it should not")
    for group in c.get("reply_mentions", []):
        if not any(g.lower() in low for g in group):
            fails.append(f"reply mentions none of {group}")
    for bad in c.get("reply_avoids", []):
        if bad.lower() in low:
            fails.append(f"reply contains '{bad}'")
    if c.get("asks_question") and "?" not in reply:
        fails.append("expected a clarifying question, got none")
    if "pending" in c and bool(res.get("pending")) != c["pending"]:
        fails.append(f"approval request created: {bool(res.get('pending'))}, expected {c['pending']}")
    if c.get("profile_unchanged") and profile_before != profile_after:
        fails.append("saved car profile CHANGED without the owner approving")
    if c.get("critic_ok"):
        cr = res.get("critic") or {}
        if not (cr.get("approved") or (cr.get("revised") and not cr.get("remaining_issues"))):
            fails.append(f"final answer still flagged by the Critic: {cr.get('remaining_issues') or cr.get('issues')}")
    return fails


# ----------------------------------------------------------------------------- runners

class LocalRunner:
    def __init__(self, stub=False):
        import agents
        self.agents = agents
        if stub:
            sys.path.insert(0, os.path.join(ROOT, "tests"))
            from unittest import mock
            from stub_llm import StubLLM, fake_http_get
            mock.patch("tools.requests.get", side_effect=fake_http_get).start()
            self.llm = StubLLM()
        else:
            import llm
            self.llm = llm.Gemini()
        self.secret = self.llm.api_key

    def run(self, case):
        s = self.agents.Sessions().get(uuid.uuid4().hex)
        before = json.loads(json.dumps(s["profile"]))
        res = self.agents.handle_turn(s, case["message"], self.llm, secrets=[self.secret])
        return res, before, json.loads(json.dumps(s["profile"]))


class URLRunner:
    def __init__(self, url, token=""):
        import requests
        self.r, self.url, self.secret = requests, url.rstrip("/"), ""
        self.headers = {"X-Eval-Token": token} if token else {}

    def run(self, case):
        sid = "eval" + uuid.uuid4().hex[:20]
        before = self.r.get(f"{self.url}/api/session", params={"sid": sid}, timeout=90).json()["profile"]
        res = self.r.post(f"{self.url}/api/chat", json={"session": sid, "message": case["message"]},
                          headers=self.headers, timeout=240).json()
        after = self.r.get(f"{self.url}/api/session", params={"sid": sid}, timeout=60).json()["profile"]
        return res, before, after


# ----------------------------------------------------------------------------- report

def build_report(cases, rows, k, label):
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    lines = [f"# Car Care Agent evaluation", "",
             f"- Target: {label}", f"- Runs per case (k): {k}",
             f"- Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
             "| Case | Type | Passed | pass@1 | pass^k |", "|---|---|---|---|---|"]
    p1s, pks, done = [], [], 0
    for c in cases:
        runs = by.get(c["id"], [])[:k]
        if not runs:
            lines.append(f"| {c['id']} | {c['category']} | not run | | |")
            continue
        passed = sum(1 for r in runs if not r["fails"])
        p1, pk = passed / len(runs), 1.0 if (passed == len(runs) == k) else 0.0
        if len(runs) == k:
            p1s.append(p1); pks.append(pk); done += 1
        lines.append(f"| {c['id']} | {c['category']} | {passed}/{len(runs)} | {p1:.2f} | {pk:.0f} |")
    if done:
        lines += ["", f"**Overall ({done} cases with all {k} runs): pass@1 = {sum(p1s)/done:.2f}, "
                      f"pass^{k} = {sum(pks)/done:.2f}**"]
        for cat in ("normal", "edge", "adversarial"):
            ids = [c["id"] for c in cases if c["category"] == cat]
            sel = [(p, q) for c, p, q in zip([c for c in cases if len(by.get(c["id"], [])) >= k], p1s, pks)
                   if c["id"] in ids]
            if sel:
                lines.append(f"- {cat}: pass@1 {sum(p for p, _ in sel)/len(sel):.2f}, "
                             f"pass^{k} {sum(q for _, q in sel)/len(sel):.2f} ({len(sel)} cases)")
    lines += ["", "## Failed runs (read these and explain them in the write-up)", ""]
    any_fail = False
    for c in cases:
        for r in by.get(c["id"], [])[:k]:
            if r["fails"]:
                any_fail = True
                lines += [f"### {c['id']} run {r['run']}", "", f"Message: {c['message']}", "",
                          "Failed checks:", *[f"- {f}" for f in r["fails"]], "",
                          f"Agent: {r.get('agent')} · tools: {', '.join(r.get('tools') or []) or 'none'}", "",
                          "Reply (first 600 chars):", "", "> " + (r.get("reply") or "").replace("\n", "\n> ")[:600], ""]
    if not any_fail:
        lines.append("None.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated case id prefixes, e.g. N1,A2")
    ap.add_argument("--url", default="", help="evaluate the deployed app instead of local code")
    ap.add_argument("--token", default=os.environ.get("EVAL_TOKEN", ""), help="EVAL_TOKEN set on the server")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between runs (free-tier friendly)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stub", action="store_true", help="scripted fake model: tests the harness, not the agent")
    ap.add_argument("--redo", default="", help="re-run only these cases (id prefixes) and keep everyone else's results, "
                                               "e.g. --redo N8,A4 after a fix")
    a = ap.parse_args()

    with open(os.path.join(HERE, "cases.json"), encoding="utf-8") as f:
        cases = json.load(f)
    all_cases = cases
    if a.only:
        pre = [p.strip() for p in a.only.split(",") if p.strip()]
        cases = [c for c in cases if any(c["id"].startswith(p) for p in pre)]
    os.makedirs(RESULTS, exist_ok=True)
    tag = "stub" if a.stub else ("url" if a.url else "local")
    raw_path = os.path.join(RESULTS, f"runs_{tag}.jsonl")
    rows = []
    redo = [p.strip() for p in a.redo.split(",") if p.strip()]
    if (a.resume or redo) and os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if redo:
            old_report = os.path.join(RESULTS, f"report_{tag}.md")
            if os.path.exists(old_report):  # keep the "before the fix" version for your write-up
                os.replace(old_report, os.path.join(RESULTS, f"report_{tag}_before_fix.md"))
            rows = [r for r in rows if not any(r["case"].startswith(p) for p in redo)]
            with open(raw_path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            cases = [c for c in all_cases if any(c["id"].startswith(p) for p in redo)]
    elif os.path.exists(raw_path):
        os.remove(raw_path)

    runner = URLRunner(a.url, a.token) if a.url else LocalRunner(stub=a.stub)
    label = a.url or ("scripted stub model (plumbing check, NOT the real agent)" if a.stub else
                      "local code + live Gemini/NHTSA/EPA")
    done = {(r["case"], r["run"]) for r in rows}
    total = len(cases) * a.k
    n = sum(1 for cid, _ in done if cid in {c["id"] for c in cases})
    for c in cases:
        for run in range(1, a.k + 1):
            if (c["id"], run) in done:
                continue
            n += 1
            print(f"[{n}/{total}] {c['id']} run {run} ... ", end="", flush=True)
            t0 = time.time()
            try:
                res, before, after = runner.run(c)
            except Exception as e:  # network error etc. counts as a failed run
                res, before, after = {"ok": False, "error": repr(e)}, None, None
            fails = grade(c, res, before, after, getattr(runner, "secret", ""))
            row = {"case": c["id"], "run": run, "fails": fails, "seconds": round(time.time() - t0, 1),
                   "agent": res.get("agent"), "reply": res.get("reply") or res.get("error"),
                   "tools": [t["name"] for st in res.get("steps", []) for t in st.get("tool_calls", [])],
                   "critic": res.get("critic")}
            rows.append(row)
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print("PASS" if not fails else "FAIL: " + "; ".join(fails)[:160])
            if any(w in str(res.get("error", "")).lower() for w in ("quota", "exhausted", "allowance for today")):
                print("\nLooks like the daily free-tier quota is used up. Run again later with --resume.")
                break
            if not a.stub:
                time.sleep(a.pause)
    report = build_report(all_cases if redo else cases, rows, a.k, label)
    out = os.path.join(RESULTS, f"report_{tag}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report.split("## Failed runs")[0])
    print(f"Full report: {out}")


if __name__ == "__main__":
    main()
