"""Offline tests: no key, no internet. Run from the project folder:  python3 tests/test_offline.py

These use a scripted stand-in for Gemini (tests/stub_llm.py) and canned NHTSA/EPA data.
They check the plumbing. They do not measure how well the real model behaves; that's eval/run_eval.py.
"""
import datetime as dt
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agents  # noqa: E402
import tools  # noqa: E402
from stub_llm import StubLLM, fake_http_get  # noqa: E402

TODAY = dt.date(2026, 11, 4)
mock.patch("tools.requests.get", side_effect=fake_http_get).start()


def new_session():
    return agents.Sessions().get("testsession01")


def turn(s, msg, llm=None):
    return agents.handle_turn(s, msg, llm or StubLLM(), today=TODAY, secrets=["STUB-SECRET-KEY-123456"])


def test_router():
    s = new_session()
    assert agents.route("My brakes make a grinding noise", s)[0] == "diagnostic"
    assert agents.route("Driving 800 miles to Miami, is my car ready for the trip?", s)[0] == "trip"
    assert agents.route("What maintenance is due?", s)[0] == "maintenance"
    assert agents.route("Any recalls on my car?", s)[0] == "maintenance"


def test_maintenance_turn_uses_tools_and_critic_approves():
    r = turn(new_session(), "What maintenance is due and any recalls?")
    assert r["ok"] and r["agent_key"] == "maintenance"
    used = [c["name"] for st in r["steps"] for c in st["tool_calls"]]
    assert "check_maintenance" in used and "get_recalls" in used
    assert "oil change" in r["reply"] and "16V725000" in r["reply"]
    assert r["critic"]["approved"] is True


def test_diagnostic_chain_cross_checks():
    r = turn(new_session(), "I hear a grinding noise when braking")
    used = [c["name"] for st in r["steps"] for c in st["tool_calls"]]
    assert r["agent_key"] == "diagnostic"
    assert used[:1] == ["lookup_symptom"] and {"check_maintenance", "get_recalls", "get_owner_complaints"} <= set(used)
    comp = [c["observation"] for st in r["steps"] for c in st["tool_calls"] if c["name"] == "get_owner_complaints"][0]
    assert comp["matching_complaints"] == 2
    assert r["critic"]["approved"] is True  # $250-$700 comes from the tool


def test_vague_symptom_asks_and_follow_up_stays_with_diagnostic():
    s = new_session()
    r = turn(s, "my car is making a weird noise")
    assert r["reply"].endswith("?") or "?" in r["reply"]
    assert s["last_agent"] == "diagnostic" and s["last_reply_asked"]
    assert agents.route("only when I go over bumps, from the front", s)[0] == "diagnostic"


def test_trip_uses_two_real_tools():
    r = turn(new_session(), "Road trip of 1,200 miles next week, is my car ready?")
    used = [c["name"] for st in r["steps"] for c in st["tool_calls"]]
    assert r["agent_key"] == "trip" and used == ["get_fuel_economy", "plan_trip"]
    plan = r["steps"][1]["tool_calls"][0]["observation"]
    assert plan["fuel"]["estimated_cost_usd"] == round(1200 / 35 * 3.20, 2)
    assert "oil change" in [i["service"] for i in plan["do_before_trip"]]
    assert r["critic"]["approved"] is True


def test_human_gate_nothing_saved_until_approve():
    s = new_session()
    before = list(s["profile"]["services"])
    r = turn(s, "I changed my oil today, please save it")
    assert r["pending"] and r["pending"][0]["arguments"]["service"] == "oil change"
    assert s["profile"]["services"] == before          # NOT saved yet
    rid = r["pending"][0]["id"]
    out = agents.decide(s, rid, approve=True)
    assert out["ok"] and len(s["profile"]["services"]) == len(before) + 1
    assert agents.decide(s, rid, approve=True)["ok"] is False   # can't approve twice


def test_reject_leaves_profile_alone():
    s = new_session()
    before = list(s["profile"]["services"])
    r = turn(s, "I changed my oil, save that")
    agents.decide(s, r["pending"][0]["id"], approve=False)
    assert s["profile"]["services"] == before


def test_critic_catches_false_saved_claim_and_revision_runs():
    s = new_session()
    r = turn(s, "I changed my oil today, save it", StubLLM({"claim_saved"}))
    assert r["critic"]["approved"] is False and r["critic"]["revised"] is True
    assert "saved it to your profile" not in r["reply"]


def test_critic_catches_invented_recall_number():
    r = turn(new_session(), "Any recalls?", StubLLM({"fake_recall"}))
    assert any("99V999000" in i for i in r["critic"]["issues"])
    assert r["critic"]["revised"] is True


def test_step_and_tool_call_limits():
    r = turn(new_session(), "What's due?", StubLLM({"endless"}))
    assert r["status"] == "max_steps" and len(r["steps"]) == agents.MAX_AGENT_STEPS
    assert r["tool_calls_used"] <= agents.MAX_TOOL_CALLS_PER_TURN


def test_secret_never_returned():
    issues, hard = agents.grounding_issues("the key is STUB-SECRET-KEY-123456", [], [], ["STUB-SECRET-KEY-123456"])
    assert hard


def test_session_turn_cap():
    s = new_session()
    s["turns"] = agents.MAX_TURNS_PER_SESSION
    assert turn(s, "hi")["ok"] is False


def test_tool_failure_is_reported():
    with mock.patch("tools.requests.get", side_effect=Exception("timeout")):
        out = tools.get_recalls(tools.ToolContext(dict(agents.EXAMPLE_PROFILE)))
    assert out["ok"] is False and "UNKNOWN" in out["note"]


def test_propose_validation():
    ctx = tools.ToolContext(dict(agents.EXAMPLE_PROFILE), today=TODAY)
    assert tools.propose_profile_update(ctx, "add_service", "unicorn wash", 1, "2026-01-01")["ok"] is False
    assert tools.propose_profile_update(ctx, "add_service", "oil change", 1, "2030-01-01")["ok"] is False
    assert tools.propose_profile_update(ctx, "delete_everything")["ok"] is False
    assert ctx.pending == []


def test_tool_not_allowed_for_agent():
    ctx = tools.ToolContext(dict(agents.EXAMPLE_PROFILE), today=TODAY)
    assert tools.run_tool("plan_trip", {"trip_miles": 5}, ctx, ["get_recalls"])["ok"] is False


def test_symptom_table():
    ctx = tools.ToolContext({}, today=TODAY)
    assert tools.lookup_symptom(ctx, "grinding when I brake")["matches"][0]["symptom_id"] == "brake_grind"
    assert tools.lookup_symptom(ctx, "weird noise")["needs_clarification"] is True
    assert tools.lookup_symptom(ctx, "check engine light is on")["matches"][0]["symptom_id"] == "check_engine"



def test_daily_quota_falls_back_then_explains():
    import llm
    def resp(code, msg):
        m = mock.Mock(status_code=code)
        m.json.return_value = {"error": {"message": msg}} if code != 200 else \
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}]}
        return m
    day = resp(429, "Quota exceeded for metric: generate_content_free_tier_requests, limit: 250 per day")
    g = llm.Gemini(api_key="k", sleep=lambda s: None)
    with mock.patch("llm.requests.post", side_effect=[day, resp(200, "")]) as post:
        _, used = g.call("sys", [], model="main-model")
    assert used == llm.FALLBACK_MODEL and post.call_count == 2      # no wasted retries
    with mock.patch("llm.requests.post", side_effect=[day, day]):
        try:
            g.call("sys", [], model="main-model"); assert False
        except llm.LLMError as e:
            assert "allowance for today" in str(e)



def test_epa_model_name_matching():
    out = tools.get_fuel_economy(tools.ToolContext(dict(agents.EXAMPLE_PROFILE), today=TODAY))
    assert out["ok"] and out["epa_model_name"] == "Civic 4Dr" and out["combined_mpg"] == 35


def test_denying_a_fake_recall_number_is_not_flagged():
    obs = [("get_recalls", {"ok": True, "recall_count": 1, "recalls": [{"campaign": "16V725000"}]})]
    ok_draft = "I cannot confirm recall 99V123000, as that campaign number does not appear in the NHTSA results."
    bad_draft = "Yes, recall 99V123000 applies to your car."
    assert agents.grounding_issues(ok_draft, obs, [])[0] == []
    assert any("99V123000" in i for i in agents.grounding_issues(bad_draft, obs, [])[0])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", t.__name__, "-", repr(e))
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
