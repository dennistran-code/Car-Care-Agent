"""A scripted stand-in for Gemini, for offline tests only. It is NOT the real model.

It mimics how a well-behaved model would use each agent's tools, so the tests can check the
plumbing: routing, tool dispatch, limits, the Critic loop, and the human gate.
"""
import json
import re

from agents import CRITIC_PROMPT

NHTSA_RECALLS_SAMPLE = {"Count": 2, "results": [
    {"NHTSACampaignNumber": "16V725000", "Component": "PARKING BRAKE",
     "Summary": "The electric parking brake may not engage if applied right after the ignition is turned off.",
     "Consequence": "The vehicle may roll away, increasing the risk of a crash.",
     "Remedy": "Dealers will update the software, free of charge."},
    {"NHTSACampaignNumber": "23V858000", "Component": "FUEL SYSTEM, GASOLINE:DELIVERY:FUEL PUMP",
     "Summary": "The fuel pump inside the fuel tank may fail.",
     "Consequence": "Fuel pump failure can cause an engine stall while driving.",
     "Remedy": "Dealers will replace the fuel pump module, free of charge."}]}
NHTSA_COMPLAINTS_SAMPLE = {"count": 3, "results": [
    {"odiNumber": 1, "components": "SERVICE BRAKES", "summary": "Brakes grind at low speed.", "crash": False, "fire": False},
    {"odiNumber": 2, "components": "SERVICE BRAKES,PARKING BRAKE", "summary": "Rear brakes noisy.", "crash": False, "fire": False},
    {"odiNumber": 3, "components": "ENGINE", "summary": "Engine stalled.", "crash": False, "fire": False}]}
FUEL_MODELS_SAMPLE = {"menuItem": [{"text": "Accord", "value": "Accord"}, {"text": "Civic 2Dr", "value": "Civic 2Dr"},
                                   {"text": "Civic 4Dr", "value": "Civic 4Dr"}, {"text": "Civic HF", "value": "Civic HF"}]}
FUEL_MENU_SAMPLE = {"menuItem": [{"text": "Auto (variable gear ratios), 4 cyl, 2.0 L", "value": "36611"}]}
FUEL_VEHICLE_SAMPLE = {"comb08": "35", "city08": "31", "highway08": "41"}


def fake_http_get(url, params=None, headers=None, timeout=None):
    class R:
        def __init__(self, data):
            self._d = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._d
    if "recallsByVehicle" in url:
        return R(NHTSA_RECALLS_SAMPLE)
    if "complaintsByVehicle" in url:
        return R(NHTSA_COMPLAINTS_SAMPLE)
    if "menu/model" in url:
        return R(FUEL_MODELS_SAMPLE)
    if "menu/options" in url:
        return R(FUEL_MENU_SAMPLE if (params or {}).get("model") == "Civic 4Dr" else {})
    return R(FUEL_VEHICLE_SAMPLE)


def _text(t):
    return {"role": "model", "parts": [{"text": t}]}


def _calls(*calls, say=None):
    parts = [{"text": say}] if say else []
    parts += [{"functionCall": {"name": n, "args": a}} for n, a in calls]
    return {"role": "model", "parts": parts}


class StubLLM:
    api_key = "STUB-SECRET-KEY-123456"

    def __init__(self, misbehave=None):
        self.calls = 0
        self.misbehave = misbehave or set()   # e.g. {"claim_saved", "fake_recall", "endless"}

    def call(self, system, contents, tools=None, model=None, json_mode=False, deadline=None):
        self.calls += 1
        if system == CRITIC_PROMPT:
            prompt = contents[0]["parts"][0]["text"]
            draft = prompt.split("Draft answer:\n", 1)[1].split("\n\nAutomatic checks", 1)[0]
            bad = "nothing" not in prompt.split("Automatic checks already found:", 1)[1]
            return _text(json.dumps({"approved": not bad, "issues": ["fix flagged items"] if bad else []})), "stub-critic"
        user_texts = [p["text"] for c in contents if c["role"] == "user" for p in c["parts"] if "text" in p]
        last_user = user_texts[-1] if user_texts else ""
        if "The Critic found these problems" in last_user:
            return _text("Here is the corrected answer, using only the tool results. Saving needs your approval."), "stub"
        obs = [p["functionResponse"]["response"]["result"] for c in contents
               for p in c["parts"] if "functionResponse" in p]
        names = [p["functionResponse"]["name"] for c in contents for p in c["parts"] if "functionResponse" in p]
        msg = last_user.lower()
        if "endless" in self.misbehave:
            return _calls(("check_maintenance", {})), "stub"
        if "poem" in msg or "system prompt" in msg:
            return _text("I only help with car care: maintenance, recalls, noises and trip prep."), "stub"

        if "You are the Maintenance agent" in system:
            if re.search(r"(changed|did|replaced).*(oil)|save", msg) and "propose_profile_update" not in names:
                return _calls(("propose_profile_update", {"action": "add_service", "service": "oil change",
                                                          "mileage": 62400, "date": "2026-11-01"})), "stub"
            if "propose_profile_update" in names:
                t = "I've saved it to your profile." if "claim_saved" in self.misbehave else \
                    "I've sent that change for your approval. Press Approve to save it."
                return _text(t), "stub"
            if not obs:
                return _calls(("check_maintenance", {}), ("get_recalls", {}), say="Checking schedule and recalls."), "stub"
            m = next(o for n, o in zip(names, obs) if n == "check_maintenance")
            over = [i["service"] for i in m["items"] if i["status"] == "overdue"]
            r = next((o for n, o in zip(names, obs) if n == "get_recalls"), {})
            camp = ["99V999000"] if "fake_recall" in self.misbehave else [x["campaign"] for x in r.get("recalls", [])]
            return _text("Overdue now: " + ", ".join(over) + ". Recalls: " + ", ".join(camp) +
                         ". Intervals are generalized."), "stub"

        if "You are the Diagnostic agent" in system:
            if not obs:
                return _calls(("lookup_symptom", {"description": last_user})), "stub"
            first = obs[0]
            if first.get("needs_clarification"):
                return _text("Can you tell me more? " + " ".join(first["clarifying_questions"][:2])), "stub"
            if len(obs) == 1:
                comp = first["matches"][0]["nhtsa_component"]
                return _calls(("check_maintenance", {}), ("get_recalls", {}),
                              ("get_owner_complaints", {"component_keyword": comp})), "stub"
            lo, hi = first["matches"][0]["cost_range_usd"]
            return _text(f"Most likely: {first['matches'][0]['likely_causes'][0]}. Please get it checked. "
                         f"Rough cost ${lo}-${hi}."), "stub"

        if "You are the Trip Prep agent" in system:
            if not obs:
                return _calls(("get_fuel_economy", {})), "stub"
            if len(obs) == 1:
                miles = int((re.search(r"(\d[\d,]*)\s*miles", msg) or re.search(r"(\d+)", "500")).group(1).replace(",", ""))
                return _calls(("plan_trip", {"trip_miles": miles, "mpg": obs[0].get("combined_mpg")})), "stub"
            plan = obs[1]
            cost = plan["fuel"].get("estimated_cost_usd")
            before = ", ".join(i["service"] for i in plan["do_before_trip"]) or "nothing"
            return _text(f"Do before you leave: {before}. Fuel estimate about ${cost}."), "stub"
        return _text("ok"), "stub"
