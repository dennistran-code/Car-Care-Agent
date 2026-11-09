"""Tools for the Car Care agents.

Real external APIs (free, no key):
  get_recalls           NHTSA recalls by make/model/year
  get_owner_complaints  NHTSA owner complaints by make/model/year
  get_fuel_economy      EPA fueleconomy.gov MPG by make/model/year

Local tools (deterministic, run on the server):
  check_maintenance     mileage/time vs. generalized service intervals
  lookup_symptom        symptom description -> likely causes from a reference table
  plan_trip             projects a trip's mileage against the schedule, estimates fuel cost
  propose_profile_update  HUMAN-GATED: only creates a pending request; nothing is saved
                          until the owner presses Approve in the chat.
"""
import datetime as dt
import re
import uuid

import requests

HTTP_TIMEOUT = 10
NHTSA_RECALLS = "https://api.nhtsa.gov/recalls/recallsByVehicle"
NHTSA_COMPLAINTS = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
FUEL_MODELS = "https://www.fueleconomy.gov/ws/rest/vehicle/menu/model"
FUEL_MENU = "https://www.fueleconomy.gov/ws/rest/vehicle/menu/options"
FUEL_VEHICLE = "https://www.fueleconomy.gov/ws/rest/vehicle/{id}"

# Generalized intervals, not manufacturer-specific: (miles, months). None = trigger doesn't apply.
MAINTENANCE_INTERVALS = {
    "oil change": (5000, 6),
    "tire rotation": (6000, 6),
    "brake inspection": (12000, 12),
    "engine air filter": (20000, 24),
    "cabin air filter": (20000, 24),
    "brake fluid": (None, 36),
    "coolant": (60000, 60),
    "transmission fluid": (45000, 48),
    "spark plugs": (60000, None),
    "battery check": (None, 12),
}
SERVICES = list(MAINTENANCE_INTERVALS)

# Symptom reference table. Costs are rough US ranges for parts + labor, for guidance only.
SYMPTOMS = [
    {"id": "brake_grind", "keywords": ["grind", "grinding", "metal on metal", "scrape", "scraping"],
     "context": ["brake", "braking", "stop", "stopping", "pedal"],
     "causes": ["Brake pads worn down to the metal backing", "Damaged or scored rotors"],
     "related_services": ["brake inspection"], "safety": True, "urgency": "Stop driving soon; get brakes checked within days.",
     "cost_range": [250, 700], "nhtsa_component": "SERVICE BRAKES"},
    {"id": "brake_squeal", "keywords": ["squeal", "squeak", "squealing", "squeaking", "screech"],
     "context": ["brake", "braking", "stop", "stopping"],
     "causes": ["Brake pad wear indicator touching the rotor", "Glazed or dusty pads"],
     "related_services": ["brake inspection"], "safety": True, "urgency": "Get brakes inspected within a week or two.",
     "cost_range": [150, 400], "nhtsa_component": "SERVICE BRAKES"},
    {"id": "belt_squeal", "keywords": ["squeal", "squealing", "chirp", "chirping"],
     "context": ["start", "starting", "cold", "accelerat", "hood", "engine", "ac", "a/c"],
     "causes": ["Worn or loose serpentine belt", "Failing belt tensioner"],
     "related_services": [], "safety": False, "urgency": "Have it checked soon; a snapped belt can stop the car.",
     "cost_range": [100, 300], "nhtsa_component": "ENGINE"},
    {"id": "clunk_bumps", "keywords": ["clunk", "clunking", "knock", "knocking", "rattle", "rattling", "thud"],
     "context": ["bump", "bumps", "pothole", "suspension", "rough road", "over"],
     "causes": ["Worn sway bar links", "Worn struts or shocks", "Loose or worn ball joints"],
     "related_services": [], "safety": True, "urgency": "Get the suspension inspected soon.",
     "cost_range": [150, 900], "nhtsa_component": "SUSPENSION"},
    {"id": "click_turning", "keywords": ["click", "clicking", "pop", "popping"],
     "context": ["turn", "turning", "corner", "steering", "parking lot"],
     "causes": ["Worn CV joint or torn CV boot"],
     "related_services": [], "safety": True, "urgency": "Get it checked within a couple of weeks.",
     "cost_range": [200, 800], "nhtsa_component": "POWER TRAIN"},
    {"id": "hum_speed", "keywords": ["hum", "humming", "whir", "whirring", "roar", "roaring", "drone"],
     "context": ["speed", "highway", "faster", "mph", "louder", "driving"],
     "causes": ["Failing wheel bearing", "Uneven tire wear"],
     "related_services": ["tire rotation"], "safety": True, "urgency": "Get it checked soon; a failed bearing is dangerous.",
     "cost_range": [200, 600], "nhtsa_component": "WHEELS"},
    {"id": "vibration_braking", "keywords": ["vibrat", "shak", "pulsat", "shudder", "wobble"],
     "context": ["brake", "braking", "stop", "stopping", "pedal"],
     "causes": ["Warped brake rotors"],
     "related_services": ["brake inspection"], "safety": True, "urgency": "Get brakes inspected soon.",
     "cost_range": [250, 600], "nhtsa_component": "SERVICE BRAKES"},
    {"id": "vibration_speed", "keywords": ["vibrat", "shak", "wobble", "shimmy"],
     "context": ["speed", "highway", "steering wheel", "mph", "driving"], "exclude": ["brake", "braking", "stopping"],
     "causes": ["Wheels out of balance", "Uneven tire wear", "Worn tie rods"],
     "related_services": ["tire rotation"], "safety": True, "urgency": "Have tires and steering checked soon.",
     "cost_range": [50, 400], "nhtsa_component": "STEERING"},
    {"id": "check_engine", "keywords": ["check engine", "engine light", "cel", "malfunction light"],
     "context": [],
     "causes": ["Many possibilities (loose gas cap, oxygen sensor, misfire). The light stores a code."],
     "related_services": ["spark plugs"], "safety": False,
     "urgency": "Get the code read with an OBD-II scanner (many parts stores do this free). A FLASHING light means stop driving.",
     "cost_range": [0, 1000], "nhtsa_component": "ENGINE"},
    {"id": "grind_shift", "keywords": ["grind", "slip", "slipping", "jerk", "jerking", "hard shift", "delay"],
     "context": ["shift", "shifting", "gear", "transmission"],
     "causes": ["Low or old transmission fluid", "Worn clutch (manual)", "Internal transmission wear"],
     "related_services": ["transmission fluid"], "safety": False, "urgency": "Get it checked soon before it gets worse.",
     "cost_range": [150, 3500], "nhtsa_component": "POWER TRAIN"},
    {"id": "sweet_smell", "keywords": ["sweet smell", "syrup", "maple", "coolant smell"],
     "context": [],
     "causes": ["Coolant leak (heater core, hose, or radiator)"],
     "related_services": ["coolant"], "safety": True, "urgency": "Check coolant level; don't drive if the temperature gauge climbs.",
     "cost_range": [100, 1000], "nhtsa_component": "ENGINE AND ENGINE COOLING"},
    {"id": "burning_smell", "keywords": ["burning smell", "burnt smell", "smells burnt", "burning oil"],
     "context": [],
     "causes": ["Oil leaking onto hot engine parts", "Overheated brakes or clutch"],
     "related_services": ["oil change"], "safety": True, "urgency": "Pull over safely if you see smoke; get it checked right away.",
     "cost_range": [100, 1200], "nhtsa_component": "ENGINE"},
    {"id": "hard_start", "keywords": ["won't start", "wont start", "slow crank", "hard to start", "clicking when", "dead battery"],
     "context": [],
     "causes": ["Weak or old battery", "Corroded battery terminals", "Failing starter"],
     "related_services": ["battery check"], "safety": False, "urgency": "Test the battery (often free at parts stores).",
     "cost_range": [100, 600], "nhtsa_component": "ELECTRICAL SYSTEM"},
]

CLARIFYING_QUESTIONS = [
    "What does it sound like (grinding, squealing, clunking, humming, clicking)?",
    "When does it happen (braking, turning, going over bumps, accelerating, at highway speed, starting up)?",
    "Where does it seem to come from (front, back, a wheel, under the hood)?",
]


# ----------------------------------------------------------------------------- context

class ToolContext:
    """What the tools may touch for one chat session. Only the session's own profile."""

    def __init__(self, profile, today=None):
        self.profile = profile
        self.today = today or dt.date.today()
        self.pending = []  # gated actions created during this turn


# ----------------------------------------------------------------------------- helpers

def _vehicle(ctx, make=None, model=None, year=None):
    p = ctx.profile
    make, model = (make or p.get("make") or "").strip(), (model or p.get("model") or "").strip()
    try:
        year = int(year or p.get("year") or 0)
    except (TypeError, ValueError):
        year = 0
    if not (make and model and year):
        return None, {"ok": False, "error": "Need make, model and year. Ask the owner or have them fill in the car profile."}
    return (make, model, year), None


def _get_json(url, params=None, headers=None):
    r = requests.get(url, params=params, headers=headers or {}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _months_between(d1, d2):
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    if d2.day < d1.day:
        months -= 1
    return max(months, 0)


def _history(ctx, service_history):
    if service_history is None:
        service_history = ctx.profile.get("services", [])
    latest = {}
    for s in service_history:
        name = str(s.get("service", "")).strip().lower()
        try:
            when = dt.date.fromisoformat(str(s.get("date")))
            miles = int(s.get("mileage"))
        except (TypeError, ValueError):
            continue
        if name not in latest or miles > latest[name]["mileage"]:
            latest[name] = {"mileage": miles, "date": when}
    return latest


def _mileage(ctx, current_mileage):
    try:
        return int(current_mileage if current_mileage is not None else ctx.profile.get("mileage"))
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- tools

def check_maintenance(ctx, current_mileage=None, service_history=None):
    miles_now = _mileage(ctx, current_mileage)
    if miles_now is None:
        return {"ok": False, "error": "Current mileage unknown. Ask the owner for it."}
    latest = _history(ctx, service_history)
    items = []
    for name, (mi_int, mo_int) in MAINTENANCE_INTERVALS.items():
        if name not in latest:
            items.append({"service": name, "status": "unknown", "reason": "No record of this service. Recommend checking."})
            continue
        last = latest[name]
        miles_since = miles_now - last["mileage"]
        months_since = _months_between(last["date"], ctx.today)
        overdue, soon = [], False
        if mi_int is not None:
            if miles_since >= mi_int:
                overdue.append(f"{miles_since} mi since last (interval {mi_int})")
            elif miles_since >= 0.9 * mi_int:
                soon = True
        if mo_int is not None:
            if months_since >= mo_int:
                overdue.append(f"{months_since} months since last (interval {mo_int})")
            elif months_since >= mo_int - 1:
                soon = True
        status = "overdue" if overdue else ("due_soon" if soon else "ok")
        items.append({"service": name, "status": status, "miles_since": miles_since,
                      "months_since": months_since, "reason": "; ".join(overdue) or "Within interval"})
    return {"ok": True, "as_of": ctx.today.isoformat(), "current_mileage": miles_now, "items": items,
            "note": "Intervals are generalized, not manufacturer-specific."}


def get_recalls(ctx, make=None, model=None, year=None):
    v, err = _vehicle(ctx, make, model, year)
    if err:
        return err
    make, model, year = v
    try:
        data = _get_json(NHTSA_RECALLS, {"make": make, "model": model, "modelYear": year})
    except Exception as e:
        return {"ok": False, "error": f"NHTSA recall lookup failed: {e}",
                "note": "Recall status is UNKNOWN. Do not tell the owner there are no recalls."}
    results = data.get("results") or data.get("Results") or []
    recalls = []
    for item in results[:10]:
        remedy = item.get("Remedy") or ""
        recalls.append({"campaign": item.get("NHTSACampaignNumber"), "component": item.get("Component"),
                        "summary": (item.get("Summary") or "")[:300], "consequence": item.get("Consequence"),
                        "free_fix": "free of charge" in remedy.lower()})
    return {"ok": True, "vehicle": f"{year} {make} {model}", "recall_count": data.get("Count", len(results)),
            "recalls": recalls, "source": "NHTSA",
            "note": "Recalls are listed by model year; whether THIS car's VIN is affected needs a VIN check at nhtsa.gov/recalls."}


def get_owner_complaints(ctx, make=None, model=None, year=None, component_keyword=None):
    v, err = _vehicle(ctx, make, model, year)
    if err:
        return err
    make, model, year = v
    try:
        data = _get_json(NHTSA_COMPLAINTS, {"make": make, "model": model, "modelYear": year})
    except Exception as e:
        return {"ok": False, "error": f"NHTSA complaint lookup failed: {e}"}
    results = data.get("results") or data.get("Results") or []
    kw = (component_keyword or "").strip().upper()
    words = [w for w in re.split(r"[^A-Z]+", kw) if len(w) > 2]
    matched = [c for c in results if not words or any(w in str(c.get("components", "")).upper() for w in words)]
    samples = [{"components": c.get("components"), "summary": str(c.get("summary") or "")[:220],
                "crash": c.get("crash"), "fire": c.get("fire")} for c in matched[:4]]
    return {"ok": True, "vehicle": f"{year} {make} {model}", "total_complaints": len(results),
            "matching_component": kw or None, "matching_complaints": len(matched), "examples": samples,
            "source": "NHTSA owner complaints",
            "note": "Complaints are unverified owner reports, not confirmed defects."}


def get_fuel_economy(ctx, make=None, model=None, year=None):
    v, err = _vehicle(ctx, make, model, year)
    if err:
        return err
    make, model, year = v
    hdr = {"Accept": "application/json"}

    def menu_items(url, params):
        items = (_get_json(url, params, hdr) or {}).get("menuItem") or []
        return [items] if isinstance(items, dict) else items
    try:
        # The EPA names models by body style ("Civic 4Dr", "Camry Hybrid"), so find the closest EPA name first.
        epa_model = model
        names = [m.get("text", "") for m in menu_items(FUEL_MODELS, {"year": year, "make": make})]
        want = model.lower().strip()
        exact = [n for n in names if n.lower() == want]
        starts = [n for n in names if n.lower().startswith(want + " ") or n.lower().startswith(want + "-")]
        contains = [n for n in names if want in n.lower()]
        if exact or starts or contains:
            epa_model = (exact or sorted(starts, key=lambda n: ("4dr" not in n.lower(), len(n))) or contains)[0]
        items = menu_items(FUEL_MENU, {"year": year, "make": make, "model": epa_model})
        if not items:
            return {"ok": False, "error": f"No EPA fuel economy entry found for {year} {make} {model}."}
        rec = _get_json(FUEL_VEHICLE.format(id=items[0]["value"]), headers=hdr)
    except Exception as e:
        return {"ok": False, "error": f"EPA fuel economy lookup failed: {e}"}

    def num(*keys):
        for k in keys:
            try:
                val = float(rec.get(k))
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
        return None
    return {"ok": True, "vehicle": f"{year} {make} {model}", "epa_model_name": epa_model, "trim_used": items[0].get("text"),
            "combined_mpg": num("comb08", "combA08"), "city_mpg": num("city08", "cityA08"),
            "highway_mpg": num("highway08", "highwayA08"), "source": "EPA fueleconomy.gov",
            "note": "Uses the first trim listed for this model year."}


def lookup_symptom(ctx, description):
    text = str(description or "").lower()
    scored = []
    for s in SYMPTOMS:
        kw = sum(1 for k in s["keywords"] if k in text)
        if not kw:
            continue
        ctx_hits = sum(1 for c in s["context"] if c in text)
        if s["context"] and not ctx_hits:
            continue  # sound matches but the situation doesn't
        if any(x in text for x in s.get("exclude", [])):
            continue
        scored.append((kw * 2 + ctx_hits, s))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return {"ok": True, "matches": [], "needs_clarification": True, "clarifying_questions": CLARIFYING_QUESTIONS,
                "note": "No confident match. Ask the owner these questions before suggesting causes or costs."}
    out = []
    for _, s in scored[:2]:
        out.append({"symptom_id": s["id"], "likely_causes": s["causes"], "safety_related": s["safety"],
                    "urgency": s["urgency"], "cost_range_usd": s["cost_range"],
                    "related_services": s["related_services"], "nhtsa_component": s["nhtsa_component"]})
    return {"ok": True, "matches": out, "needs_clarification": False,
            "note": "Reference-table guidance, not a mechanic's diagnosis. Costs are rough parts+labor ranges."}


def plan_trip(ctx, trip_miles, current_mileage=None, mpg=None, gas_price=None):
    try:
        trip = int(float(trip_miles))
    except (TypeError, ValueError):
        return {"ok": False, "error": "trip_miles must be a number."}
    if trip <= 0 or trip > 20000:
        return {"ok": False, "error": "trip_miles must be between 1 and 20,000."}
    status = check_maintenance(ctx, current_mileage)
    if not status["ok"]:
        return status
    now = status["current_mileage"]
    before, during = [], []
    for it in status["items"]:
        mi_int = MAINTENANCE_INTERVALS[it["service"]][0]
        if it["status"] in ("overdue", "due_soon"):
            before.append({"service": it["service"], "why": it["status"].replace("_", " ")})
        elif it["status"] == "ok" and mi_int and it["miles_since"] + trip >= mi_int:
            during.append({"service": it["service"], "why": f"reaches its {mi_int}-mile interval during the trip"})
    unknown = [it["service"] for it in status["items"] if it["status"] == "unknown"]
    fuel = None
    try:
        price = float(gas_price) if gas_price else 3.20
        if mpg:
            gallons = trip / float(mpg)
            fuel = {"mpg_used": float(mpg), "gallons": round(gallons, 1), "gas_price_used": price,
                    "estimated_cost_usd": round(gallons * price, 2),
                    "assumption": "gas price is the owner's figure" if gas_price else "gas price assumed $3.20/gal"}
    except (TypeError, ValueError, ZeroDivisionError):
        fuel = None
    return {"ok": True, "trip_miles": trip, "mileage_after_trip": now + trip,
            "do_before_trip": before, "would_come_due_during_trip": during, "unknown_status": unknown,
            "fuel": fuel or {"note": "Pass mpg (from get_fuel_economy) to estimate fuel cost."}}


def propose_profile_update(ctx, action, service=None, mileage=None, date=None):
    """Creates a PENDING change. Nothing is written until the owner approves in the UI."""
    action = str(action or "").strip()
    args = {"action": action}
    if action == "add_service":
        svc = str(service or "").strip().lower()
        if svc not in MAINTENANCE_INTERVALS:
            return {"ok": False, "error": f"Unknown service. Use one of: {', '.join(SERVICES)}"}
        try:
            miles = int(mileage)
            day = dt.date.fromisoformat(str(date or ctx.today.isoformat()))
        except (TypeError, ValueError):
            return {"ok": False, "error": "add_service needs mileage (number) and date (YYYY-MM-DD)."}
        if not 0 <= miles <= 1_000_000 or day > ctx.today:
            return {"ok": False, "error": "Mileage must be 0-1,000,000 and the date can't be in the future."}
        args.update(service=svc, mileage=miles, date=day.isoformat())
    elif action == "set_mileage":
        try:
            miles = int(mileage)
        except (TypeError, ValueError):
            return {"ok": False, "error": "set_mileage needs mileage (number)."}
        if not 0 <= miles <= 1_000_000:
            return {"ok": False, "error": "Mileage must be 0-1,000,000."}
        args.update(mileage=miles)
    else:
        return {"ok": False, "error": "action must be add_service or set_mileage."}
    pending = {"id": uuid.uuid4().hex[:10], "tool": "propose_profile_update", "arguments": args}
    ctx.pending.append(pending)
    return {"ok": True, "status": "WAITING_FOR_OWNER_APPROVAL", "request_id": pending["id"], "arguments": args,
            "note": "NOT saved. The owner sees an Approve / Reject card. Tell them it is waiting for their "
                    "approval; never say it has been saved."}


def apply_profile_update(profile, args):
    """Called ONLY by the server after the owner presses Approve."""
    if args["action"] == "add_service":
        profile.setdefault("services", []).append(
            {"service": args["service"], "mileage": args["mileage"], "date": args["date"]})
        if args["mileage"] > int(profile.get("mileage") or 0):
            profile["mileage"] = args["mileage"]
    elif args["action"] == "set_mileage":
        profile["mileage"] = args["mileage"]
    return profile


# ----------------------------------------------------------------------------- schemas

_VEH = {"make": {"type": "string", "description": "e.g. Honda. Omit to use the car profile."},
        "model": {"type": "string", "description": "e.g. Civic. Omit to use the car profile."},
        "year": {"type": "integer", "description": "model year. Omit to use the car profile."}}

SCHEMAS = {
    "check_maintenance": {
        "description": "Compare current mileage and service history against standard intervals. Returns each "
                       "service as overdue, due_soon, ok, or unknown. Omit arguments to use the saved car profile.",
        "parameters": {"type": "object", "properties": {
            "current_mileage": {"type": "integer"},
            "service_history": {"type": "array", "description": "Only if the owner gave history in chat that "
                                "isn't in the profile.", "items": {"type": "object", "properties": {
                                    "service": {"type": "string"}, "mileage": {"type": "integer"},
                                    "date": {"type": "string", "description": "YYYY-MM-DD"}}}}}}},
    "get_recalls": {
        "description": "Live NHTSA lookup of safety recalls for a make/model/year. Use for ANY question about "
                       "recalls or factory defects. Never state recall status without calling this.",
        "parameters": {"type": "object", "properties": _VEH}},
    "get_owner_complaints": {
        "description": "Live NHTSA lookup of owner-filed complaints for a make/model/year, optionally filtered by "
                       "component (e.g. SERVICE BRAKES). Shows whether other owners report the same problem.",
        "parameters": {"type": "object", "properties": dict(_VEH, component_keyword={
            "type": "string", "description": "e.g. SERVICE BRAKES, STEERING, ENGINE (from lookup_symptom)"})}},
    "get_fuel_economy": {
        "description": "Live EPA fueleconomy.gov lookup of city/highway/combined MPG for a make/model/year.",
        "parameters": {"type": "object", "properties": _VEH}},
    "lookup_symptom": {
        "description": "Match the owner's description of a noise, smell, vibration or warning light to likely "
                       "causes, urgency, related services and a rough cost range. If needs_clarification is true, "
                       "ask the returned questions instead of guessing.",
        "parameters": {"type": "object", "properties": {"description": {"type": "string"}},
                       "required": ["description"]}},
    "plan_trip": {
        "description": "Project a trip's miles against the maintenance schedule: what to do before leaving, what "
                       "comes due during the trip, and fuel cost if mpg is given. Uses the car profile.",
        "parameters": {"type": "object", "properties": {
            "trip_miles": {"type": "integer", "description": "Total miles, round trip if the owner is coming back."},
            "current_mileage": {"type": "integer"},
            "mpg": {"type": "number", "description": "From get_fuel_economy."},
            "gas_price": {"type": "number", "description": "Only if the owner gave one."}},
            "required": ["trip_miles"]}},
    "propose_profile_update": {
        "description": "Ask to save a service record or new mileage to the owner's car profile. This does NOT "
                       "save anything: it shows the owner an Approve/Reject card. Use only when the owner reports "
                       "work done or new mileage.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add_service", "set_mileage"]},
            "service": {"type": "string", "enum": SERVICES},
            "mileage": {"type": "integer"},
            "date": {"type": "string", "description": "YYYY-MM-DD; today if the owner says today."}},
            "required": ["action"]}},
}

FUNCS = {"check_maintenance": check_maintenance, "get_recalls": get_recalls,
         "get_owner_complaints": get_owner_complaints, "get_fuel_economy": get_fuel_economy,
         "lookup_symptom": lookup_symptom, "plan_trip": plan_trip,
         "propose_profile_update": propose_profile_update}


def declarations(names):
    return [{"name": n, "description": SCHEMAS[n]["description"], "parameters": SCHEMAS[n]["parameters"]}
            for n in names]


def run_tool(name, args, ctx, allowed):
    if name not in allowed:
        return {"ok": False, "error": f"Tool '{name}' is not available to this agent."}
    try:
        return FUNCS[name](ctx, **(args or {}))
    except TypeError as e:
        return {"ok": False, "error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # a tool bug must never crash the chat
        return {"ok": False, "error": f"{name} failed: {e}"}
