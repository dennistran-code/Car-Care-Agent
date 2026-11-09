# Car Care Agent

A multi-agent chat assistant that tells car owners what maintenance is due, what a noise or warning light might mean, whether there are open NHTSA recalls, and whether the car is ready for a trip.

## How it works

```
owner message
  → Router (code): picks one specialist from the words in the message
  → Maintenance agent | Diagnostic agent | Trip Prep agent   (each: own prompt, own tool set, LLM tool loop)
  → Critic agent (separate model call on a cheaper model) + automatic grounding checks
       approved → reply      rejected → the specialist rewrites once → reply
  → Human gate: any change to the saved car record shows an Approve / Reject card first
```

| Tool | Kind | Used by |
|---|---|---|
| `get_recalls` | real API: NHTSA recalls | Maintenance, Diagnostic |
| `get_owner_complaints` | real API: NHTSA owner complaints | Diagnostic |
| `get_fuel_economy` | real API: EPA fueleconomy.gov | Trip Prep |
| `check_maintenance` | local: generalized service intervals | all three |
| `lookup_symptom` | local: symptom reference table | Diagnostic |
| `plan_trip` | local: trip miles vs. schedule, fuel cost | Trip Prep |
| `propose_profile_update` | **human-gated**: only creates an approval request | all three |

Memory: each chat keeps a car record and recent messages in server memory for 3 hours.

Caps: 5 steps and 8 tool calls per message, 2,048 output tokens per model reply, 25 messages per chat, 5 messages per minute per visitor, 2 chats processed at once.

## Run it on your computer

```
python3 -m pip install -r requirements.txt
echo "YOUR-GEMINI-KEY" > key.txt
python3 server.py            # opens http://localhost:8000
```

## Tests and evaluation

```
python3 tests/test_offline.py              # 17 plumbing tests, no key or internet needed
python3 eval/run_eval.py                   # 16 cases x 3 runs with the real model -> eval/results/report_local.md
python3 eval/run_eval.py --resume          # continue if the free quota ran out partway
python3 eval/run_eval.py --url https://YOUR-LINK.onrender.com --token YOUR_EVAL_TOKEN
```

To put it online, see `DEPLOY.md`. For data handling and limits, see `GOVERNANCE.md`.
