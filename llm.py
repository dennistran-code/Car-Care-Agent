"""Gemini client: one request at a time, with retries, a fallback model, and output caps."""
import json
import os
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")  # fast; set GEMINI_MODEL to change
CRITIC_MODEL = os.environ.get("GEMINI_CRITIC_MODEL", "gemini-flash-lite-latest")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-flash-latest")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "2048"))  # cost cap per model reply (includes any thinking)
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BUSY_CODES = {429, 500, 503}
RETRY_WAITS = [2, 5, 10]
REQUEST_TIMEOUT = (10, 45)  # seconds: connect, then wait for Gemini's reply


def log(msg):
    """Progress line in the server window. Never includes what the owner typed."""
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LLMError(Exception):
    pass


class BusyError(LLMError):
    pass


class SlowError(BusyError):
    """The model didn't answer in time. Waiting again on the same model rarely helps."""


class DailyQuotaError(BusyError):
    """The free tier's per-day limit for this model is used up. Retrying won't help today."""


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        path = os.path.join(HERE, "key.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                key = f.read().strip()
    if not key or key.startswith("PASTE"):
        raise LLMError("No Gemini API key found. Set GEMINI_API_KEY, or paste the key into key.txt.")
    return key


def _once(api_key, model, system, contents, tools, json_mode):
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
            "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": 0.2}}
    if tools:
        body["tools"] = [{"functionDeclarations": tools}]
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    try:
        r = requests.post(URL.format(model=model), params={"key": api_key}, json=body, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        raise SlowError("Gemini didn't answer within 45 seconds.")
    except requests.RequestException as e:
        raise BusyError(f"Could not reach Gemini: {e}")
    if r.status_code != 200:
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except ValueError:
            msg = r.text
        low = str(msg).lower()
        if r.status_code == 429 and ("per day" in low or "perday" in low or "daily" in low):
            raise DailyQuotaError(f"Gemini daily limit reached for {model}.")
        raise (BusyError if r.status_code in BUSY_CODES else LLMError)(
            f"Gemini error {r.status_code}: {str(msg)[:300]}")
    data = r.json()
    cands = data.get("candidates") or []
    if not cands or "content" not in cands[0]:
        reason = cands[0].get("finishReason") if cands else data.get("promptFeedback", {}).get("blockReason")
        raise LLMError(f"Gemini gave no usable reply (reason: {reason or 'none'}).")
    content = cands[0]["content"]
    content.setdefault("role", "model")
    content.setdefault("parts", [])
    if not content["parts"] and cands[0].get("finishReason") == "MAX_TOKENS":
        raise LLMError("The model hit the output-token cap before answering. Try a shorter question.")
    return content


class Gemini:
    """call(system, contents, tools=None, model=None, json_mode=False) -> (content, model_used)"""

    def __init__(self, api_key=None, sleep=time.sleep):
        self.api_key = api_key or load_api_key()
        self.sleep = sleep
        self.calls = 0

    def call(self, system, contents, tools=None, model=None, json_mode=False, deadline=None):
        model = model or MAIN_MODEL
        chain = [model] + ([FALLBACK_MODEL] if FALLBACK_MODEL != model else [])
        last, daily = None, 0
        for m in chain:
            for attempt in range(len(RETRY_WAITS) + 1):
                if deadline and time.monotonic() > deadline:
                    raise LLMError("This is taking too long (Google's AI is slow or busy right now). "
                                   "Try again in a minute.")
                t0 = time.monotonic()
                try:
                    self.calls += 1
                    out = _once(self.api_key, m, system, contents, tools, json_mode)
                    log(f"Gemini {m}: ok in {time.monotonic() - t0:.1f}s")
                    return out, m
                except DailyQuotaError as e:
                    log(f"Gemini {m}: daily limit reached")
                    last, daily = e, daily + 1
                    break  # no point retrying this model today; try the fallback model
                except SlowError as e:
                    log(f"Gemini {m}: no answer in 45s, switching model")
                    last = e
                    break
                except BusyError as e:
                    log(f"Gemini {m}: {str(e)[:120]} (after {time.monotonic() - t0:.1f}s)")
                    last = e
                    if attempt < len(RETRY_WAITS):
                        self.sleep(RETRY_WAITS[attempt])
                except LLMError as e:
                    log(f"Gemini {m}: {str(e)[:160]}")
                    raise
        if daily == len(chain):
            raise LLMError("The agent used up its free Google Gemini allowance for today. "
                           "It resets at midnight Pacific time (3 a.m. Eastern).")
        raise LLMError(f"{last} Google's servers stayed busy. Wait a minute and try again.")


def text_of(content):
    return "\n".join(p["text"] for p in content.get("parts", []) if p.get("text") and not p.get("thought")).strip()


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1]) if start >= 0 else {}
