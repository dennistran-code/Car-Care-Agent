# Put the Car Care Agent online

## Option A (recommended): GitHub + Render, free, stays up without your laptop

You get a permanent link like `https://car-care-agent.onrender.com`. Neither service needs a credit card.

### 1. Put the code on GitHub
1. Make a free account at github.com.
2. Click **+** (top right), then **New repository**. Name it `car-care-agent`, keep it **Public**, and click **Create repository**.
3. Click **uploading an existing file**.
4. Unzip `car_care_deploy.zip`. Select everything inside it, including the `static`, `eval` and `tests` folders, and drag it into the page.
   - The zip has **no key.txt** on purpose. Never upload your key.
5. Click **Commit changes**. Check that `server.py` and `requirements.txt` sit at the top level of the repo, not inside another folder.

### 2. Run it on Render
1. Go to render.com and sign up with **GitHub**.
2. Click **New**, then **Web Service**, and pick `car-care-agent`.
3. Fill in:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python server.py`
   - **Instance Type:** Free
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY` = a **new** key from aistudio.google.com (make a fresh one, since the old one was in a screenshot)
   - `EVAL_TOKEN` = any long random word you make up. Only your eval script uses it to skip the per-visitor limit.
5. Click **Create Web Service**. When the log says **Live**, your link is at the top.

### 3. Test that it's really public
Open the link on your phone with Wi-Fi off. If it loads on cellular data, it's public.

### Keeping it open for class
- Render's free plan sleeps after 15 minutes with no visitors, and the next visit takes about a minute to wake it.
- Open the link about 5 minutes before class and before you submit.
- **Optional, keeps it awake all the time:** make a free account at uptimerobot.com, add an HTTP monitor for `https://YOUR-LINK.onrender.com/healthz` every 5 minutes, and keep it on through Week 12. One always-on service fits inside Render's 750 free hours a month.

### Updating
Upload the changed file to the same GitHub repo. Render redeploys on its own in a few minutes.

## Option B (backup): run it on your Mac and share it through a tunnel

No accounts needed, but your Mac has to stay on and awake, and the link changes every time you restart.

```
brew install cloudflared          # one time (needs Homebrew)
python3 server.py                 # window 1
cloudflared tunnel --url http://localhost:8000    # window 2: prints https://something.trycloudflare.com
caffeinate -i                     # window 3: keeps the Mac awake
```
Share the `trycloudflare.com` link. Use this only if Render is down on class day.

## If something goes wrong
- **Build failed:** `requirements.txt` isn't at the top level of the repo.
- **"No Gemini API key found":** the variable name must be exactly `GEMINI_API_KEY`.
- **"Google's servers stayed busy":** Gemini's free tier is overloaded or you hit its limit. Wait a minute.
- **Anything else:** copy the error from Render's **Logs** tab into the chat.
