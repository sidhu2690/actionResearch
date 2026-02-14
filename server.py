#!/usr/bin/env python3
"""Colloquium — Multi-Agent AI Research Roundtable."""

import json, os, re, random, time, queue, threading, uuid
import urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from flask import Flask, Response, request, jsonify

# ━━ CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOOT        = time.time()
MAX_UP      = 21300
AGENT_GAP   = 28
USER_WAIT   = 5
MODEL       = "llama-3.1-8b-instant"
BACKUP      = "meta-llama/llama-4-scout-17b-16e-instruct"
PORT        = 8080
MAX_PAPERS  = 15

ARXIV_CATS = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML",
    "cs.NE", "cs.RO", "cs.CR", "math.OC", "physics.comp-ph",
]

AGENTS = [
    {
        "name": "Euler", "role": "Mathematical Analyst", "avatar": "📐",
        "color": "#ff9800",
        "personality": "Rigorous, precise, obsessed with formal correctness and elegance",
        "style": "Checks derivations, questions mathematical assumptions, suggests generalizations",
        "focus": "equations, proofs, formal consistency, mathematical generalization",
    },
    {
        "name": "Feynman", "role": "Physics & Intuition", "avatar": "⚛️",
        "color": "#03a9f4",
        "personality": "Curious, intuitive, loves analogies, hates unnecessary complexity",
        "style": "Checks physical plausibility, connects to known theories, simplifies",
        "focus": "physical plausibility, grounding, intuition, simplification",
    },
    {
        "name": "Hinton", "role": "ML Architect", "avatar": "🧠",
        "color": "#9c27b0",
        "personality": "Deep thinker about learning, skeptical of hype, practical",
        "style": "Questions model capacity, architecture, worries about generalization",
        "focus": "architecture, training, generalization, scalability, compute",
    },
    {
        "name": "Popper", "role": "Critical Reviewer", "avatar": "🧩",
        "color": "#f44336",
        "personality": "Relentlessly critical, demands evidence, hates overstatement",
        "style": "Finds weak claims, demands ablation, checks experimental validity",
        "focus": "claims vs evidence, reproducibility, experimental design, rigor",
    },
    {
        "name": "Tesla", "role": "Speculative Visionary", "avatar": "🔮",
        "color": "#4caf50",
        "personality": "Wild imagination grounded in technical understanding",
        "style": "Proposes bold extensions, cross-field connections, future applications",
        "focus": "novel extensions, cross-domain ideas, future implications",
    },
]

PHASES = [
    {
        "name": "SUMMARY", "label": "📖 Summary Phase",
        "desc": "Each agent summarizes the paper from their perspective",
        "rounds": 5,
        "prompt": 'Summarize this paper from YOUR perspective as {role}. What stands out? Core idea? Focus on {focus}. Under 100 words.',
    },
    {
        "name": "CRITIQUE", "label": "🔍 Critique Phase",
        "desc": "Each agent critiques from their domain",
        "rounds": 5,
        "prompt": 'Critique this paper as {role}. Weaknesses? Missing pieces? Bad assumptions? Be specific, technical. Reference others if useful. Under 100 words.',
    },
    {
        "name": "CROSSFIRE", "label": "⚔️ Cross-Examination",
        "desc": "Agents challenge each other directly",
        "rounds": 10,
        "prompt": 'Respond to {target}\'s point — agree, disagree, or push deeper. Challenge them. Stay in character as {role}. Under 90 words.',
    },
    {
        "name": "PROPOSAL", "label": "💡 Proposal Phase",
        "desc": "Propose extensions, experiments, new directions",
        "rounds": 5,
        "prompt": 'Propose ONE concrete extension, experiment, or improvement based on this paper. Be specific — what would you build or test? Think as {role} ({focus}). Under 100 words.',
    },
    {
        "name": "VERDICT", "label": "🏛️ Verdict Phase",
        "desc": "Final ratings and takeaways",
        "rounds": 5,
        "prompt": 'Final verdict. Key strength, key weakness, significance rating 1-10. As {role}, your bottom line. Under 80 words.',
    },
]

USER_COLORS = [
    "#ff9800", "#e91e63", "#9c27b0", "#03a9f4",
    "#4caf50", "#ff5722", "#00bcd4", "#cddc39",
]

FALLBACK_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
        "authors": "Vaswani, Shazeer, Parmar et al.",
        "link": "https://arxiv.org/abs/1706.03762",
        "categories": ["cs.CL", "cs.LG"],
    },
    {
        "title": "Scaling Laws for Neural Language Models",
        "abstract": "We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude.",
        "authors": "Kaplan, McCandlish, Henighan et al.",
        "link": "https://arxiv.org/abs/2001.08361",
        "categories": ["cs.LG", "cs.CL"],
    },
    {
        "title": "Denoising Diffusion Probabilistic Models",
        "abstract": "We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics.",
        "authors": "Ho, Jain, Abbeel",
        "link": "https://arxiv.org/abs/2006.11239",
        "categories": ["cs.LG", "stat.ML"],
    },
]

# ━━ STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
state = {
    "paper": None, "paper_num": 0,
    "phase": None, "phase_idx": -1,
    "agents": AGENTS, "messages": [],
    "typing": None, "total_papers": MAX_PAPERS,
}
users = {}
color_idx = [0]
user_queue = queue.Queue()

# ━━ SSE BUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Bus:
    def __init__(self):
        self._q, self._lock = [], threading.Lock()

    def listen(self):
        q = queue.Queue(maxsize=400)
        with self._lock:
            self._q.append(q)
        return q

    def drop(self, q):
        with self._lock:
            try: self._q.remove(q)
            except: pass

    @property
    def viewers(self):
        with self._lock:
            return len(self._q)

    def emit(self, ev, data):
        m = f"event: {ev}\ndata: {json.dumps(data)}\n\n"
        dead = []
        with self._lock:
            for q in self._q:
                try: q.put_nowait(m)
                except queue.Full: dead.append(q)
            for q in dead:
                try: self._q.remove(q)
                except: pass

bus = Bus()

# ━━ GROQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_client = None

def groq():
    global _client
    if not _client:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

def llm(system, history, instruction, model=MODEL):
    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-20:])
    msgs.append({"role": "user", "content": instruction})
    try:
        r = groq().chat.completions.create(
            model=model, messages=msgs,
            temperature=0.8, max_tokens=200,
        )
        t = r.choices[0].message.content.strip()
        t = re.sub(r'^[\w]+\s*[:—\-]\s*', '', t)
        return t.strip('"\'')
    except Exception as e:
        if model == MODEL:
            return llm(system, history, instruction, BACKUP)
        raise

def tleft():
    return max(0, int(MAX_UP - (time.time() - BOOT)))

def now_hm():
    return datetime.now(timezone.utc).strftime("%H:%M")

# ━━ ARXIV ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_papers():
    papers = []
    cats = random.sample(ARXIV_CATS, min(5, len(ARXIV_CATS)))
    q = "+OR+".join(f"cat:{c}" for c in cats)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={q}&sortBy=submittedDate"
        f"&sortOrder=descending&max_results=40"
    )
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Colloquium/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ns):
            title = re.sub(r'\s+', ' ', entry.find("a:title", ns).text.strip())
            abstract = re.sub(r'\s+', ' ', entry.find("a:summary", ns).text.strip())
            authors = [a.find("a:name", ns).text for a in entry.findall("a:author", ns)]
            link = entry.find("a:id", ns).text
            for l in entry.findall("a:link", ns):
                if l.get("title") == "pdf":
                    link = l.get("href", link)
                    break
            categories = [c.get("term", "") for c in entry.findall("a:category", ns)]
            if len(abstract) > 120:
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors[:4]) + ("..." if len(authors) > 4 else ""),
                    "link": link,
                    "categories": categories[:5],
                })
    except Exception as e:
        print(f"  ⚠ arXiv fetch: {e}")
    return papers

# ━━ STREAM AI MESSAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stream_ai(agent, text, history):
    words = text.split()
    wps = max(0.04, min(14 / max(len(words), 1), 0.35))
    bus.emit("msgstart", {
        "speaker": agent["name"], "avatar": agent["avatar"],
        "color": agent["color"], "role": agent["role"],
        "time": now_hm(), "is_ai": True,
    })
    for i, w in enumerate(words):
        bus.emit("word", {"w": w, "i": i, "of": len(words)})
        time.sleep(wps)
    msg = {
        "type": "message", "speaker": agent["name"],
        "avatar": agent["avatar"], "color": agent["color"],
        "role": agent["role"], "text": text, "time": now_hm(),
    }
    state["messages"].append(msg)
    state["typing"] = None
    history.append({"role": "assistant", "content": f"[{agent['name']}]: {text}"})
    bus.emit("msgdone", {"speaker": agent["name"], "text": text, "time": now_hm()})
    print(f"    {agent['avatar']} {agent['name']}: {text[:72]}...")

# ━━ HANDLE USER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def handle_user(paper, history):
    try:
        umsg = user_queue.get_nowait()
    except queue.Empty:
        return False
    while not user_queue.empty():
        try: user_queue.get_nowait()
        except: break
    time.sleep(random.uniform(2, USER_WAIT))
    agent = random.choice(AGENTS)
    recent = []
    for m in state["messages"][-8:]:
        if m.get("type") == "user":
            recent.append(f'{m["user_name"]}: {m["text"]}')
        elif m.get("type") == "message":
            recent.append(f'{m["speaker"]}: {m["text"]}')
    ctx = "\n".join(recent[-5:])
    system = (
        f"You are {agent['name']} — {agent['role']}.\n"
        f"Personality: {agent['personality']}. Style: {agent['style']}.\n"
        f'Roundtable discussing: "{paper["title"]}"\n'
        f"Abstract: {paper['abstract'][:400]}\n"
        f"A human spoke. Respond warmly, in character, use their name. Under 80 words."
    )
    inst = f"Recent:\n{ctx}\n\nRespond to the human. Under 80 words."
    state["typing"] = agent["name"]
    bus.emit("typing", {
        "name": agent["name"], "avatar": agent["avatar"],
        "color": agent["color"], "role": agent["role"],
    })
    try:
        text = llm(system, history, inst)
        stream_ai(agent, text, history)
    except Exception as e:
        print(f"  ✖ user-reply: {e}")
        state["typing"] = None
    return True

# ━━ ENGINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def engine():
    time.sleep(1.5)

    print("\n🔬 Fetching papers from arXiv...")
    papers = fetch_papers()
    if len(papers) < MAX_PAPERS:
        print(f"  ⚠ Only got {len(papers)}, adding fallbacks")
        papers.extend(FALLBACK_PAPERS)
    random.shuffle(papers)
    papers = papers[:MAX_PAPERS]
    print(f"  ✓ {len(papers)} papers queued\n")

    for pidx, paper in enumerate(papers):
        if tleft() < 120:
            break

        state["paper"] = paper
        state["paper_num"] = pidx + 1

        print(f"\n{'='*58}")
        print(f"  📄 [{pidx+1}/{len(papers)}] {paper['title'][:50]}...")
        print(f"     {paper['authors']}")
        print(f"     {', '.join(paper['categories'][:3])}")
        print(f"     ⏱ {tleft()//60}m remaining")
        print(f"{'='*58}")

        pmsg = {
            "type": "paper", "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": paper["authors"], "link": paper["link"],
            "categories": paper["categories"],
            "number": pidx + 1, "total": len(papers),
            "time": now_hm(),
        }
        state["messages"].append(pmsg)
        bus.emit("newpaper", pmsg)

        history = [{"role": "user", "content":
            f'PAPER: "{paper["title"]}"\n'
            f'Authors: {paper["authors"]}\n'
            f'Abstract: {paper["abstract"]}'
        }]

        for phi, phase in enumerate(PHASES):
            if tleft() < 90:
                break

            state["phase"] = phase["name"]
            state["phase_idx"] = phi

            ph_msg = {
                "type": "phase", "name": phase["name"],
                "label": phase["label"], "desc": phase["desc"],
                "time": now_hm(),
            }
            state["messages"].append(ph_msg)
            bus.emit("newphase", ph_msg)
            print(f"\n    {phase['label']}")
            time.sleep(3)

            for rnd in range(phase["rounds"]):
                if tleft() < 60:
                    break

                # check user messages first
                if handle_user(paper, history):
                    continue

                agent = AGENTS[rnd % len(AGENTS)]

                # pick crossfire target
                target = None
                if phase["name"] == "CROSSFIRE":
                    others = [a for a in AGENTS if a["name"] != agent["name"]]
                    for m in reversed(state["messages"][-8:]):
                        if m.get("type") == "message" and m["speaker"] != agent["name"]:
                            target = next((a for a in others if a["name"] == m["speaker"]), None)
                            break
                    if not target:
                        target = random.choice(others)

                state["typing"] = agent["name"]
                bus.emit("typing", {
                    "name": agent["name"], "avatar": agent["avatar"],
                    "color": agent["color"], "role": agent["role"],
                })

                system = (
                    f"You are {agent['name']} — {agent['role']}.\n"
                    f"Personality: {agent['personality']}\n"
                    f"Style: {agent['style']}\nFocus: {agent['focus']}\n\n"
                    f'Research roundtable analyzing: "{paper["title"]}"\n'
                    f"Abstract: {paper['abstract'][:500]}\n\n"
                    f"Phase: {phase['label']} — {phase['desc']}\n"
                    f"{'Humans are watching. Acknowledge them if relevant.' if users else ''}\n"
                    f"Be substantive, specific, in character. Don't start with your name."
                )

                inst = phase["prompt"].format(
                    role=agent["role"],
                    focus=agent["focus"],
                    target=target["name"] if target else "",
                )
                inst = f'Paper: "{paper["title"]}"\n{inst}'

                if phase["name"] in ("CROSSFIRE", "PROPOSAL", "VERDICT"):
                    recent = [
                        f'{m["speaker"]}: {m["text"]}'
                        for m in state["messages"][-8:]
                        if m.get("type") == "message"
                    ]
                    if recent:
                        inst += "\n\nRecent:\n" + "\n".join(recent[-5:])

                for m in reversed(state["messages"][-10:]):
                    if m.get("type") == "user":
                        if random.random() < 0.25:
                            inst += f'\n(Human {m["user_name"]} said: "{m["text"]}" — weave in if relevant.)'
                        break

                try:
                    text = llm(system, history, inst)
                    stream_ai(agent, text, history)
                except Exception as e:
                    print(f"    ✖ {e}")
                    state["typing"] = None

                if rnd < phase["rounds"] - 1:
                    nxt = AGENTS[(rnd + 1) % len(AGENTS)]
                    bus.emit("waiting", {
                        "name": nxt["name"], "avatar": nxt["avatar"],
                        "color": nxt["color"], "gap": AGENT_GAP,
                        "timeleft": tleft(),
                    })
                    deadline = time.time() + AGENT_GAP
                    while time.time() < deadline and tleft() > 60:
                        try:
                            u = user_queue.get(timeout=1)
                            user_queue.put(u)
                            break
                        except queue.Empty:
                            pass

            time.sleep(2)

        # paper done
        if tleft() > 60:
            dmsg = {
                "type": "system",
                "text": f"✅ Paper #{pidx+1} analysis complete — {len(papers)-pidx-1} remaining",
                "time": now_hm(),
            }
            state["messages"].append(dmsg)
            bus.emit("system", dmsg)
            time.sleep(4)

    # session end
    cnt = len([m for m in state["messages"] if m.get("type") == "message"])
    ucnt = len([m for m in state["messages"] if m.get("type") == "user"])
    bus.emit("shutdown", {
        "total_msgs": cnt, "total_papers": state["paper_num"],
        "user_msgs": ucnt, "users": len(users),
    })
    print(f"\n⏰ Done. {cnt} agent msgs · {ucnt} human · {state['paper_num']} papers.")

# ━━ FLASK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = Flask(__name__)

@app.route("/join", methods=["POST"])
def join():
    data = request.json or {}
    name = (data.get("name") or "").strip()[:20]
    if not name:
        return jsonify({"error": "name required"}), 400
    uid = str(uuid.uuid4())[:8]
    color = USER_COLORS[color_idx[0] % len(USER_COLORS)]
    color_idx[0] += 1
    users[uid] = {"name": name, "color": color}
    sysmsg = {"type": "system", "text": f"👋 {name} joined the roundtable", "time": now_hm()}
    state["messages"].append(sysmsg)
    bus.emit("system", sysmsg)
    bus.emit("presence", {"users": list(users.values()), "viewers": bus.viewers})
    print(f"  👋 {name} joined ({uid})")
    return jsonify({"id": uid, "name": name, "color": color})

@app.route("/send", methods=["POST"])
def send():
    data = request.json or {}
    uid = data.get("id", "")
    text = (data.get("text") or "").strip()[:500]
    if uid not in users:
        return jsonify({"error": "not joined"}), 403
    if not text:
        return jsonify({"error": "empty"}), 400
    user = users[uid]
    msg = {
        "type": "user", "user_id": uid, "user_name": user["name"],
        "color": user["color"], "text": text, "time": now_hm(),
    }
    state["messages"].append(msg)
    bus.emit("usermsg", msg)
    user_queue.put(msg)
    print(f"  💬 {user['name']}: {text[:60]}")
    return jsonify({"ok": True})

@app.route("/stream")
def stream():
    q = bus.listen()
    def gen():
        init = {
            "paper": state["paper"],
            "paper_num": state["paper_num"],
            "total_papers": MAX_PAPERS,
            "phase": state["phase"],
            "phase_idx": state["phase_idx"],
            "agents": AGENTS,
            "messages": state["messages"][-150:],
            "typing": state["typing"],
            "boot": BOOT, "max_up": MAX_UP,
            "timeleft": tleft(),
            "users": list(users.values()),
            "viewers": bus.viewers,
        }
        yield f"event: fullstate\ndata: {json.dumps(init)}\n\n"
        try:
            while True:
                try:
                    yield q.get(timeout=25)
                except queue.Empty:
                    yield f"event: ping\ndata: {json.dumps({'tl':tleft(),'v':bus.viewers})}\n\n"
        except GeneratorExit:
            bus.drop(q)
    return Response(gen(), content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/")
def index():
    return HTML

# ━━ HTML ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Colloquium — AI Research Roundtable</title>
<style>
:root{
  --bg:#0a0e14;--pnl:#111820;--card:#161d28;--brd:#1e2a38;
  --tx:#cdd6e2;--tx2:#5a6d82;--acc:#00e5a0;--acc2:#00b8d4;
  --warn:#ff6b6b;--hi:#14293f;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{
  font-family:'SF Mono','Fira Code',Consolas,'Courier New',monospace;
  background:var(--bg);color:var(--tx);display:flex;justify-content:center;
}
.overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.92);
  display:flex;align-items:center;justify-content:center;
  z-index:100;backdrop-filter:blur(12px);
}
.overlay.hidden{display:none}
.jc{
  background:var(--pnl);border:1px solid var(--brd);
  border-radius:12px;padding:2rem 1.6rem;
  width:92%;max-width:400px;text-align:center;
}
.jc h1{font-size:1.4rem;margin-bottom:.2rem;font-weight:700}
.jc .ac{color:var(--acc)}
.jc .sub{color:var(--tx2);font-size:.7rem;margin-bottom:1.2rem;line-height:1.6}
.jc input{
  width:100%;padding:.7rem .9rem;background:var(--bg);
  border:1px solid var(--brd);border-radius:6px;
  color:var(--tx);font-size:.82rem;font-family:inherit;
  outline:none;margin-bottom:.6rem;
}
.jc input:focus{border-color:var(--acc)}
.jc input::placeholder{color:var(--tx2)}
.btn{
  width:100%;padding:.7rem;border:none;border-radius:6px;
  font-size:.78rem;font-weight:600;cursor:pointer;
  font-family:inherit;margin-bottom:.35rem;
}
.btn:hover{opacity:.85}
.btn-go{background:var(--acc);color:#000}
.btn-w{background:transparent;color:var(--tx2);border:1px solid var(--brd)}
.jc .ht{color:var(--tx2);font-size:.58rem;margin-top:.8rem;line-height:1.7}
.app{
  width:100%;max-width:580px;height:100vh;height:100dvh;
  display:flex;flex-direction:column;background:var(--bg);position:relative;
}
.hdr{
  display:flex;align-items:center;gap:.5rem;
  padding:.5rem .7rem;background:var(--pnl);
  border-bottom:1px solid var(--brd);min-height:52px;z-index:10;
}
.hdr-ico{font-size:1.2rem}
.hdr-info{flex:1;min-width:0}
.hdr-name{font-size:.82rem;font-weight:700;letter-spacing:.03em}
.hdr-sub{font-size:.62rem;color:var(--tx2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hdr-sub .typ{color:var(--acc)}
.hdr-badges{display:flex;gap:.2rem}
.bdg{padding:.12rem .4rem;border-radius:3px;font-weight:600;font-size:.56rem;letter-spacing:.02em}
.b-eye{background:rgba(0,184,212,.1);color:var(--acc2)}
.b-msg{background:rgba(0,229,160,.1);color:var(--acc)}
.b-die{background:rgba(255,107,107,.1);color:var(--warn)}
.b-ppr{background:rgba(255,152,0,.1);color:#ff9800}
.pbar{
  display:flex;gap:2px;padding:.2rem .7rem;
  background:rgba(0,0,0,.2);border-bottom:1px solid var(--brd);
}
.pbar .seg{flex:1;height:3px;border-radius:2px;background:var(--brd);transition:background .3s}
.pbar .seg.done{background:var(--acc)}
.pbar .seg.on{background:var(--acc);animation:pls 1.5s ease infinite}
@keyframes pls{0%,100%{opacity:1}50%{opacity:.35}}
.astrip{
  display:flex;gap:.2rem;padding:.3rem .6rem;
  background:rgba(0,0,0,.15);border-bottom:1px solid var(--brd);
  overflow-x:auto;font-size:.6rem;
}
.astrip::-webkit-scrollbar{display:none}
.ach{
  display:flex;align-items:center;gap:.2rem;
  padding:.12rem .4rem;border-radius:3px;
  background:rgba(255,255,255,.03);border:1px solid var(--brd);
  white-space:nowrap;flex-shrink:0;
}
.ach .dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.ach .rl{color:var(--tx2);font-size:.48rem}
.chat{flex:1;overflow-y:auto;overflow-x:hidden;padding:.5rem .6rem}
.chat::-webkit-scrollbar{width:3px}
.chat::-webkit-scrollbar-thumb{background:var(--brd);border-radius:3px}
.sys{text-align:center;margin:.6rem 0}
.pill{
  display:inline-block;background:var(--card);color:var(--tx2);
  padding:.3rem .7rem;border-radius:4px;font-size:.66rem;
  max-width:94%;line-height:1.5;border:1px solid var(--brd);
}
.pill.ph{
  color:var(--acc);font-weight:600;font-size:.7rem;
  border-color:rgba(0,229,160,.2);background:rgba(0,229,160,.04);
  text-align:left;
}
.pill.ph .pd{color:var(--tx2);font-weight:400;font-size:.58rem;margin-top:2px}
.pc{
  background:var(--card);border:1px solid var(--brd);
  border-radius:8px;padding:.7rem .8rem;margin:.6rem 0;
  font-size:.7rem;line-height:1.5;
}
.pc .pn{color:var(--acc);font-weight:700;font-size:.58rem;letter-spacing:.05em;margin-bottom:.3rem}
.pc .pt{color:var(--tx);font-weight:600;font-size:.78rem;margin-bottom:.3rem;line-height:1.4}
.pc .pa{color:var(--acc2);font-size:.6rem;margin-bottom:.4rem}
.pc .pabs{color:var(--tx2);font-size:.62rem;line-height:1.6;max-height:72px;overflow:hidden;transition:max-height .3s}
.pc .pabs.open{max-height:600px}
.pc .ptog{color:var(--acc);font-size:.58rem;cursor:pointer;margin-top:.3rem;display:inline-block}
.pc .ptog:hover{text-decoration:underline}
.pc .pcats{display:flex;gap:.2rem;margin-top:.4rem;flex-wrap:wrap}
.pc .pcat{padding:.08rem .3rem;border-radius:2px;font-size:.52rem;background:rgba(0,184,212,.08);color:var(--acc2);border:1px solid rgba(0,184,212,.12)}
.pc .plnk{display:inline-block;margin-top:.4rem;font-size:.56rem;color:var(--acc);text-decoration:none}
.pc .plnk:hover{text-decoration:underline}
.msg{display:flex;flex-direction:column;margin-bottom:2px;animation:up .2s ease}
@keyframes up{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.msg.ai{align-items:flex-start;padding-right:2.5rem}
.msg.usr{align-items:flex-end;padding-left:2.5rem}
.msg .body{
  position:relative;padding:.4rem .6rem .25rem;
  border-radius:6px;max-width:100%;font-size:.76rem;
  line-height:1.45;word-wrap:break-word;
}
.msg.ai .body{background:var(--card);border:1px solid var(--brd);border-top-left-radius:0}
.msg.usr .body{background:var(--hi);border:1px solid rgba(20,41,63,.6);border-top-right-radius:0}
.msg.cont .body{border-radius:6px}
.msg.cont{margin-top:1px}
.msg .who{font-size:.66rem;font-weight:600;margin-bottom:2px;display:flex;align-items:center;gap:.3rem}
.msg .who .rl{font-size:.52rem;font-weight:400;color:var(--tx2)}
.msg .txt{color:var(--tx)}
.msg .meta{
  float:right;display:flex;align-items:center;gap:3px;
  margin-left:8px;margin-top:3px;font-size:.52rem;
  color:rgba(255,255,255,.25);white-space:nowrap;
}
.msg .spacer{display:inline-block;width:3.5rem;height:1px}
.cursor{
  display:inline-block;width:2px;height:.82em;background:var(--acc);
  margin-left:1px;animation:bk .7s step-end infinite;vertical-align:text-bottom;
}
@keyframes bk{0%,100%{opacity:1}50%{opacity:0}}
.tdots{display:inline-flex;gap:3px;align-items:center;padding:4px 0}
.tdots span{width:5px;height:5px;border-radius:50%;background:var(--tx2);animation:dp 1.4s ease-in-out infinite}
.tdots span:nth-child(2){animation-delay:.2s}
.tdots span:nth-child(3){animation-delay:.4s}
@keyframes dp{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
.ibar{
  display:flex;align-items:center;gap:.4rem;
  padding:.4rem .6rem;background:var(--pnl);
  border-top:1px solid var(--brd);min-height:48px;
}
.ibar.off{opacity:.3;pointer-events:none}
.iwrap{
  flex:1;display:flex;align-items:center;background:var(--bg);
  border:1px solid var(--brd);border-radius:6px;padding:.1rem .2rem .1rem .7rem;
}
.iwrap:focus-within{border-color:var(--acc)}
.iwrap input{
  flex:1;background:none;border:none;outline:none;
  color:var(--tx);font-size:.78rem;padding:.5rem 0;font-family:inherit;
}
.iwrap input::placeholder{color:var(--tx2)}
.sbtn{
  width:36px;height:36px;border-radius:6px;border:none;
  background:var(--acc);color:#000;font-size:.95rem;font-weight:700;
  cursor:pointer;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;
}
.sbtn:hover{opacity:.85}
.sbtn:disabled{opacity:.2;cursor:default}
.wbar{
  display:flex;align-items:center;justify-content:center;
  padding:.4rem .7rem;background:var(--pnl);
  border-top:1px solid var(--brd);min-height:42px;
  font-size:.68rem;color:var(--tx2);gap:.4rem;
}
.wbar .live{color:var(--acc);font-weight:700}
.wbar .jl{color:var(--acc);cursor:pointer;text-decoration:underline;margin-left:.5rem}
.scb{
  display:none;position:absolute;bottom:60px;right:12px;
  width:34px;height:34px;background:var(--pnl);border:1px solid var(--brd);
  border-radius:6px;align-items:center;justify-content:center;
  cursor:pointer;z-index:20;box-shadow:0 2px 10px rgba(0,0,0,.5);
  font-size:.85rem;color:var(--tx2);
}
.scb:hover{background:var(--card)}
.scb .ub{
  position:absolute;top:-4px;right:-4px;background:var(--acc);
  color:#000;font-size:.5rem;font-weight:700;
  min-width:15px;height:15px;border-radius:3px;
  display:flex;align-items:center;justify-content:center;padding:0 3px;
}
.sd{
  text-align:center;padding:1rem .8rem;
  background:rgba(255,107,107,.05);border:1px solid rgba(255,107,107,.12);
  border-radius:6px;margin:.6rem 0;
}
.sd .big{color:var(--warn);font-size:.78rem;font-weight:700}
.sd .sm{color:var(--tx2);font-size:.6rem;margin-top:.2rem}
@media(max-width:580px){.app{max-width:100%}}
@media(min-width:581px){body{align-items:center;padding:1rem 0}.app{border-radius:10px;height:96vh;overflow:hidden;border:1px solid var(--brd)}}
</style>
</head><body>

<div class="overlay" id="ov">
  <div class="jc">
    <h1>🔬 <span class="ac">Colloquium</span></h1>
    <div class="sub">
      Multi-agent AI research roundtable<br>
      5 specialist AIs analyze real arXiv papers live<br>
      15 papers per session · 5 phases each
    </div>
    <input type="text" id="nin" placeholder="Your name..." maxlength="20"
      onkeydown="if(event.key==='Enter')doJ()">
    <button class="btn btn-go" onclick="doJ()">Join Roundtable</button>
    <button class="btn btn-w" onclick="doW()">Just Observe</button>
    <div class="ht">
      📐 Euler · ⚛️ Feynman · 🧠 Hinton · 🧩 Popper · 🔮 Tesla<br>
      Summary → Critique → Cross-Exam → Proposals → Verdict
    </div>
  </div>
</div>

<div class="app" id="app" style="display:none">
  <div class="hdr">
    <div class="hdr-ico">🔬</div>
    <div class="hdr-info">
      <div class="hdr-name">Colloquium</div>
      <div class="hdr-sub" id="hsub">connecting...</div>
    </div>
    <div class="hdr-badges">
      <div class="bdg b-ppr" id="bppr">📄 0/15</div>
      <div class="bdg b-eye" id="beye">👁 0</div>
      <div class="bdg b-msg" id="bmsg">💬 0</div>
      <div class="bdg b-die" id="bdie">⏱ --</div>
    </div>
  </div>
  <div class="pbar" id="pbar">
    <div class="seg" data-p="0"></div>
    <div class="seg" data-p="1"></div>
    <div class="seg" data-p="2"></div>
    <div class="seg" data-p="3"></div>
    <div class="seg" data-p="4"></div>
  </div>
  <div class="astrip" id="ags"></div>
  <div class="chat" id="chat"></div>
  <div class="ibar" id="ibar" style="display:none">
    <div class="iwrap">
      <input type="text" id="min" placeholder="Ask a question..." maxlength="500"
        onkeydown="if(event.key==='Enter')doS()">
    </div>
    <button class="sbtn" onclick="doS()">➤</button>
  </div>
  <div class="wbar" id="wbar">
    <span class="live">● LIVE</span>
    <span id="wtxt">starting...</span>
    <span class="jl" onclick="showJ()">join</span>
  </div>
  <div class="scb" id="scb" onclick="jmpB()">↓<div class="ub" id="ub" style="display:none">0</div></div>
</div>

<script>
const $=id=>document.getElementById(id);
const chat=$('chat');
const PN=['SUMMARY','CRITIQUE','CROSSFIRE','PROPOSAL','VERDICT'];

let myId=null,myN=null,myC=null,joined=false;
let agents=[],boot=0,maxU=0,mc=0,lw='',tIv=null,cPh=-1;
let cBub=null,cTxt=null,tBub=null;
let sUp=false,miss=0;

function scr(){if(!sUp)chat.scrollTop=chat.scrollHeight}
function jmpB(){sUp=false;miss=0;chat.scrollTop=chat.scrollHeight;$('scb').style.display='none';$('ub').style.display='none'}
chat.addEventListener('scroll',()=>{
  const g=chat.scrollHeight-chat.scrollTop-chat.clientHeight;
  const w=sUp;sUp=g>80;
  if(!sUp&&w){miss=0;$('ub').style.display='none'}
  $('scb').style.display=sUp?'flex':'none';
});
function ntf(){if(sUp){miss++;$('ub').textContent=miss;$('ub').style.display='flex'}}

async function doJ(){
  const n=$('nin').value.trim();
  if(!n)return $('nin').focus();
  try{
    const r=await fetch('/join',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n})});
    const d=await r.json();
    if(d.error){alert(d.error);return}
    myId=d.id;myN=d.name;myC=d.color;joined=true;
    $('ov').classList.add('hidden');$('app').style.display='flex';
    $('ibar').style.display='flex';$('wbar').style.display='none';
    sse();setTimeout(()=>$('min').focus(),300);
  }catch(e){alert('Connection failed')}
}
function doW(){
  joined=false;$('ov').classList.add('hidden');$('app').style.display='flex';
  $('ibar').style.display='none';$('wbar').style.display='flex';sse();
}
function showJ(){$('ov').classList.remove('hidden');$('nin').focus()}

async function doS(){
  const inp=$('min'),t=inp.value.trim();
  if(!t||!myId)return;inp.value='';inp.focus();
  try{await fetch('/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:myId,text:t})})}catch(e){}
}

function fmt(s){
  s=Math.max(0,Math.floor(s));
  const h=Math.floor(s/3600),m=Math.floor(s%3600/60),sc=s%60;
  return h>0?h+'h '+String(m).padStart(2,'0')+'m':m+'m '+String(sc).padStart(2,'0')+'s';
}
function stmr(){
  if(tIv)clearInterval(tIv);
  tIv=setInterval(()=>{
    const l=Math.max(0,maxU-(Date.now()/1000-boot));
    $('bdie').textContent='⏱ '+fmt(l);
    if(l<300)$('bdie').style.background='rgba(255,107,107,.25)';
    if(l<=0){$('bdie').textContent='⏱ END';clearInterval(tIv)}
  },1000);
}
function sH(h){$('hsub').innerHTML=h}
function sW(h){$('wtxt').innerHTML=h}

function uPB(idx){
  cPh=idx;
  document.querySelectorAll('.seg').forEach((el,i)=>{
    el.classList.remove('done','on');
    if(i<idx)el.classList.add('done');
    else if(i===idx)el.classList.add('on');
  });
}

function rAgs(ul){
  let h='';
  agents.forEach(a=>{h+=`<div class="ach"><div class="dot" style="background:${a.color}"></div>${a.avatar} ${a.name} <span class="rl">${a.role}</span></div>`});
  (ul||[]).forEach(u=>{
    const me=(myN&&u.name===myN)?' (you)':'';
    h+=`<div class="ach"><div class="dot" style="background:${u.color}"></div>${u.name}${me}</div>`;
  });
  $('ags').innerHTML=h;
}

function sPill(h,c){
  const d=document.createElement('div');d.className='sys';
  d.innerHTML=`<span class="pill ${c||''}">${h}</span>`;chat.appendChild(d);scr();
}
function phPill(p){
  const d=document.createElement('div');d.className='sys';
  d.innerHTML=`<span class="pill ph">${p.label}<div class="pd">${p.desc}</div></span>`;
  chat.appendChild(d);scr();
}
function pCard(p){
  const id='a'+Date.now();
  const d=document.createElement('div');d.className='pc';
  d.innerHTML=`<div class="pn">PAPER #${p.number}${p.total?' / '+p.total:''} · ${p.time}</div>`+
    `<div class="pt">${p.title}</div>`+
    `<div class="pa">${p.authors}</div>`+
    `<div class="pabs" id="${id}">${p.abstract}</div>`+
    `<span class="ptog" onclick="var e=document.getElementById('${id}');if(e.classList.contains('open')){e.classList.remove('open');this.textContent='show more ▾'}else{e.classList.add('open');this.textContent='show less ▴'}">show more ▾</span>`+
    `<div class="pcats">${(p.categories||[]).map(c=>'<span class="pcat">'+c+'</span>').join('')}</div>`+
    (p.link?`<a class="plnk" href="${p.link}" target="_blank" rel="noopener">→ arXiv</a>`:'');
  chat.appendChild(d);scr();
}

function rmT(){if(tBub){tBub.remove();tBub=null}}
function addT(n,av,cl,rl){
  rmT();const co=(lw===n);
  const d=document.createElement('div');d.className=`msg ai${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${cl}">${av} ${n} <span class="rl">${rl}</span></div>`;
  h+=`<div class="body"><div class="tdots"><span></span><span></span><span></span></div></div>`;
  d.innerHTML=h;chat.appendChild(d);tBub=d;scr();
}
function sBub(sp,av,cl,rl,tm){
  rmT();const co=(lw===sp);
  const d=document.createElement('div');d.className=`msg ai${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${cl}">${av} ${sp} <span class="rl">${rl}</span></div>`;
  h+=`<div class="body"><span class="meta"><span class="tm">${tm}</span></span><span class="txt"></span><span class="cursor"></span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);cBub=d;cTxt=d.querySelector('.txt');scr();
}
function addWd(w){if(!cTxt)return;const t=cTxt.textContent;cTxt.textContent=t?(t+' '+w):w;scr()}
function fBub(sp,tm){
  if(cBub){const c=cBub.querySelector('.cursor');if(c)c.remove();const m=cBub.querySelector('.meta');if(m)m.innerHTML=`<span class="tm">${tm}</span>`}
  lw=sp;cBub=null;cTxt=null;mc++;$('bmsg').textContent='💬 '+mc;ntf();scr();
}

function fMsg(m){
  const w=m.speaker||m.user_name,isU=(m.type==='user'),co=(lw===w);
  const d=document.createElement('div');d.className=`msg ${isU?'usr':'ai'}${co?' cont':''}`;
  const cl=m.color||'#aaa',av=m.avatar||'',rl=m.role||'';
  let h='';
  if(!co){
    if(isU)h+=`<div class="who" style="color:${cl}">${w}</div>`;
    else h+=`<div class="who" style="color:${cl}">${av} ${w} <span class="rl">${rl}</span></div>`;
  }
  h+=`<div class="body"><span class="meta"><span class="tm">${m.time||''}</span></span><span class="txt">${m.text}</span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);lw=w;if(!isU)mc++;
}
function uBub(m){
  const me=(myN&&m.user_name===myN),co=(lw===m.user_name);
  const d=document.createElement('div');d.className=`msg usr${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${m.color}">${m.user_name}${me?' (you)':''}</div>`;
  h+=`<div class="body"><span class="meta"><span class="tm">${m.time||''}</span></span><span class="txt">${m.text}</span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);lw=m.user_name;scr();
}

function sse(){
  sH('connecting...');
  const es=new EventSource('/stream');

  es.addEventListener('fullstate',e=>{
    const d=JSON.parse(e.data);
    boot=d.boot;maxU=d.max_up;agents=d.agents||[];
    rAgs(d.users);
    $('beye').textContent='👁 '+(d.viewers||0);
    $('bppr').textContent='📄 '+(d.paper_num||0)+'/'+(d.total_papers||15);
    if(d.phase_idx>=0)uPB(d.phase_idx);
    sH(d.paper?d.paper.title.substring(0,48)+'...':'waiting for papers...');
    chat.innerHTML='';mc=0;lw='';
    sPill('🔬 <b>Colloquium</b> — AI Research Roundtable · '+agents.length+' agents · '+(d.total_papers||15)+' papers','');
    if(d.messages)d.messages.forEach(m=>{
      if(m.type==='paper')pCard(m);
      else if(m.type==='phase')phPill(m);
      else if(m.type==='message')fMsg(m);
      else if(m.type==='user')fMsg(m);
      else if(m.type==='system')sPill(m.text,'');
    });
    $('bmsg').textContent='💬 '+mc;stmr();scr();sW('analysis in progress');
  });

  es.addEventListener('newpaper',e=>{
    const d=JSON.parse(e.data);pCard(d);
    sH(d.title.substring(0,48)+'...');uPB(-1);
    $('bppr').textContent='📄 '+d.number+'/'+(d.total||15);
  });
  es.addEventListener('newphase',e=>{
    const d=JSON.parse(e.data);phPill(d);
    const i=PN.indexOf(d.name);if(i>=0)uPB(i);
  });
  es.addEventListener('typing',e=>{
    const d=JSON.parse(e.data);addT(d.name,d.avatar,d.color,d.role);
    sH(`<span class="typ">${d.avatar} ${d.name} thinking...</span>`);
    sW(`${d.avatar} ${d.name} analyzing...`);
  });
  es.addEventListener('msgstart',e=>{
    const d=JSON.parse(e.data);sBub(d.speaker,d.avatar,d.color,d.role,d.time);
    sH(`<span class="typ">${d.avatar} ${d.speaker} writing...</span>`);
    sW(`${d.avatar} ${d.speaker} writing...`);
  });
  es.addEventListener('word',e=>{addWd(JSON.parse(e.data).w)});
  es.addEventListener('msgdone',e=>{
    const d=JSON.parse(e.data);fBub(d.speaker,d.time);
    const t=document.querySelector('.pc:last-of-type .pt');
    sH(t?t.textContent.substring(0,48)+'...':'roundtable');
    sW('analysis in progress');
  });
  es.addEventListener('usermsg',e=>{uBub(JSON.parse(e.data))});
  es.addEventListener('system',e=>{sPill(JSON.parse(e.data).text,'')});
  es.addEventListener('presence',e=>{
    const d=JSON.parse(e.data);rAgs(d.users);
    $('beye').textContent='👁 '+(d.viewers||0);
  });
   es.addEventListener('waiting',e=>{
    const d=JSON.parse(e.data);let g=d.gap;
    sW(`${d.avatar} <span style="color:${d.color}">${d.name}</span> in <span id="gcd">${g}s</span>`);
    const iv=setInterval(()=>{g--;const el=document.getElementById('gcd');if(el)el.textContent=g+'s';if(g<=0){clearInterval(iv);sW('next agent...')}
    },1000);
  });
  es.addEventListener('shutdown',e=>{
    const d=JSON.parse(e.data);rmT();
    const div=document.createElement('div');div.className='sd';
    div.innerHTML=`<div class="big">⏱ Session Complete</div>`+
      `<div class="sm">Next cycle starts on schedule</div>`+
      `<div class="sm">${d.total_msgs} agent messages · ${d.user_msgs} human · ${d.total_papers} papers analyzed · ${d.users} participants</div>`;
    chat.appendChild(div);scr();
    sH('session ended');sW('offline — next cycle on schedule');
    $('bdie').textContent='⏱ END';
    if(tIv)clearInterval(tIv);
    if(joined){$('ibar').classList.add('off');$('min').placeholder='Session ended'}
    uPB(5);
  });
  es.addEventListener('ping',e=>{
    const d=JSON.parse(e.data);
    $('beye').textContent='👁 '+(d.v||0);
  });
  es.onerror=()=>{
    sH('reconnecting...');sW('reconnecting...');
    es.close();setTimeout(sse,3000);
  };
}

$('nin').focus();
</script>
</body></html>"""

if __name__ == "__main__":
    print("=" * 55)
    print("🔬 Colloquium — Multi-Agent AI Research Roundtable")
    print(f"   model       : {MODEL}")
    print(f"   backup      : {BACKUP}")
    print(f"   agent gap   : {AGENT_GAP}s")
    print(f"   papers      : {MAX_PAPERS}")
    print(f"   phases      : {len(PHASES)} ({', '.join(p['name'] for p in PHASES)})")
    print(f"   agents      : {', '.join(a['avatar']+' '+a['name'] for a in AGENTS)}")
    print(f"   max uptime  : {MAX_UP//3600}h {MAX_UP%3600//60}m")
    print("=" * 55)
    import sys
    sys.stdout.flush()

    threading.Thread(target=engine, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)

  
