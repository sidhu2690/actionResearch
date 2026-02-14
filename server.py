#!/usr/bin/env python3
"""Colloquium — Multi-Agent AI Research Roundtable."""

import json, os, re, random, time, queue, threading, uuid
import urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from flask import Flask, Response, request, jsonify

# ━━ CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOOT          = time.time()
MAX_UP        = 21300
AGENT_GAP     = 22
USER_WAIT     = 5
MODEL         = "llama-3.1-8b-instant"
BACKUP        = "meta-llama/llama-4-scout-17b-16e-instruct"
PORT          = 8080

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
        "name": "Feynman", "role": "Physics & Intuition Specialist", "avatar": "⚛️",
        "color": "#03a9f4",
        "personality": "Curious, intuitive, loves analogies, hates unnecessary complexity",
        "style": "Checks physical plausibility, connects to known theories, simplifies ideas",
        "focus": "physical plausibility, real-world grounding, intuition, established science",
    },
    {
        "name": "Hinton", "role": "ML & Architecture Critic", "avatar": "🧠",
        "color": "#9c27b0",
        "personality": "Deep thinker about learning, skeptical of hype, focused on what works",
        "style": "Questions model capacity, architecture alternatives, worries about generalization",
        "focus": "architecture, training, generalization, scalability, compute costs",
    },
    {
        "name": "Popper", "role": "Critical Reviewer & Skeptic", "avatar": "🧩",
        "color": "#f44336",
        "personality": "Relentlessly critical, demands evidence, hates overstatement",
        "style": "Finds weak claims, demands ablation studies, checks experimental validity",
        "focus": "claims vs evidence, experimental design, reproducibility, limitations",
    },
    {
        "name": "Tesla", "role": "Speculative Visionary", "avatar": "🔮",
        "color": "#4caf50",
        "personality": "Wild imagination grounded in technical understanding, cross-domain thinker",
        "style": "Proposes bold extensions, cross-field connections, future applications",
        "focus": "novel extensions, cross-domain ideas, future implications, wild combinations",
    },
]

PHASES = [
    {
        "name": "SUMMARY", "label": "📖 Summary Phase",
        "desc": "Agents summarize the paper from their perspective",
        "rounds": 5,
        "prompt": 'Summarize this paper\'s key contribution FROM YOUR PERSPECTIVE as {role}. What stands out? What is the core idea? Focus on aspects relevant to your expertise ({focus}). Under 100 words.',
    },
    {
        "name": "CRITIQUE", "label": "🔍 Critique Phase",
        "desc": "Each agent critiques from their domain expertise",
        "rounds": 5,
        "prompt": 'Critique this paper from your perspective as {role}. Weaknesses? Missing pieces? Questionable assumptions? Be specific and technical. Reference what others said if useful. Under 100 words.',
    },
    {
        "name": "CROSSFIRE", "label": "⚔️ Cross-Examination",
        "desc": "Agents challenge each other's points",
        "rounds": 8,
        "prompt": 'Respond to {target}\'s point. Agree, disagree, or push deeper. Ask a follow-up or challenge them. Stay in character as {role}. Under 80 words.',
    },
    {
        "name": "PROPOSAL", "label": "💡 Proposal Phase",
        "desc": "Propose extensions, experiments, and new directions",
        "rounds": 5,
        "prompt": 'Propose ONE concrete extension, experiment, or improvement. Be specific — what would you actually build or test? Think from your perspective as {role} ({focus}). Under 100 words.',
    },
    {
        "name": "VERDICT", "label": "🤝 Verdict Phase",
        "desc": "Final verdicts and key takeaways",
        "rounds": 5,
        "prompt": 'Final verdict. Key strength, key weakness, significance rating (1-10). As {role}, give your bottom line on this paper. Under 80 words.',
    },
]

USER_COLORS = [
    "#ff9800","#e91e63","#9c27b0","#03a9f4",
    "#4caf50","#ff5722","#00bcd4","#cddc39",
]

# ━━ STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
state = {
    "paper": None, "paper_num": 0,
    "phase": None, "phase_idx": -1,
    "agents": AGENTS, "messages": [],
    "typing": None,
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
        with self._lock: self._q.append(q)
        return q
    def drop(self, q):
        with self._lock:
            try: self._q.remove(q)
            except: pass
    @property
    def viewers(self):
        with self._lock: return len(self._q)
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
            temperature=0.8, max_tokens=200)
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

# ━━ ARXIV FETCHER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_papers(count=12):
    papers = []
    cats = random.sample(ARXIV_CATS, min(4, len(ARXIV_CATS)))
    q = "+OR+".join(f"cat:{c}" for c in cats)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={q}&sortBy=submittedDate"
        f"&sortOrder=descending&max_results={count}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Colloquium/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
        ns = {
            "a": "http://www.w3.org/2005/Atom",
            "x": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ns):
            title = entry.find("a:title", ns).text.strip().replace("\n", " ")
            title = re.sub(r'\s+', ' ', title)
            abstract = entry.find("a:summary", ns).text.strip().replace("\n", " ")
            abstract = re.sub(r'\s+', ' ', abstract)
            authors = [
                a.find("a:name", ns).text
                for a in entry.findall("a:author", ns)
            ]
            link = entry.find("a:id", ns).text
            for l in entry.findall("a:link", ns):
                if l.get("title") == "pdf":
                    link = l.get("href", link)
                    break
            categories = [c.get("term","") for c in entry.findall("a:category", ns)]
            if len(abstract) > 100:
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors[:4]) + ("..." if len(authors) > 4 else ""),
                    "link": link,
                    "categories": categories[:4],
                })
    except Exception as e:
        print(f"  ⚠ arXiv fetch error: {e}")
    return papers

FALLBACK_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
        "authors": "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
        "link": "https://arxiv.org/abs/1706.03762",
        "categories": ["cs.CL", "cs.LG"],
    },
    {
        "title": "Scaling Laws for Neural Language Models",
        "abstract": "We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range.",
        "authors": "Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei",
        "link": "https://arxiv.org/abs/2001.08361",
        "categories": ["cs.LG", "cs.CL"],
    },
    {
        "title": "Denoising Diffusion Probabilistic Models",
        "abstract": "We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics.",
        "authors": "Ho, Jain, Abbeel",
        "link": "https://arxiv.org/abs/2006.11239",
        "categories": ["cs.LG", "stat.ML"],
    },
]

# ━━ STREAM AI MESSAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stream_ai(agent, text, history):
    words = text.split()
    wps = max(0.05, min(16 / max(len(words), 1), 0.4))

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
    print(f"  {agent['avatar']} {agent['name']}: {text[:70]}...")

# ━━ HANDLE USER MESSAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def handle_user(paper, history):
    """Drain queue, pick agent, respond. Returns True if handled."""
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

    system = f"""You are {agent['name']} — {agent['role']}.
Personality: {agent['personality']}. Style: {agent['style']}.
You're in a research roundtable discussing:
"{paper['title']}"
Abstract: {paper['abstract'][:400]}
A human asked a question or made a comment. Respond helpfully and warmly, staying in character. Use their name. Under 80 words."""

    inst = f"Recent discussion:\n{ctx}\n\nRespond to the human. Under 80 words."

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
    time.sleep(1)
    print("\n🔬 Fetching papers from arXiv...")
    papers = fetch_papers(12)
    if not papers:
        print("  ⚠ Using fallback papers")
        papers = list(FALLBACK_PAPERS)
    random.shuffle(papers)
    print(f"  ✓ {len(papers)} papers loaded\n")

    pidx = 0

    while tleft() > 120 and pidx < len(papers):
        paper = papers[pidx]; pidx += 1
        state["paper"] = paper
        state["paper_num"] += 1

        print(f"\n{'='*55}")
        print(f"📄 Paper #{state['paper_num']}: {paper['title'][:55]}...")
        print(f"   {paper['authors']}")
        print(f"   {', '.join(paper['categories'])}")
        print(f"{'='*55}")

        pmsg = {
            "type": "paper", "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": paper["authors"], "link": paper["link"],
            "categories": paper["categories"],
            "number": state["paper_num"], "time": now_hm(),
        }
        state["messages"].append(pmsg)
        bus.emit("newpaper", pmsg)

        history = [{"role": "user", "content":
            f'PAPER: "{paper["title"]}"\n'
            f'Authors: {paper["authors"]}\n'
            f'Abstract: {paper["abstract"]}'}]

        for phi, phase in enumerate(PHASES):
            if tleft() < 120:
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
            print(f"\n  {phase['label']}")

            time.sleep(3)

            for rnd in range(phase["rounds"]):
                if tleft() < 60:
                    break

                # user check
                if handle_user(paper, history):
                    continue

                # pick agent
                agent = AGENTS[rnd % len(AGENTS)]

                # pick target for crossfire
                target = None
                if phase["name"] == "CROSSFIRE":
                    others = [a for a in AGENTS if a["name"] != agent["name"]]
                    # prefer whoever spoke last
                    for m in reversed(state["messages"][-6:]):
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

                system = f"""You are {agent['name']} — {agent['role']}.
Personality: {agent['personality']}
Style: {agent['style']}
Focus: {agent['focus']}

Research roundtable analyzing a paper.
Paper: "{paper['title']}"
Abstract: {paper['abstract'][:500]}

Phase: {phase['label']} — {phase['desc']}
{"Humans are watching and may ask questions — acknowledge them if relevant." if users else ""}
Be substantive, specific, in character. Don't start with your name."""

                inst = phase["prompt"].format(
                    role=agent["role"],
                    focus=agent["focus"],
                    target=target["name"] if target else "",
                )
                inst = f'Paper: "{paper["title"]}"\n{inst}'

                # add recent context for later phases
                if phase["name"] in ("CROSSFIRE", "PROPOSAL", "VERDICT"):
                    recent = [
                        f'{m["speaker"]}: {m["text"]}'
                        for m in state["messages"][-6:]
                        if m.get("type") == "message"
                    ]
                    if recent:
                        inst += "\n\nRecent:\n" + "\n".join(recent[-4:])

                # sometimes reference user
                for m in reversed(state["messages"][-8:]):
                    if m.get("type") == "user":
                        if random.random() < 0.25:
                            inst += f'\n(Human {m["user_name"]} said: "{m["text"]}" — weave in if relevant.)'
                        break

                try:
                    text = llm(system, history, inst)
                    stream_ai(agent, text, history)
                except Exception as e:
                    print(f"  ✖ {e}")
                    state["typing"] = None

                # wait before next turn
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
            done_msg = {
                "type": "system",
                "text": f"✅ Analysis of paper #{state['paper_num']} complete — moving on...",
                "time": now_hm(),
            }
            state["messages"].append(done_msg)
            bus.emit("system", done_msg)
            time.sleep(5)

        # if we exhausted papers, fetch more
        if pidx >= len(papers) and tleft() > 300:
            print("\n🔬 Fetching more papers...")
            more = fetch_papers(10)
            if more:
                random.shuffle(more)
                papers.extend(more)
                print(f"  ✓ {len(more)} more loaded")

    # shutdown
    cnt  = len([m for m in state["messages"] if m.get("type") == "message"])
    ucnt = len([m for m in state["messages"] if m.get("type") == "user"])
    bus.emit("shutdown", {
        "total_msgs": cnt, "total_papers": state["paper_num"],
        "user_msgs": ucnt, "users": len(users),
    })
    print(f"\n⏰ Done. {cnt} AI · {ucnt} human msgs · {state['paper_num']} papers.")

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
            "paper": state["paper"], "paper_num": state["paper_num"],
            "phase": state["phase"], "phase_idx": state["phase_idx"],
            "agents": AGENTS,
            "messages": state["messages"][-150:],
            "typing": state["typing"],
            "boot": BOOT, "max_up": MAX_UP, "timeleft": tleft(),
            "users": list(users.values()), "viewers": bus.viewers,
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
  --bg:#0a0e14;--panel:#131921;--card:#1a2030;--brd:#252d3a;
  --tx:#d4dae4;--tx2:#6b7a8d;--acc:#00e5a0;--acc2:#00b8d4;
  --warn:#ff6b6b;--hi:#1e3a5f;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{
  font-family:'SF Mono','Fira Code','Cascadia Code',Consolas,monospace;
  background:var(--bg);color:var(--tx);display:flex;justify-content:center;
}

/* overlay */
.overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.9);
  display:flex;align-items:center;justify-content:center;
  z-index:100;backdrop-filter:blur(12px);
}
.overlay.hidden{display:none}
.join-card{
  background:var(--panel);border:1px solid var(--brd);
  border-radius:12px;padding:2rem 1.6rem;width:90%;max-width:380px;text-align:center;
}
.join-card h1{font-size:1.4rem;margin-bottom:.2rem}
.join-card .acc{color:var(--acc)}
.join-card .sub{color:var(--tx2);font-size:.72rem;margin-bottom:1.2rem;line-height:1.5}
.join-card input{
  width:100%;padding:.7rem .9rem;background:var(--bg);
  border:1px solid var(--brd);border-radius:6px;
  color:var(--tx);font-size:.85rem;font-family:inherit;
  outline:none;margin-bottom:.7rem;
}
.join-card input:focus{border-color:var(--acc)}
.join-card input::placeholder{color:var(--tx2)}
.btn{
  width:100%;padding:.7rem;border:none;border-radius:6px;
  font-size:.8rem;font-weight:600;cursor:pointer;
  font-family:inherit;margin-bottom:.4rem;
}
.btn:hover{opacity:.85}
.btn-go{background:var(--acc);color:#000}
.btn-w{background:transparent;color:var(--tx2);border:1px solid var(--brd)}
.join-card .ht{color:var(--tx2);font-size:.6rem;margin-top:.8rem;line-height:1.6}

/* app */
.app{
  width:100%;max-width:560px;height:100vh;height:100dvh;
  display:flex;flex-direction:column;background:var(--bg);
  position:relative;
}

/* header */
.hdr{
  display:flex;align-items:center;gap:.6rem;
  padding:.55rem .8rem;background:var(--panel);
  border-bottom:1px solid var(--brd);min-height:54px;z-index:10;
}
.hdr-ico{font-size:1.3rem}
.hdr-info{flex:1;min-width:0}
.hdr-name{font-size:.85rem;font-weight:700;letter-spacing:.03em}
.hdr-sub{font-size:.65rem;color:var(--tx2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hdr-sub .typing{color:var(--acc)}
.hdr-badges{display:flex;gap:.25rem}
.badge{
  padding:.15rem .45rem;border-radius:4px;font-weight:600;
  font-size:.58rem;font-family:inherit;letter-spacing:.02em;
}
.b-eye{background:rgba(0,184,212,.1);color:var(--acc2)}
.b-msg{background:rgba(0,229,160,.1);color:var(--acc)}
.b-die{background:rgba(255,107,107,.1);color:var(--warn)}

/* agents strip */
.agents-strip{
  display:flex;gap:.25rem;padding:.35rem .6rem;
  background:rgba(0,0,0,.2);border-bottom:1px solid var(--brd);
  overflow-x:auto;font-size:.62rem;
}
.agents-strip::-webkit-scrollbar{display:none}
.ag-chip{
  display:flex;align-items:center;gap:.2rem;
  padding:.15rem .45rem;border-radius:4px;
  background:rgba(255,255,255,.03);border:1px solid var(--brd);
  white-space:nowrap;flex-shrink:0;
}
.ag-chip .dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.ag-chip .role{color:var(--tx2);font-size:.52rem}

/* chat */
.chat{
  flex:1;overflow-y:auto;overflow-x:hidden;
  padding:.5rem .6rem;
}
.chat::-webkit-scrollbar{width:3px}
.chat::-webkit-scrollbar-thumb{background:var(--brd);border-radius:3px}

/* system pills */
.sys{text-align:center;margin:.6rem 0}
.pill{
  display:inline-block;background:var(--card);color:var(--tx2);
  padding:.3rem .7rem;border-radius:4px;font-size:.68rem;
  max-width:92%;line-height:1.5;border:1px solid var(--brd);
}
.pill.phase{
  color:var(--acc);font-weight:600;font-size:.72rem;
  border-color:rgba(0,229,160,.2);background:rgba(0,229,160,.05);
  text-align:left;
}
.pill.phase .ph-desc{color:var(--tx2);font-weight:400;font-size:.62rem;margin-top:2px}

/* paper card */
.paper-card{
  background:var(--card);border:1px solid var(--brd);
  border-radius:8px;padding:.7rem .8rem;margin:.6rem 0;
  font-size:.72rem;line-height:1.5;
}
.paper-card .p-num{color:var(--acc);font-weight:700;font-size:.6rem;letter-spacing:.05em;margin-bottom:.3rem}
.paper-card .p-title{color:var(--tx);font-weight:600;font-size:.8rem;margin-bottom:.3rem;line-height:1.4}
.paper-card .p-authors{color:var(--acc2);font-size:.62rem;margin-bottom:.4rem}
.paper-card .p-abstract{color:var(--tx2);font-size:.65rem;line-height:1.6;max-height:80px;overflow:hidden;transition:max-height .3s}
.paper-card .p-abstract.open{max-height:none}
.paper-card .p-toggle{
  color:var(--acc);font-size:.6rem;cursor:pointer;margin-top:.3rem;
  display:inline-block;
}
.paper-card .p-toggle:hover{text-decoration:underline}
.paper-card .p-cats{display:flex;gap:.2rem;margin-top:.4rem;flex-wrap:wrap}
.paper-card .p-cat{
  padding:.1rem .35rem;border-radius:3px;font-size:.55rem;
  background:rgba(0,184,212,.08);color:var(--acc2);border:1px solid rgba(0,184,212,.15);
}
.paper-card .p-link{
  display:inline-block;margin-top:.4rem;font-size:.58rem;
  color:var(--acc);text-decoration:none;
}
.paper-card .p-link:hover{text-decoration:underline}
/* message bubbles */
.msg{display:flex;flex-direction:column;margin-bottom:2px;animation:up .2s ease}
@keyframes up{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.msg.ai{align-items:flex-start;padding-right:2rem}
.msg.usr{align-items:flex-end;padding-left:2rem}
.msg .body{
  position:relative;padding:.4rem .6rem .2rem .6rem;
  border-radius:6px;max-width:100%;font-size:.78rem;
  line-height:1.45;word-wrap:break-word;
}
.msg.ai .body{background:var(--card);border:1px solid var(--brd);border-top-left-radius:0}
.msg.usr .body{background:var(--hi);border:1px solid rgba(30,58,95,.5);border-top-right-radius:0}
.msg.cont .body{border-radius:6px}
.msg.cont{margin-top:1px}
.msg .who{font-size:.68rem;font-weight:600;margin-bottom:2px;display:flex;align-items:center;gap:.3rem}
.msg .who .rl{font-size:.55rem;font-weight:400;color:var(--tx2)}
.msg .txt{color:var(--tx)}
.msg .meta{
  float:right;display:flex;align-items:center;gap:3px;
  margin-left:8px;margin-top:3px;font-size:.55rem;
  color:rgba(255,255,255,.3);white-space:nowrap;
}
.msg .spacer{display:inline-block;width:3.8rem;height:1px}

.cursor{
  display:inline-block;width:2px;height:.85em;background:var(--acc);
  margin-left:1px;animation:blinkcur .7s step-end infinite;vertical-align:text-bottom;
}
@keyframes blinkcur{0%,100%{opacity:1}50%{opacity:0}}

.typing-dots{display:inline-flex;gap:3px;align-items:center;padding:4px 0}
.typing-dots span{
  width:6px;height:6px;border-radius:50%;background:var(--tx2);
  animation:dp 1.4s ease-in-out infinite;
}
.typing-dots span:nth-child(2){animation-delay:.2s}
.typing-dots span:nth-child(3){animation-delay:.4s}
@keyframes dp{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}

/* progress bar for phases */
.phase-bar{
  display:flex;gap:2px;padding:.25rem .6rem;
  background:rgba(0,0,0,.15);border-bottom:1px solid var(--brd);
}
.phase-bar .pb-seg{
  flex:1;height:3px;border-radius:2px;
  background:var(--brd);transition:background .3s;
}
.phase-bar .pb-seg.done{background:var(--acc)}
.phase-bar .pb-seg.active{background:var(--acc);animation:pulse 1.5s ease infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* input bar */
.inputbar{
  display:flex;align-items:center;gap:.4rem;
  padding:.4rem .6rem;background:var(--panel);
  border-top:1px solid var(--brd);min-height:50px;
}
.inputbar.off{opacity:.3;pointer-events:none}
.inwrap{
  flex:1;display:flex;align-items:center;background:var(--bg);
  border:1px solid var(--brd);border-radius:6px;padding:.1rem .2rem .1rem .7rem;
}
.inwrap:focus-within{border-color:var(--acc)}
.inwrap input{
  flex:1;background:none;border:none;outline:none;
  color:var(--tx);font-size:.8rem;padding:.5rem 0;font-family:inherit;
}
.inwrap input::placeholder{color:var(--tx2)}
.sendbtn{
  width:38px;height:38px;border-radius:6px;border:none;
  background:var(--acc);color:#000;font-size:1rem;font-weight:700;
  cursor:pointer;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;font-family:inherit;
}
.sendbtn:hover{opacity:.85}
.sendbtn:disabled{opacity:.2;cursor:default}

/* watcher bar */
.watchbar{
  display:flex;align-items:center;justify-content:center;
  padding:.45rem .8rem;background:var(--panel);
  border-top:1px solid var(--brd);min-height:44px;
  font-size:.7rem;color:var(--tx2);gap:.4rem;
}
.watchbar .live{color:var(--acc);font-weight:700}
.watchbar .joinlink{
  color:var(--acc);cursor:pointer;text-decoration:underline;margin-left:.5rem;
}

/* scroll btn */
.scbtn{
  display:none;position:absolute;bottom:64px;right:12px;
  width:36px;height:36px;background:var(--panel);border:1px solid var(--brd);
  border-radius:6px;align-items:center;justify-content:center;
  cursor:pointer;z-index:20;box-shadow:0 2px 12px rgba(0,0,0,.6);
  font-size:.9rem;color:var(--tx2);
}
.scbtn:hover{background:var(--card)}
.scbtn .ubadge{
  position:absolute;top:-4px;right:-4px;background:var(--acc);
  color:#000;font-size:.52rem;font-weight:700;
  min-width:16px;height:16px;border-radius:3px;
  display:flex;align-items:center;justify-content:center;padding:0 3px;
}

/* shutdown */
.shutdown{
  text-align:center;padding:1.2rem .8rem;
  background:rgba(255,107,107,.06);border:1px solid rgba(255,107,107,.15);
  border-radius:6px;margin:.8rem 0;
}
.shutdown .big{color:var(--warn);font-size:.8rem;font-weight:700}
.shutdown .sm{color:var(--tx2);font-size:.62rem;margin-top:.3rem}

@media(max-width:560px){.app{max-width:100%}}
@media(min-width:561px){body{align-items:center;padding:1rem 0}.app{border-radius:10px;height:96vh;overflow:hidden;border:1px solid var(--brd)}}
</style>
</head><body>

<!-- JOIN SCREEN -->
<div class="overlay" id="overlay">
  <div class="join-card">
    <h1>🔬 <span class="acc">Colloquium</span></h1>
    <div class="sub">
      Multi-agent AI research roundtable<br>
      Five specialized AIs analyze live papers from arXiv<br>
      Watch them think — or join the discussion
    </div>
    <input type="text" id="namein" placeholder="Your name..." maxlength="20"
      onkeydown="if(event.key==='Enter')doJoin()">
    <button class="btn btn-go" onclick="doJoin()">Join Roundtable</button>
    <button class="btn btn-w" onclick="doWatch()">Just Observe</button>
    <div class="ht">
      📐 Euler · ⚛️ Feynman · 🧠 Hinton · 🧩 Popper · 🔮 Tesla<br>
      Five phases per paper: Summary → Critique → Cross-Exam → Proposals → Verdict
    </div>
  </div>
</div>

<!-- APP -->
<div class="app" id="app" style="display:none">
  <div class="hdr">
    <div class="hdr-ico">🔬</div>
    <div class="hdr-info">
      <div class="hdr-name">Colloquium</div>
      <div class="hdr-sub" id="hdrsub">connecting...</div>
    </div>
    <div class="hdr-badges">
      <div class="badge b-eye" id="beye">👁 0</div>
      <div class="badge b-msg" id="bmsg">💬 0</div>
      <div class="badge b-die" id="bdie">⏱ --</div>
    </div>
  </div>

  <div class="phase-bar" id="phasebar">
    <div class="pb-seg" data-p="0"></div>
    <div class="pb-seg" data-p="1"></div>
    <div class="pb-seg" data-p="2"></div>
    <div class="pb-seg" data-p="3"></div>
    <div class="pb-seg" data-p="4"></div>
  </div>

  <div class="agents-strip" id="agents"></div>
  <div class="chat" id="chat"></div>

  <div class="inputbar" id="inputbar" style="display:none">
    <div class="inwrap">
      <input type="text" id="msgin" placeholder="Ask a question or comment..." maxlength="500"
        onkeydown="if(event.key==='Enter')doSend()">
    </div>
    <button class="sendbtn" onclick="doSend()">➤</button>
  </div>

  <div class="watchbar" id="watchbar">
    <span class="live">● LIVE</span>
    <span id="wtext">analysis in progress</span>
    <span class="joinlink" onclick="showJoin()">join discussion</span>
  </div>

  <div class="scbtn" id="scbtn" onclick="jumpBottom()">
    ↓<div class="ubadge" id="ubadge" style="display:none">0</div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const chat=$('chat');
const PHASE_NAMES=['SUMMARY','CRITIQUE','CROSSFIRE','PROPOSAL','VERDICT'];

let myId=null,myName=null,myColor=null,joined=false;
let agents=[],bootTime=0,maxUp=0;
let msgCount=0,lastWho='';
let curBub=null,curTxt=null,typBub=null,timerIv=null;
let curPhase=-1;

// scroll
let scrolledUp=false,missed=0;
function scr(){if(!scrolledUp)chat.scrollTop=chat.scrollHeight}
function jumpBottom(){
  scrolledUp=false;missed=0;
  chat.scrollTop=chat.scrollHeight;
  $('scbtn').style.display='none';
  $('ubadge').style.display='none';
}
chat.addEventListener('scroll',()=>{
  const g=chat.scrollHeight-chat.scrollTop-chat.clientHeight;
  const was=scrolledUp;scrolledUp=g>80;
  if(!scrolledUp&&was){missed=0;$('ubadge').style.display='none'}
  $('scbtn').style.display=scrolledUp?'flex':'none';
});
function notif(){
  if(scrolledUp){missed++;$('ubadge').textContent=missed;$('ubadge').style.display='flex'}
}

// join
async function doJoin(){
  const n=$('namein').value.trim();
  if(!n)return $('namein').focus();
  try{
    const r=await fetch('/join',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n})});
    const d=await r.json();
    if(d.error){alert(d.error);return}
    myId=d.id;myName=d.name;myColor=d.color;joined=true;
    $('overlay').classList.add('hidden');
    $('app').style.display='flex';
    $('inputbar').style.display='flex';
    $('watchbar').style.display='none';
    startSSE();
    setTimeout(()=>$('msgin').focus(),300);
  }catch(e){alert('Connection failed')}
}
function doWatch(){
  joined=false;
  $('overlay').classList.add('hidden');
  $('app').style.display='flex';
  $('inputbar').style.display='none';
  $('watchbar').style.display='flex';
  startSSE();
}
function showJoin(){
  $('overlay').classList.remove('hidden');
  $('namein').focus();
}

// send
async function doSend(){
  const inp=$('msgin'),txt=inp.value.trim();
  if(!txt||!myId)return;
  inp.value='';inp.focus();
  try{await fetch('/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:myId,text:txt})})}
  catch(e){console.error(e)}
}

// helpers
function fmt(s){
  s=Math.max(0,Math.floor(s));
  const h=Math.floor(s/3600),m=Math.floor(s%3600/60),sc=s%60;
  return h>0?h+'h '+String(m).padStart(2,'0')+'m':m+'m '+String(sc).padStart(2,'0')+'s';
}
function startTimer(){
  if(timerIv)clearInterval(timerIv);
  timerIv=setInterval(()=>{
    const l=Math.max(0,maxUp-(Date.now()/1000-bootTime));
    $('bdie').textContent='⏱ '+fmt(l);
    if(l<300)$('bdie').style.background='rgba(255,107,107,.25)';
    if(l<=0){$('bdie').textContent='⏱ END';clearInterval(timerIv)}
  },1000);
}
function setH(h){$('hdrsub').innerHTML=h}
function setW(h){$('wtext').innerHTML=h}

function updatePhaseBar(idx){
  curPhase=idx;
  document.querySelectorAll('.pb-seg').forEach((el,i)=>{
    el.classList.remove('done','active');
    if(i<idx)el.classList.add('done');
    else if(i===idx)el.classList.add('active');
  });
}

function renderAgents(ul){
  let h='';
  agents.forEach(a=>{
    h+=`<div class="ag-chip"><div class="dot" style="background:${a.color}"></div>${a.avatar} ${a.name} <span class="role">${a.role}</span></div>`;
  });
  (ul||[]).forEach(u=>{
    const me=(myName&&u.name===myName)?' (you)':'';
    h+=`<div class="ag-chip"><div class="dot" style="background:${u.color}"></div>${u.name}${me}</div>`;
  });
  $('agents').innerHTML=h;
}

// render helpers
function sysPill(html,cls){
  const d=document.createElement('div');d.className='sys';
  d.innerHTML=`<span class="pill ${cls||''}">${html}</span>`;
  chat.appendChild(d);scr();
}
function phasePill(p){
  const d=document.createElement('div');d.className='sys';
  d.innerHTML=`<span class="pill phase">${p.label}<div class="ph-desc">${p.desc}</div></span>`;
  chat.appendChild(d);scr();
}
function paperCard(p){
  const id='abs'+Date.now();
  const d=document.createElement('div');d.className='paper-card';
  d.innerHTML=`
    <div class="p-num">PAPER #${p.number} · ${p.time}</div>
    <div class="p-title">${p.title}</div>
    <div class="p-authors">${p.authors}</div>
    <div class="p-abstract" id="${id}">${p.abstract}</div>
    <span class="p-toggle" onclick="
      var el=document.getElementById('${id}');
      if(el.classList.contains('open')){el.classList.remove('open');this.textContent='show more ▾'}
      else{el.classList.add('open');this.textContent='show less ▴'}
    ">show more ▾</span>
    <div class="p-cats">${(p.categories||[]).map(c=>'<span class="p-cat">'+c+'</span>').join('')}</div>
    ${p.link?'<a class="p-link" href="'+p.link+'" target="_blank" rel="noopener">→ view on arXiv</a>':''}
  `;
  chat.appendChild(d);scr();
}

function rmTyp(){if(typBub){typBub.remove();typBub=null}}
function addTyp(name,ava,col,role){
  rmTyp();
  const co=(lastWho===name);
  const d=document.createElement('div');
  d.className=`msg ai${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${col}">${ava} ${name} <span class="rl">${role}</span></div>`;
  h+=`<div class="body"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  d.innerHTML=h;chat.appendChild(d);typBub=d;scr();
}

function startBub(sp,ava,col,role,tm){
  rmTyp();
  const co=(lastWho===sp);
  const d=document.createElement('div');
  d.className=`msg ai${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${col}">${ava} ${sp} <span class="rl">${role}</span></div>`;
  h+=`<div class="body"><span class="meta"><span class="tm">${tm}</span></span>`;
  h+=`<span class="txt"></span><span class="cursor"></span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);
  curBub=d;curTxt=d.querySelector('.txt');scr();
}
function addW(w){
  if(!curTxt)return;
  const t=curTxt.textContent;
  curTxt.textContent=t?(t+' '+w):w;scr();
}
function finBub(sp,tm){
  if(curBub){
    const c=curBub.querySelector('.cursor');if(c)c.remove();
    const m=curBub.querySelector('.meta');
    if(m)m.innerHTML=`<span class="tm">${tm}</span>`;
  }
  lastWho=sp;curBub=null;curTxt=null;
  msgCount++;$('bmsg').textContent='💬 '+msgCount;
  notif();scr();
}

function fullMsg(m){
  const who=m.speaker||m.user_name;
  const isU=(m.type==='user');
  const co=(lastWho===who);
  const d=document.createElement('div');
  d.className=`msg ${isU?'usr':'ai'}${co?' cont':''}`;
  const col=m.color||'#aaa',ava=m.avatar||'',role=m.role||'';
  let h='';
  if(!co){
    if(isU) h+=`<div class="who" style="color:${col}">${who}</div>`;
    else h+=`<div class="who" style="color:${col}">${ava} ${who} <span class="rl">${role}</span></div>`;
  }
  h+=`<div class="body"><span class="meta"><span class="tm">${m.time||''}</span></span>`;
  h+=`<span class="txt">${m.text}</span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);
  lastWho=who;if(!isU)msgCount++;
}

function userBub(m){
  const isMe=(myName&&m.user_name===myName);
  const co=(lastWho===m.user_name);
  const d=document.createElement('div');
  d.className=`msg usr${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${m.color}">${m.user_name}${isMe?' (you)':''}</div>`;
  h+=`<div class="body"><span class="meta"><span class="tm">${m.time||''}</span></span>`;
  h+=`<span class="txt">${m.text}</span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);lastWho=m.user_name;scr();
}

// SSE
function startSSE(){
  setH('connecting...');
  const es=new EventSource('/stream');

  es.addEventListener('fullstate',e=>{
    const d=JSON.parse(e.data);
    bootTime=d.boot;maxUp=d.max_up;
    agents=d.agents||[];
    renderAgents(d.users);
    $('beye').textContent='👁 '+(d.viewers||0);
    if(d.phase_idx>=0)updatePhaseBar(d.phase_idx);
    setH(d.paper?d.paper.title.substring(0,50)+'...':'waiting for paper...');

    chat.innerHTML='';msgCount=0;lastWho='';
    sysPill('🔬 <b>Colloquium</b> — AI Research Roundtable','');

    if(d.messages) d.messages.forEach(m=>{
      if(m.type==='paper') paperCard(m);
      else if(m.type==='phase') phasePill(m);
      else if(m.type==='message') fullMsg(m);
      else if(m.type==='user') fullMsg(m);
      else if(m.type==='system') sysPill(m.text,'');
    });
    $('bmsg').textContent='💬 '+msgCount;
    startTimer();scr();
    setW('analysis in progress');
  });

  es.addEventListener('newpaper',e=>{
    const d=JSON.parse(e.data);
    paperCard(d);
    setH(d.title.substring(0,50)+'...');
    updatePhaseBar(-1);
  });

  es.addEventListener('newphase',e=>{
    const d=JSON.parse(e.data);
    phasePill(d);
    const idx=PHASE_NAMES.indexOf(d.name);
    if(idx>=0)updatePhaseBar(idx);
  });

  es.addEventListener('typing',e=>{
    const d=JSON.parse(e.data);
    addTyp(d.name,d.avatar,d.color,d.role);
    setH(`<span class="typing">${d.avatar} ${d.name} is thinking...</span>`);
    setW(`${d.avatar} ${d.name} is analyzing...`);
  });

  es.addEventListener('msgstart',e=>{
    const d=JSON.parse(e.data);
    startBub(d.speaker,d.avatar,d.color,d.role,d.time);
    setH(`<span class="typing">${d.avatar} ${d.speaker} is writing...</span>`);
    setW(`${d.avatar} ${d.speaker} is writing...`);
  });

  es.addEventListener('word',e=>{addW(JSON.parse(e.data).w)});

  es.addEventListener('msgdone',e=>{
    const d=JSON.parse(e.data);
    finBub(d.speaker,d.time);
    setH(document.querySelector('.paper-card .p-title')?.textContent?.substring(0,50)+'...'||'roundtable');
    setW('analysis in progress');
  });

  es.addEventListener('usermsg',e=>{userBub(JSON.parse(e.data))});
  es.addEventListener('system',e=>{sysPill(JSON.parse(e.data).text,'')});

  es.addEventListener('presence',e=>{
    const d=JSON.parse(e.data);
    renderAgents(d.users);
    $('beye').textContent='👁 '+(d.viewers||0);
  });

  es.addEventListener('waiting',e=>{
    const d=JSON.parse(e.data);
    let g=d.gap;
    setW(`${d.avatar} <span style="color:${d.color}">${d.name}</span> in <span id="gcd">${g}s</span>`);
    const iv=setInterval(()=>{
      g--;const el=document.getElementById('gcd');
      if(el)el.textContent=g+'s';
      if(g<=0){clearInterval(iv);setW('next agent thinking...')}
    },1000);
  });

  es.addEventListener('shutdown',e=>{
    const d=JSON.parse(e.data);rmTyp();
    const div=document.createElement('div');div.className='shutdown';
    div.innerHTML=`<div class="big">⏱ Session Complete</div>`+
      `<div class="sm">Next cycle starts on schedule</div>`+
      `<div class="sm">${d.total_msgs} agent messages · ${d.user_msgs} human · ${d.total_papers} papers analyzed · ${d.users} humans joined</div>`;
    chat.appendChild(div);scr();
    setH('session ended');setW('offline — next cycle on schedule');
    $('bdie').textContent='⏱ END';
    if(timerIv)clearInterval(timerIv);
    if(joined){$('inputbar').classList.add('off');$('msgin').placeholder='Session ended'}
    updatePhaseBar(5);
  });

  es.addEventListener('ping',e=>{
    const d=JSON.parse(e.data);
    $('beye').textContent='👁 '+(d.v||0);
  });

  es.onerror=()=>{
    setH('reconnecting...');setW('reconnecting...');
    es.close();setTimeout(startSSE,3000);
  };
}

$('namein').focus();
</script>
</body></html>"""






