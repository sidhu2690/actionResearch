#!/usr/bin/env python3
"""Colloquium — Deep single-paper AI research teardown."""

import json, os, re, random, time, queue, threading, uuid
import urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from flask import Flask, Response, request, jsonify

# ━━ CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOOT       = time.time()
MAX_UP     = 21300
GAP        = 30
USER_WAIT  = 5
MODEL      = "llama-3.1-8b-instant"
BACKUP     = "meta-llama/llama-4-scout-17b-16e-instruct"
PORT       = 8080

AGENTS = [
    {"name": "Aria",  "avatar": "🔹", "color": "#42a5f5",
     "tone": "sharp, detail-oriented, catches subtle errors others miss, zero patience for hand-waving"},
    {"name": "Kai",   "avatar": "🔸", "color": "#ff9800",
     "tone": "big-picture thinker, questions significance and real-world impact, blunt about hype"},
    {"name": "Noor",  "avatar": "🟢", "color": "#66bb6a",
     "tone": "practical engineer mindset, asks 'does this actually work', focused on reproducibility"},
    {"name": "Sasha", "avatar": "🔻", "color": "#ef5350",
     "tone": "harshest critic, zero tolerance for BS, calls out papers published for the sake of publishing"},
    {"name": "Ravi",  "avatar": "🟣", "color": "#ab47bc",
     "tone": "methodical, fair but thorough, finds genuine strengths but also tears apart weak arguments"},
]

PHASES = [
    {
        "name": "FIRST_LOOK", "label": "📖 First Read",
        "desc": "Initial reactions — what is this, what jumps out, what smells off",
        "rounds": 20,
        "prompts": [
            "What's your gut reaction to this abstract? Don't hold back.",
            "What's the actual core claim here? Strip away the fluff.",
            "What red flags do you already see just from the abstract?",
            "What's suspiciously missing from this abstract?",
            "Does the writing quality tell you anything? Buzzword density?",
            "What questions would you immediately ask the authors?",
            "How does this position itself in the field? Honest or overselling?",
            "What can you infer about the methodology just from the abstract?",
        ],
    },
    {
        "name": "CLAIMS", "label": "⚖️ Claims Audit",
        "desc": "Dissecting every claim — justified or BS?",
        "rounds": 30,
        "prompts": [
            "List the specific claims this paper makes. Which ones are testable?",
            "Which claim is the weakest? Why?",
            "Are there hidden implicit claims not explicitly stated?",
            "What evidence would you NEED to believe these claims?",
            "Is there overclaiming? Where exactly?",
            "Do the claimed contributions match what's actually described?",
            "Challenge {target}'s take on the claims. Push harder.",
            "If you were a reviewer, which claim would you reject first?",
            "Are the claims novel or just restatements of known results?",
            "What's the gap between what they claim and what they likely prove?",
        ],
    },
    {
        "name": "METHODOLOGY", "label": "🔧 Methodology Teardown",
        "desc": "How they did it — and everything wrong with how they did it",
        "rounds": 35,
        "prompts": [
            "What methodology can you infer? What are its fundamental flaws?",
            "What alternatives should they have considered?",
            "What assumptions does this approach REQUIRE to work?",
            "What are the obvious failure modes they probably don't discuss?",
            "Is this methodology appropriate for the problem? Or forced?",
            "Respond to {target} — is their critique of the method fair?",
            "What would break this approach in practice?",
            "Is this method overcomplicated for what it does?",
            "What's the simplest baseline that might match their results?",
            "Would this methodology survive a real-world deployment?",
            "What computational costs are they probably hiding?",
            "Is the approach principled or just empirical hacking?",
        ],
    },
    {
        "name": "MATH", "label": "📐 Math & Theory",
        "desc": "Checking the math — equations, derivations, assumptions, rigor",
        "rounds": 30,
        "prompts": [
            "What mathematical framework is implied? Is it the right one?",
            "What mathematical assumptions are they probably making? Which are dangerous?",
            "What convergence/stability guarantees would this need?",
            "Is the theoretical contribution actually novel mathematically?",
            "What's the likely complexity? Are they honest about it?",
            "Respond to {target}'s math point. Agree or tear it apart.",
            "What proofs would you demand to see in the full paper?",
            "Are there obvious mathematical shortcuts or hand-waves?",
            "What edge cases would break their theoretical claims?",
            "Is the math actual contribution or just window dressing?",
        ],
    },
    {
        "name": "EXPERIMENTS", "label": "🧪 Experiments & Evaluation",
        "desc": "Is the evaluation sound or theater?",
        "rounds": 30,
        "prompts": [
            "What experiments would properly validate these claims?",
            "What baselines MUST they compare against? Bet they're missing some.",
            "Which metrics are appropriate and which might be misleading here?",
            "What datasets would you need? Are they likely cherry-picked?",
            "What ablation studies are essential but probably missing?",
            "Respond to {target}'s point about experiments.",
            "How would YOU design the evaluation if you were doing this right?",
            "Is the evaluation comprehensive or just showing what works?",
            "What failure cases should they report but probably don't?",
            "Could the results be explained by a simpler hypothesis?",
        ],
    },
    {
        "name": "NOVELTY_BS", "label": "🚩 Novelty & BS Check",
        "desc": "Is this actually new? Or published for the sake of publishing?",
        "rounds": 30,
        "prompts": [
            "How novel is this REALLY? Be brutal.",
            "Is this incremental work dressed up as a breakthrough?",
            "Could this be seen as repackaging existing ideas with new jargon?",
            "Does this solve a real problem or a manufactured one?",
            "Would a tough NeurIPS/ICML reviewer accept this? Why or why not?",
            "Respond to {target}. Is their novelty assessment fair?",
            "Is this the kind of paper that inflates citation counts but adds nothing?",
            "What prior work are they probably failing to cite?",
            "Is the 'gap in literature' they're filling actually a gap worth filling?",
            "Rate the BS level 1-10. Justify.",
            "Is this engineering masquerading as science?",
            "Would the field be any different if this paper didn't exist?",
        ],
    },
    {
        "name": "STRENGTHS", "label": "✅ What's Actually Good",
        "desc": "Fair assessment — genuine contributions and merits",
        "rounds": 20,
        "prompts": [
            "In fairness, what IS genuinely good about this work?",
            "What could be impactful IF the claims hold up?",
            "What technical decisions seem well-thought-out?",
            "Is there a kernel of a good idea buried in here?",
            "Respond to {target}. Do you agree that's a real strength?",
            "What's the best-case scenario for this research line?",
            "Would you build on this work? What part specifically?",
            "What did they get RIGHT that others get wrong?",
        ],
    },
    {
        "name": "OPEN_DEBATE", "label": "💬 Open Discussion",
        "desc": "Free-form — challenge each other, go deeper, connect dots",
        "rounds": 50,
        "prompts": [
            "Respond to what was just said. Push deeper.",
            "Challenge {target}'s last point directly.",
            "Bring up something nobody has mentioned yet about this paper.",
            "Connect this paper to a broader trend in ML. Good or bad?",
            "What would you do differently if you were the authors?",
            "Play devil's advocate — defend the paper against the criticism so far.",
            "What's the most important unresolved question about this work?",
            "If you had 5 minutes with the authors, what would you ask?",
            "Respond to {target} — you disagree. Explain why.",
            "What does this paper tell us about the state of ML research?",
            "Is this the kind of work that gets funded but doesn't advance the field?",
            "What experiment would DEFINITIVELY prove or disprove their claims?",
            "Synthesize the discussion so far. What's the consensus emerging?",
            "Where do you disagree with the group? Make your case.",
        ],
    },
    {
        "name": "VERDICT", "label": "🏛️ Final Verdict",
        "desc": "Bottom line — accept, reject, score, and why",
        "rounds": 20,
        "prompts": [
            "Final verdict: accept or reject? Score 1-10. Justify in detail.",
            "One-line summary of this paper's actual value to the field.",
            "Would you cite this paper? In what context?",
            "What's the lasting impact (if any) of this work?",
            "Respond to {target}'s verdict. Agree or disagree?",
            "If you could rewrite the abstract honestly, what would it say?",
            "Summarize the biggest problem and the biggest strength.",
            "Confidence score: how sure are you about your assessment?",
        ],
    },
]

USER_COLORS = [
    "#ff9800","#e91e63","#9c27b0","#03a9f4",
    "#4caf50","#ff5722","#00bcd4","#cddc39",
]

FALLBACK_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.",
        "authors": "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
        "link": "https://arxiv.org/abs/1706.03762",
        "categories": ["cs.CL", "cs.LG"],
    },
]

# ━━ STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
state = {
    "paper": None, "phase": None, "phase_idx": -1,
    "agents": AGENTS, "messages": [],
    "typing": None, "total_phases": len(PHASES),
}
users = {}
color_idx = [0]
user_queue = queue.Queue()

# ━━ SSE BUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Bus:
    def __init__(self):
        self._q, self._lock = [], threading.Lock()
    def listen(self):
        q = queue.Queue(maxsize=500)
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

# ━━ GROQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_client = None
def groq():
    global _client
    if not _client:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

def llm(system, history, instruction, model=MODEL):
    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-24:])
    msgs.append({"role": "user", "content": instruction})
    try:
        r = groq().chat.completions.create(
            model=model, messages=msgs,
            temperature=0.82, max_tokens=250,
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

# ━━ ARXIV ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_paper():
    cats = ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"]
    q = "+OR+".join(f"cat:{c}" for c in cats)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={q}&sortBy=submittedDate"
        f"&sortOrder=descending&max_results=30"
    )
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Colloquium/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
        root = ET.fromstring(raw)
        papers = []
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
            if len(abstract) > 150:
                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors[:5]) + ("..." if len(authors) > 5 else ""),
                    "link": link,
                    "categories": categories[:5],
                })
        if papers:
            return random.choice(papers)
    except Exception as e:
        print(f"  ⚠ arXiv: {e}")
    return FALLBACK_PAPERS[0]

# ━━ STREAM AI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stream_ai(agent, text, history):
    words = text.split()
    wps = max(0.04, min(14 / max(len(words), 1), 0.3))
    bus.emit("msgstart", {
        "speaker": agent["name"], "avatar": agent["avatar"],
        "color": agent["color"], "time": now_hm(), "is_ai": True,
    })
    for i, w in enumerate(words):
        bus.emit("word", {"w": w, "i": i, "of": len(words)})
        time.sleep(wps)
    msg = {
        "type": "message", "speaker": agent["name"],
        "avatar": agent["avatar"], "color": agent["color"],
        "text": text, "time": now_hm(),
    }
    state["messages"].append(msg)
    state["typing"] = None
    history.append({"role": "assistant", "content": f"[{agent['name']}]: {text}"})
    bus.emit("msgdone", {"speaker": agent["name"], "text": text, "time": now_hm()})
    print(f"    {agent['avatar']} {agent['name']}: {text[:72]}...")

# ━━ HANDLE USER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    for m in state["messages"][-10:]:
        if m.get("type") == "user":
            recent.append(f'{m["user_name"]}: {m["text"]}')
        elif m.get("type") == "message":
            recent.append(f'{m["speaker"]}: {m["text"]}')
    ctx = "\n".join(recent[-6:])
    system = (
        f"You are {agent['name']}, an ML researcher.\n"
        f"Tone: {agent['tone']}\n"
        f'Discussing: "{paper["title"]}"\n'
        f"Abstract: {paper['abstract'][:500]}\n"
        f"A human asked something. Respond directly, use their name. Stay sharp. Under 90 words."
    )
    inst = f"Recent:\n{ctx}\n\nRespond to the human. Under 90 words."
    state["typing"] = agent["name"]
    bus.emit("typing", {
        "name": agent["name"], "avatar": agent["avatar"],
        "color": agent["color"],
    })
    try:
        text = llm(system, history, inst)
        stream_ai(agent, text, history)
    except Exception as e:
        print(f"  ✖ user-reply: {e}")
        state["typing"] = None
    return True

# ━━ GENERATE FINAL REPORT ━━━━━━━━━━━━━━━━━━━━━━━
def generate_report(paper, history):
    print("\n  📝 Generating final report...")

    all_msgs = [
        f'[{m["speaker"]}]: {m["text"]}'
        for m in state["messages"]
        if m.get("type") == "message"
    ]
    digest = "\n".join(all_msgs[-80:])

    sections = [
        ("EXECUTIVE SUMMARY",
         "Write a 150-word executive summary of the full discussion. What was concluded about this paper overall?"),
        ("KEY STRENGTHS IDENTIFIED",
         "Based on the discussion, list the genuine strengths found. Be specific. Reference what was discussed. 120 words max."),
        ("CRITICAL PROBLEMS FOUND",
         "List every problem, flaw, and weakness identified during the discussion. Be thorough and specific. 150 words max."),
        ("MATHEMATICAL & THEORETICAL CONCERNS",
         "Summarize all math/theory concerns raised. Assumptions questioned, rigor issues, missing proofs. 120 words max."),
        ("METHODOLOGY ISSUES",
         "Summarize methodology problems identified. What's wrong with how they did it? 120 words max."),
        ("EXPERIMENTAL GAPS",
         "What experiments are missing? What evaluation problems were identified? 120 words max."),
        ("NOVELTY ASSESSMENT",
         "How novel is this work really? Summarize the group's consensus on originality. Is this published for the sake of publishing? 100 words max."),
        ("ACCEPT / REJECT RECOMMENDATION",
         "Final recommendation: accept or reject at a top venue. Score 1-10. Confidence level. Clear justification. 100 words max."),
        ("SUGGESTED IMPROVEMENTS",
         "What would make this paper actually good? Concrete actionable suggestions. 120 words max."),
        ("ONE-LINE VERDICT",
         "One devastating or praising sentence that captures the essence of this paper. Be memorable."),
    ]

    report_parts = []
    system = (
        f"You are writing a structured research analysis report.\n"
        f'Paper: "{paper["title"]}"\n'
        f"Authors: {paper['authors']}\n"
        f"Abstract: {paper['abstract']}\n\n"
        f"Based on a 6-hour roundtable discussion among 5 ML researchers. "
        f"Write clearly, specifically, and honestly. No fluff."
    )

    for title, inst in sections:
        try:
            full_inst = f"Discussion transcript (last portion):\n{digest}\n\n{inst}"
            text = llm(system, [], full_inst)
            report_parts.append({"title": title, "content": text})
            print(f"    ✓ {title}")
            time.sleep(1)
        except Exception as e:
            print(f"    ✖ {title}: {e}")
            report_parts.append({"title": title, "content": f"[Generation failed: {e}]"})

    report_msg = {
        "type": "report",
        "paper_title": paper["title"],
        "paper_authors": paper["authors"],
        "paper_link": paper["link"],
        "sections": report_parts,
        "total_messages": len(all_msgs),
        "time": now_hm(),
    }
    state["messages"].append(report_msg)
    bus.emit("report", report_msg)
    print("  📝 Report complete!")

# ━━ ENGINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def engine():
    time.sleep(2)

    print("\n🔬 Fetching paper from arXiv...")
    paper = fetch_paper()
    state["paper"] = paper

    print(f"\n{'='*60}")
    print(f"  📄 {paper['title']}")
    print(f"     {paper['authors']}")
    print(f"     {', '.join(paper['categories'][:3])}")
    print(f"     ⏱ {tleft()//60}m session")
    print(f"{'='*60}\n")

    pmsg = {
        "type": "paper", "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper["authors"], "link": paper["link"],
        "categories": paper["categories"],
        "time": now_hm(),
    }
    state["messages"].append(pmsg)
    bus.emit("newpaper", pmsg)

    history = [{"role": "user", "content":
        f'PAPER: "{paper["title"]}"\n'
        f'Authors: {paper["authors"]}\n'
        f'Abstract: {paper["abstract"]}'
    }]

    total_budget = MAX_UP - 600  # reserve 10 min for report
    phase_weights = [p["rounds"] for p in PHASES]
    total_weight = sum(phase_weights)
    phase_times = [int(total_budget * w / total_weight) for w in phase_weights]

    for phi, phase in enumerate(PHASES):
        if tleft() < 660:
            break

        state["phase"] = phase["name"]
        state["phase_idx"] = phi

        ph_msg = {
            "type": "phase", "name": phase["name"],
            "label": phase["label"], "desc": phase["desc"],
            "idx": phi, "total": len(PHASES),
            "time": now_hm(),
        }
        state["messages"].append(ph_msg)
        bus.emit("newphase", ph_msg)
        print(f"\n  {phase['label']} ({phase_times[phi]//60}m budget)")
        time.sleep(3)

        phase_start = time.time()
        phase_deadline = phase_start + phase_times[phi]
        rnd = 0

        while rnd < phase["rounds"] and time.time() < phase_deadline and tleft() > 660:

            if handle_user(paper, history):
                continue

            agent = AGENTS[rnd % len(AGENTS)]

            # pick target for cross-talk
            target = None
            others = [a for a in AGENTS if a["name"] != agent["name"]]
            for m in reversed(state["messages"][-10:]):
                if m.get("type") == "message" and m["speaker"] != agent["name"]:
                    target = next((a for a in others if a["name"] == m["speaker"]), None)
                    break
            if not target:
                target = random.choice(others)

            state["typing"] = agent["name"]
            bus.emit("typing", {
                "name": agent["name"], "avatar": agent["avatar"],
                "color": agent["color"],
            })

            prompt = random.choice(phase["prompts"]).format(target=target["name"])

            # build context from recent discussion
            recent = [
                f'{m["speaker"]}: {m["text"]}'
                for m in state["messages"][-12:]
                if m.get("type") == "message"
            ]
            recent_ctx = "\n".join(recent[-8:])

            system = (
                f"You are {agent['name']}, an ML researcher in a deep paper review session.\n"
                f"Tone: {agent['tone']}\n\n"
                f'Paper: "{paper["title"]}"\n'
                f"Authors: {paper['authors']}\n"
                f"Abstract: {paper['abstract']}\n\n"
                f"Phase: {phase['label']} — {phase['desc']}\n"
                f"{'Some humans are in the discussion too.' if users else ''}\n"
                f"Be substantive, specific, critical, honest. No generic statements.\n"
                f"Reference the actual paper content. Don't start with your name.\n"
                f"Under 120 words. Make every sentence count."
            )

            inst = f"{prompt}\n\nRecent discussion:\n{recent_ctx}" if recent_ctx else prompt

            # weave in user comments
            for m in reversed(state["messages"][-10:]):
                if m.get("type") == "user":
                    if random.random() < 0.3:
                        inst += f'\n(Human {m["user_name"]} said: "{m["text"]}" — address if relevant.)'
                    break

            try:
                text = llm(system, history, inst)
                stream_ai(agent, text, history)
                rnd += 1
            except Exception as e:
                print(f"    ✖ {e}")
                state["typing"] = None
                rnd += 1

            # gap
            if rnd < phase["rounds"] and time.time() < phase_deadline:
                nxt = AGENTS[rnd % len(AGENTS)]
                remaining_phase = max(0, int(phase_deadline - time.time()))
                bus.emit("waiting", {
                    "name": nxt["name"], "avatar": nxt["avatar"],
                    "color": nxt["color"], "gap": GAP,
                    "timeleft": tleft(),
                    "phase_left": remaining_phase,
                })
                deadline = time.time() + GAP
                while time.time() < deadline and tleft() > 660:
                    try:
                        u = user_queue.get(timeout=1)
                        user_queue.put(u)
                        break
                    except queue.Empty:
                        pass

        # phase done
        phase_msgs = len([m for m in state["messages"] if m.get("type") == "message"])
        dmsg = {
            "type": "system",
            "text": f"✅ {phase['label']} complete — {phase_msgs} total messages so far",
            "time": now_hm(),
        }
        state["messages"].append(dmsg)
        bus.emit("system", dmsg)
        time.sleep(3)

    # ━━ FINAL REPORT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    rpt_msg = {
        "type": "system",
        "text": "📝 Generating comprehensive analysis report... this may take a minute.",
        "time": now_hm(),
    }
    state["messages"].append(rpt_msg)
    bus.emit("system", rpt_msg)

    generate_report(paper, history)

    # shutdown
    cnt = len([m for m in state["messages"] if m.get("type") == "message"])
    ucnt = len([m for m in state["messages"] if m.get("type") == "user"])
    bus.emit("shutdown", {
        "total_msgs": cnt, "user_msgs": ucnt,
        "users": len(users), "paper": paper["title"],
        "phases_completed": state["phase_idx"] + 1,
    })
    print(f"\n⏰ Done. {cnt} agent msgs · {ucnt} human · {state['phase_idx']+1} phases.")

# ━━ FLASK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    sysmsg = {"type": "system", "text": f"👋 {name} joined the session", "time": now_hm()}
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
            "phase": state["phase"],
            "phase_idx": state["phase_idx"],
            "total_phases": len(PHASES),
            "agents": AGENTS,
            "messages": state["messages"][-200:],
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

# ━━ HTML ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Colloquium — Deep Paper Teardown</title>
<style>
:root{
  --bg:#0b141a;--hdr:#1f2c34;--in:#1f2c34;--out:#005c4b;
  --tx:#e9edef;--tx2:#8696a0;--grn:#00a884;--blu:#53bdeb;
  --sys:#182229;--brd:#2a3942;
}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:#111;color:var(--tx);display:flex;justify-content:center;
}
.overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.88);
  display:flex;align-items:center;justify-content:center;
  z-index:100;backdrop-filter:blur(8px);
}
.overlay.hidden{display:none}
.jc{
  background:var(--hdr);border:1px solid var(--brd);
  border-radius:16px;padding:2rem 1.8rem;
  width:90%;max-width:380px;text-align:center;
}
.jc h1{font-size:1.5rem;margin-bottom:.3rem}
.jc .ac{color:var(--grn)}
.jc .sub{color:var(--tx2);font-size:.78rem;margin-bottom:1.3rem;line-height:1.5}
.jc input{
  width:100%;padding:.75rem 1rem;background:var(--bg);
  border:1px solid var(--brd);border-radius:8px;
  color:var(--tx);font-size:1rem;outline:none;margin-bottom:.7rem;
}
.jc input:focus{border-color:var(--grn)}
.jc input::placeholder{color:var(--tx2)}
.btn{
  width:100%;padding:.75rem;border:none;border-radius:8px;
  font-size:.88rem;font-weight:600;cursor:pointer;margin-bottom:.4rem;
}
.btn:hover{opacity:.85}
.btn-go{background:var(--grn);color:#fff}
.btn-w{background:transparent;color:var(--tx2);border:1px solid var(--brd)}
.jc .ht{color:var(--tx2);font-size:.62rem;margin-top:.8rem;line-height:1.6}

.app{
  width:100%;max-width:500px;height:100vh;height:100dvh;
  display:flex;flex-direction:column;background:var(--bg);
  box-shadow:0 0 60px rgba(0,0,0,.6);position:relative;
}

/* header */
.hdr{
  display:flex;align-items:center;gap:.6rem;
  padding:.5rem .8rem;background:var(--hdr);min-height:56px;z-index:10;
}
.hdr-ava{
  width:40px;height:40px;border-radius:50%;background:var(--brd);
  display:flex;align-items:center;justify-content:center;font-size:1.1rem;
}
.hdr-info{flex:1;min-width:0}
.hdr-name{font-size:.95rem;font-weight:600}
.hdr-sub{font-size:.72rem;color:var(--tx2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hdr-sub .typ{color:var(--grn)}
.hdr-right{display:flex;align-items:center;gap:.3rem}
.bdg{padding:.15rem .5rem;border-radius:10px;font-weight:600;font-size:.65rem}
.b-die{background:rgba(255,68,68,.15);color:#f44}
.b-msg{background:rgba(0,168,132,.15);color:var(--grn)}
.b-eye{background:rgba(83,189,235,.12);color:var(--blu)}
.b-ph{background:rgba(255,152,0,.12);color:#ff9800}

/* phase bar */
.pbar{
  display:flex;gap:1px;padding:.2rem .8rem;
  background:rgba(0,0,0,.15);border-bottom:1px solid var(--brd);
}
.pbar .seg{
  flex:1;height:3px;border-radius:2px;
  background:var(--brd);transition:background .3s;position:relative;
}
.pbar .seg.done{background:var(--grn)}
.pbar .seg.on{background:var(--grn);animation:pls 1.5s ease infinite}
@keyframes pls{0%,100%{opacity:1}50%{opacity:.35}}

/* participants */
.parts{
  display:flex;gap:.3rem;padding:.3rem .8rem;
  background:rgba(0,0,0,.15);border-bottom:1px solid var(--brd);
  overflow-x:auto;font-size:.7rem;
}
.parts::-webkit-scrollbar{display:none}
.chip{
  display:flex;align-items:center;gap:.25rem;
  padding:.15rem .5rem;border-radius:12px;
  white-space:nowrap;background:rgba(255,255,255,.05);
  border:1px solid var(--brd);flex-shrink:0;
}
.chip .cdot{width:6px;height:6px;border-radius:50%;flex-shrink:0}

/* chat */
.chat{
  flex:1;overflow-y:auto;overflow-x:hidden;
  padding:.5rem .6rem;background:var(--bg);
}
.chat::-webkit-scrollbar{width:4px}
.chat::-webkit-scrollbar-thumb{background:var(--brd);border-radius:4px}

/* system pills */
.sys{text-align:center;margin:.7rem 0}
.pill{
  display:inline-block;background:var(--sys);color:var(--tx2);
  padding:.3rem .8rem;border-radius:8px;font-size:.75rem;
  max-width:90%;line-height:1.4;box-shadow:0 1px 1px rgba(0,0,0,.2);
}
.pill.ph{
  color:var(--tx);font-weight:600;font-size:.8rem;
  background:rgba(0,168,132,.08);border:1px solid rgba(0,168,132,.15);
  text-align:left;
}
.pill.ph .pd{color:var(--tx2);font-weight:400;font-size:.68rem;margin-top:2px}

/* paper card */
.pc{
  background:var(--hdr);border:1px solid var(--brd);
  border-radius:10px;padding:.8rem .9rem;margin:.6rem 0;
  box-shadow:0 2px 8px rgba(0,0,0,.3);
}
.pc .lab{
  display:inline-block;background:rgba(0,168,132,.15);color:var(--grn);
  padding:.12rem .5rem;border-radius:6px;font-size:.6rem;
  font-weight:700;letter-spacing:.05em;margin-bottom:.5rem;
}
.pc .pt{color:var(--tx);font-weight:600;font-size:.88rem;margin-bottom:.4rem;line-height:1.4}
.pc .pa{color:var(--blu);font-size:.68rem;margin-bottom:.5rem}
.pc .pabs{color:var(--tx2);font-size:.72rem;line-height:1.6;max-height:80px;overflow:hidden;transition:max-height .3s}
.pc .pabs.open{max-height:800px}
.pc .ptog{color:var(--grn);font-size:.65rem;cursor:pointer;margin-top:.3rem;display:inline-block}
.pc .ptog:hover{text-decoration:underline}
.pc .pcats{display:flex;gap:.2rem;margin-top:.5rem;flex-wrap:wrap}
.pc .pcat{
  padding:.1rem .4rem;border-radius:10px;font-size:.58rem;
  background:rgba(83,189,235,.08);color:var(--blu);
  border:1px solid rgba(83,189,235,.15);
}
.pc .plnk{
  display:inline-block;margin-top:.4rem;font-size:.62rem;
  color:var(--grn);text-decoration:none;
}
.pc .plnk:hover{text-decoration:underline}

/* report card */
.rpt{
  background:var(--hdr);border:1px solid var(--grn);
  border-radius:10px;padding:1rem;margin:.8rem 0;
  box-shadow:0 4px 20px rgba(0,168,132,.15);
}
.rpt .rpt-hdr{
  font-size:.9rem;font-weight:700;color:var(--grn);
  margin-bottom:.3rem;
}
.rpt .rpt-paper{font-size:.72rem;color:var(--tx2);margin-bottom:.8rem}
.rpt .rpt-sec{margin-bottom:.7rem}
.rpt .rpt-sec h3{
  font-size:.72rem;color:var(--grn);font-weight:700;
  letter-spacing:.04em;margin-bottom:.3rem;
  padding-bottom:.2rem;border-bottom:1px solid rgba(0,168,132,.2);
}
.rpt .rpt-sec p{font-size:.74rem;color:var(--tx);line-height:1.6}
.rpt .rpt-stats{
  margin-top:.6rem;padding-top:.5rem;
  border-top:1px solid var(--brd);
  font-size:.6rem;color:var(--tx2);
}

/* bubbles */
.msg{display:flex;flex-direction:column;margin-bottom:2px;animation:up .25s ease}
@keyframes up{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.msg.left{align-items:flex-start;padding-right:3rem}
.msg.right{align-items:flex-end;padding-left:3rem}
.msg .body{
  position:relative;padding:.35rem .5rem .15rem .55rem;
  border-radius:8px;max-width:100%;font-size:.9rem;
  line-height:1.35;word-wrap:break-word;
  box-shadow:0 1px 1px rgba(0,0,0,.15);
}
.msg.left .body{background:var(--in);border-top-left-radius:0}
.msg.right .body{background:var(--out);border-top-right-radius:0}
.msg.left .body::before{
  content:'';position:absolute;top:0;left:-7px;
  border-right:8px solid var(--in);border-bottom:8px solid transparent;
}
.msg.right .body::before{
  content:'';position:absolute;top:0;right:-7px;
  border-left:8px solid var(--out);border-bottom:8px solid transparent;
}
.msg.cont .body{border-radius:8px}
.msg.cont .body::before{display:none}
.msg.cont{margin-top:1px}
.msg .who{font-size:.78rem;font-weight:600;margin-bottom:1px}
.msg .txt{color:var(--tx)}
.msg .meta{
  float:right;display:flex;align-items:center;gap:3px;
  margin-left:8px;margin-top:3px;font-size:.62rem;
  color:rgba(255,255,255,.4);white-space:nowrap;
}
.msg .ticks{color:var(--blu);font-size:.68rem}
.msg .spacer{display:inline-block;width:4.2rem;height:1px}

.msg.umsg.right .body{background:#1a3a3a}
.msg.umsg.right .body::before{border-left-color:#1a3a3a}
.msg.umsg.left .body{background:#1a2a3a}
.msg.umsg.left .body::before{border-right-color:#1a2a3a}

.cursor{
  display:inline-block;width:2px;height:.95em;background:var(--grn);
  margin-left:1px;animation:bk .7s step-end infinite;vertical-align:text-bottom;
}
@keyframes bk{0%,100%{opacity:1}50%{opacity:0}}

.tdots{display:inline-flex;gap:3px;align-items:center;padding:4px 0}
.tdots span{width:7px;height:7px;border-radius:50%;background:var(--tx2);animation:dp 1.4s ease-in-out infinite}
.tdots span:nth-child(2){animation-delay:.2s}
.tdots span:nth-child(3){animation-delay:.4s}
@keyframes dp{0%,80%,100%{opacity:.3;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}

/* input bar */
.ibar{
  display:flex;align-items:center;gap:.5rem;
  padding:.45rem .6rem;background:var(--hdr);
  border-top:1px solid var(--brd);min-height:52px;
}
.ibar.off{opacity:.4;pointer-events:none}
.iwrap{
  flex:1;display:flex;align-items:center;background:var(--bg);
  border:1px solid var(--brd);border-radius:22px;padding:.1rem .2rem .1rem .8rem;
}
.iwrap:focus-within{border-color:var(--grn)}
.iwrap input{
  flex:1;background:none;border:none;outline:none;
  color:var(--tx);font-size:.9rem;padding:.55rem 0;
}
.iwrap input::placeholder{color:var(--tx2)}
.sbtn{
  width:42px;height:42px;border-radius:50%;border:none;
  background:var(--grn);color:#fff;font-size:1.2rem;
  cursor:pointer;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;
}
.sbtn:hover{opacity:.85}
.sbtn:disabled{opacity:.3;cursor:default}

.wbar{
  display:flex;align-items:center;justify-content:center;
  padding:.5rem .8rem;background:var(--hdr);
  border-top:1px solid var(--brd);min-height:48px;
  font-size:.78rem;color:var(--tx2);gap:.4rem;
}
.wbar .live{color:var(--grn);font-weight:600}
.wbar .jl{color:var(--grn);cursor:pointer;text-decoration:underline;margin-left:.5rem}

.scb{
  display:none;position:absolute;bottom:68px;right:14px;
  width:40px;height:40px;background:var(--hdr);border:1px solid var(--brd);
  border-radius:50%;align-items:center;justify-content:center;
  cursor:pointer;z-index:20;box-shadow:0 2px 8px rgba(0,0,0,.5);
  font-size:1.1rem;color:var(--tx2);
}
.scb:hover{background:var(--brd)}
.scb .ub{
  position:absolute;top:-5px;right:-5px;background:var(--grn);
  color:#fff;font-size:.58rem;font-weight:700;
  min-width:18px;height:18px;border-radius:9px;
  display:flex;align-items:center;justify-content:center;padding:0 4px;
}

.sd{
  text-align:center;padding:1.5rem 1rem;
  background:rgba(255,68,68,.06);border-top:1px solid rgba(255,68,68,.15);
  margin-top:.8rem;
}
.sd .big{color:#f44;font-size:.88rem;font-weight:600}
.sd .sm{color:var(--tx2);font-size:.68rem;margin-top:.3rem}

@media(max-width:500px){.app{max-width:100%}.msg .body{font-size:.87rem}}
@media(min-width:501px){body{align-items:center;padding:1rem 0}.app{border-radius:12px;height:96vh;overflow:hidden}}
</style>
</head><body>

<div class="overlay" id="ov">
  <div class="jc">
    <h1>🔬 <span class="ac">Colloquium</span></h1>
    <div class="sub">
      6-hour deep teardown of one ML paper<br>
      5 researchers · 9 phases · no mercy
    </div>
    <input type="text" id="nin" placeholder="Your name..." maxlength="20"
      onkeydown="if(event.key==='Enter')doJ()">
    <button class="btn btn-go" onclick="doJ()">Join Session</button>
    <button class="btn btn-w" onclick="doW()">Just Watch</button>
    <div class="ht">
      🔹 Aria · 🔸 Kai · 🟢 Noor · 🔻 Sasha · 🟣 Ravi<br>
      First Read → Claims → Method → Math → Experiments →<br>
      Novelty/BS → Strengths → Open Debate → Verdict<br>
      Full analysis report generated at session end
    </div>
  </div>
</div>

<div class="app" id="app" style="display:none">
  <div class="hdr">
    <div class="hdr-ava">🔬</div>
    <div class="hdr-info">
      <div class="hdr-name">Colloquium</div>
      <div class="hdr-sub" id="hsub">connecting...</div>
    </div>
    <div class="hdr-right">
      <div class="bdg b-ph" id="bph">📋 0/9</div>
      <div class="bdg b-eye" id="beye">👁 0</div>
      <div class="bdg b-msg" id="bmsg">💬 0</div>
      <div class="bdg b-die" id="bdie">💀 --</div>
    </div>
  </div>

  <div class="pbar" id="pbar"></div>
  <div class="parts" id="parts"></div>
  <div class="chat" id="chat"></div>

  <div class="ibar" id="ibar" style="display:none">
    <div class="iwrap">
      <input type="text" id="min" placeholder="Ask a question or comment..." maxlength="500"
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

let myId=null,myN=null,myC=null,joined=false;
let agents=[],boot=0,maxU=0,mc=0,lw='',tIv=null,nPh=9;
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
    $('bdie').textContent='💀 '+fmt(l);
    if(l<300)$('bdie').style.background='rgba(255,68,68,.3)';
    if(l<=0){$('bdie').textContent='💀 DEAD';clearInterval(tIv)}
  },1000);
}
function sH(h){$('hsub').innerHTML=h}
function sW(h){$('wtxt').innerHTML=h}

function uPB(idx,tot){
  nPh=tot||nPh;
  let h='';for(let i=0;i<nPh;i++){
    let c='';if(i<idx)c='done';else if(i===idx)c='on';
    h+=`<div class="seg ${c}"></div>`;
  }
  $('pbar').innerHTML=h;
  $('bph').textContent='📋 '+(idx+1)+'/'+nPh;
}

function rParts(ul){
  let h='';
  agents.forEach(a=>{
    h+=`<div class="chip"><div class="cdot" style="background:${a.color}"></div>${a.avatar} ${a.name}</div>`;
  });
  (ul||[]).forEach(u=>{
    const me=(myN&&u.name===myN)?' (you)':'';
    h+=`<div class="chip"><div class="cdot" style="background:${u.color}"></div>${u.name}${me}</div>`;
  });
  $('parts').innerHTML=h;
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
  d.innerHTML=
    `<div class="lab">TODAY'S PAPER · 6-HOUR DEEP DIVE</div>`+
    `<div class="pt">${p.title}</div>`+
    `<div class="pa">${p.authors}</div>`+
    `<div class="pabs" id="${id}">${p.abstract}</div>`+
    `<span class="ptog" onclick="var e=document.getElementById('${id}');if(e.classList.contains('open')){e.classList.remove('open');this.textContent='show full abstract ▾'}else{e.classList.add('open');this.textContent='collapse ▴'}">show full abstract ▾</span>`+
    `<div class="pcats">${(p.categories||[]).map(c=>'<span class="pcat">'+c+'</span>').join('')}</div>`+
    (p.link?`<a class="plnk" href="${p.link}" target="_blank" rel="noopener">→ view on arXiv</a>`:'');
  chat.appendChild(d);scr();
}
function rptCard(r){
  const d=document.createElement('div');d.className='rpt';
  let h=`<div class="rpt-hdr">📝 ANALYSIS REPORT</div>`;
  h+=`<div class="rpt-paper">${r.paper_title} — ${r.paper_authors}</div>`;
  (r.sections||[]).forEach(s=>{
    h+=`<div class="rpt-sec"><h3>${s.title}</h3><p>${s.content}</p></div>`;
  });
  h+=`<div class="rpt-stats">${r.total_messages} messages analyzed · ${r.time} UTC</div>`;
  if(r.paper_link)h+=`<a class="plnk" href="${r.paper_link}" target="_blank" style="display:block;margin-top:.4rem;font-size:.62rem">→ original paper</a>`;
  d.innerHTML=h;chat.appendChild(d);scr();
}

function sd(name){
  if(myN&&name===myN)return'right';
  return'left';
}
function rmT(){if(tBub){tBub.remove();tBub=null}}
function addT(n,av,cl){
  rmT();const co=(lw===n),s=sd(n);
  const d=document.createElement('div');d.className=`msg ${s}${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${cl}">${av} ${n}</div>`;
  h+=`<div class="body"><div class="tdots"><span></span><span></span><span></span></div></div>`;
  d.innerHTML=h;chat.appendChild(d);tBub=d;scr();
}
function sBub(sp,av,cl,tm){
  rmT();const co=(lw===sp),s=sd(sp);
  const d=document.createElement('div');d.className=`msg ${s}${co?' cont':''}`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${cl}">${av} ${sp}</div>`;
  h+=`<div class="body"><span class="meta"><span class="tm">${tm}</span></span><span class="txt"></span><span class="cursor"></span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);cBub=d;cTxt=d.querySelector('.txt');scr();
}
function addWd(w){if(!cTxt)return;const t=cTxt.textContent;cTxt.textContent=t?(t+' '+w):w;scr()}
function fBub(sp,tm){
  if(cBub){const c=cBub.querySelector('.cursor');if(c)c.remove();const m=cBub.querySelector('.meta');if(m)m.innerHTML=`<span class="tm">${tm}</span><span class="ticks"> ✓✓</span>`}
  lw=sp;cBub=null;cTxt=null;mc++;$('bmsg').textContent='💬 '+mc;ntf();scr();
}
function fMsg(m){
  const w=m.speaker||m.user_name,isU=(m.type==='user'),co=(lw===w),s=isU?(myN&&m.user_name===myN?'right':'left'):sd(w);
  const d=document.createElement('div');d.className=`msg ${s}${co?' cont':''}${isU?' umsg':''}`;
  const cl=m.color||'#aaa',av=m.avatar||'';
  let h='';
  if(!co)h+=`<div class="who" style="color:${cl}">${av}${av?' ':''}${w}</div>`;
  h+=`<div class="body"><span class="meta"><span class="tm">${m.time||''}</span><span class="ticks"> ✓✓</span></span><span class="txt">${m.text}</span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);lw=w;if(!isU)mc++;
}
function uBub(m){
  const me=(myN&&m.user_name===myN),s=me?'right':'left',co=(lw===m.user_name);
  const d=document.createElement('div');d.className=`msg ${s}${co?' cont':''} umsg`;
  let h='';
  if(!co)h+=`<div class="who" style="color:${m.color}">${m.user_name}${me?' (you)':''}</div>`;
  h+=`<div class="body"><span class="meta"><span class="tm">${m.time||''}</span><span class="ticks"> ✓✓</span></span><span class="txt">${m.text}</span><span class="spacer"></span></div>`;
  d.innerHTML=h;chat.appendChild(d);lw=m.user_name;scr();
}

function sse(){
  sH('connecting...');
  const es=new EventSource('/stream');

  es.addEventListener('fullstate',e=>{
    const d=JSON.parse(e.data);
    boot=d.boot;maxU=d.max_up;agents=d.agents||[];nPh=d.total_phases||9;
    rParts(d.users);
    $('beye').textContent='👁 '+(d.viewers||0);
    if(d.phase_idx>=0)uPB(d.phase_idx,nPh);
    else{let h='';for(let i=0;i<nPh;i++)h+=`<div class="seg"></div>`;$('pbar').innerHTML=h}
    sH(d.paper?d.paper.title.substring(0,45)+'...':'loading paper...');
    chat.innerHTML='';mc=0;lw='';
    sPill('🔬 <b>Colloquium</b> — deep paper teardown','');
    if(d.messages)d.messages.forEach(m=>{
      if(m.type==='paper')pCard(m);
      else if(m.type==='phase')phPill(m);
      else if(m.type==='message')fMsg(m);
      else if(m.type==='user')fMsg(m);
      else if(m.type==='system')sPill(m.text,'');
      else if(m.type==='report')rptCard(m);
    });
    $('bmsg').textContent='💬 '+mc;stmr();scr();sW('analysis in progress');
  });

  es.addEventListener('newpaper',e=>{
    const d=JSON.parse(e.data);pCard(d);
    sH(d.title.substring(0,45)+'...');
  });
  es.addEventListener('newphase',e=>{
    const d=JSON.parse(e.data);phPill(d);
    if(typeof d.idx==='number')uPB(d.idx,d.total||nPh);
  });
  es.addEventListener('typing',e=>{
    const d=JSON.parse(e.data);addT(d.name,d.avatar,d.color);
    sH(`<span class="typ">${d.avatar} ${d.name} thinking...</span>`);
    sW(`${d.avatar} ${d.name} analyzing...`);
  });
  es.addEventListener('msgstart',e=>{
    const d=JSON.parse(e.data);sBub(d.speaker,d.avatar,d.color,d.time);
    sH(`<span class="typ">${d.avatar} ${d.speaker} writing...</span>`);
    sW(`${d.avatar} ${d.speaker} writing...`);
  });
  es.addEventListener('word',e=>{addWd(JSON.parse(e.data).w)});
  es.addEventListener('msgdone',e=>{
    const d=JSON.parse(e.data);fBub(d.speaker,d.time);
    sH(document.querySelector('.pc .pt')?.textContent?.substring(0,45)+'...'||'roundtable');
    sW('analysis in progress');
  });
  es.addEventListener('usermsg',e=>{uBub(JSON.parse(e.data))});
  es.addEventListener('system',e=>{sPill(JSON.parse(e.data).text,'')});
  es.addEventListener('report',e=>{rptCard(JSON.parse(e.data))});
  es.addEventListener('presence',e=>{
    const d=JSON.parse(e.data);rParts(d.users);
    $('beye').textContent='👁 '+(d.viewers||0);
  });
  es.addEventListener('waiting',e=>{
    const d=JSON.parse(e.data);let g=d.gap;
    sW(`${d.avatar} <span style="color:${d.color}">${d.name}</span> in <span id="gcd">${g}s</span>`);
    const iv=setInterval(()=>{g--;const el=document.getElementById('gcd');if(el)el.textContent=g+'s';if(g<=0){clearInterval(iv);sW('next...')}},1000);
  });
  es.addEventListener('shutdown',e=>{
    const d=JSON.parse(e.data);rmT();
    const div=document.createElement('div');div.className='sd';
    div.innerHTML=`<div class="big">⏱ Session Complete</div>`+
      `<div class="sm">"${d.paper}"</div>`+
      `<div class="sm">${d.total_msgs} agent messages · ${d.user_msgs} human · ${d.phases_completed} phases · ${d.users} participants</div>`;
    chat.appendChild(div);scr();
    sH('session ended');sW('offline');
    $('bdie').textContent='💀 DEAD';
    if(tIv)clearInterval(tIv);
    if(joined){$('ibar').classList.add('off');$('min').placeholder='Session ended'}
    uPB(nPh,nPh);
  });
  es.addEventListener('ping',e=>{$('beye').textContent='👁 '+(JSON.parse(e.data).v||0)});
  es.onerror=()=>{sH('reconnecting...');sW('reconnecting...');es.close();setTimeout(sse,3000)};
}

$('nin').focus();
</script>
</body></html>"""

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 Colloquium — Deep Single-Paper Teardown")
    print(f"   model       : {MODEL}")
    print(f"   backup      : {BACKUP}")
    print(f"   gap         : {GAP}s")
    print(f"   phases      : {len(PHASES)} ({', '.join(p['name'] for p in PHASES)})")
    print(f"   agents      : {', '.join(a['avatar']+' '+a['name'] for a in AGENTS)}")
    print(f"   max uptime  : {MAX_UP//3600}h {MAX_UP%3600//60}m")
    print(f"   report      : generated at end")
    print("=" * 60)
    import sys
    sys.stdout.flush()

    threading.Thread(target=engine, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)



# ━━ STATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
