"""AgentLens — Chat-first Streamlit frontend."""

import json
import os
import time

import requests
import streamlit as st

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

st.set_page_config(
    page_title="AgentLens",
    page_icon="🔭",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Layout */
    .block-container { padding-top: 1.5rem; padding-bottom: 6rem; max-width: 820px; }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #2563eb; border: none; border-radius: 8px;
        font-weight: 600; font-size: 0.88rem; padding: 0.5rem 1.4rem;
        transition: background 0.15s, box-shadow 0.15s, transform 0.1s;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8; box-shadow: 0 4px 14px rgba(37,99,235,0.4);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"]:active { transform: translateY(0); }

    /* Secondary button */
    .stButton > button[kind="secondary"] {
        border-radius: 8px; border: 1px solid #1e2d4a; font-size: 0.88rem;
        transition: border-color 0.15s, background 0.15s;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #2563eb; background: rgba(37,99,235,0.07);
    }

    /* Example prompt buttons — borderless, left-aligned */
    [data-testid="stBaseButton-secondary"][data-example="true"] {
        background: transparent !important; border: none !important;
        text-align: left !important; color: #64748b !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #111827; border: 1px solid #1e2d4a;
        border-radius: 10px; padding: 12px 16px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important; font-weight: 700 !important; color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important; font-weight: 700 !important; color: #475569 !important;
        text-transform: uppercase; letter-spacing: 0.08em;
    }

    /* Expanders */
    .stExpander { border: 1px solid #1e2d4a !important; border-radius: 8px !important; }
    .stExpander summary { font-size: 0.83rem !important; color: #94a3b8 !important; }
    .stExpander summary:hover { color: #f1f5f9 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #1e2d4a; gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.81rem; padding: 6px 14px; color: #475569;
        border-radius: 6px 6px 0 0; border: none; transition: all 0.15s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #cbd5e1; background: rgba(255,255,255,0.04); }
    .stTabs [aria-selected="true"] {
        color: #f1f5f9 !important; background: rgba(37,99,235,0.1) !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Alert boxes */
    [data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.88rem !important; }

    /* Divider */
    hr { border-color: #1e2d4a !important; margin: 1.25rem 0 !important; }

    /* Status widget */
    [data-testid="stStatusWidget"], [data-testid="stStatus"] {
        border-radius: 10px !important; border: 1px solid #1e2d4a !important;
    }

    /* Caption */
    .stCaption { color: #475569 !important; font-size: 0.76rem !important; }

    /* Chat input */
    [data-testid="stChatInput"] { border-radius: 12px !important; }
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important; border: 1.5px solid #1e2d4a !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }

    /* Chat message avatar */
    [data-testid="stChatMessageContent"] { font-size: 0.92rem; line-height: 1.65; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages":        [],    # stored conversation history
    "phase":           "input",
    "thread_id":       None,
    "current_outputs": {},    # accumulates during active stream
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Constants ──────────────────────────────────────────────────────────────────
NODE_LABELS: dict[str, str | None] = {
    "intent_classifier": "Intent Check",
    "moonshot_evaluator": "Moonshot Gate",
    "chat_responder": None,          # suppress from stream log
    "technical":     "Technical Feasibility",
    "market":        "Market Opportunity",
    "risk":          "Risk Assessment",
    "cost":          "Cost Estimation",
    "rag":           "Knowledge Retrieval",
    "kill_shot_agent": "Kill Shot Experiment",
    "rejection_report": None,        # suppress
    "parallel_spawn": None,
    "human_review":  None,
}

TERMINAL = {"hitl_pause", "pipeline_complete", "pipeline_rejected"}

RESULT_TABS = [
    ("🎯 Kill Shot",  "kill_shot_agent"),
    ("🌙 Gate",       "moonshot_evaluator"),
    ("⚙️ Technical",  "technical"),
    ("📈 Market",     "market"),
    ("⚠️ Risk",       "risk"),
    ("💰 Cost",       "cost"),
    ("📚 Knowledge",  "rag"),
]

EXAMPLES = [
    "Solid-state batteries giving EVs a 1,000-mile range",
    "Ocean plastic collection at industrial scale using autonomous ships",
    "Brain-computer interface for restoring speech in paralysis patients",
    "Carbon capture from air at $50/ton using engineered algae",
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def _parse_sse(resp):
    buf = ""
    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        buf += chunk
        while "\n\n" in buf:
            msg, buf = buf.split("\n\n", 1)
            for line in msg.splitlines():
                if line.startswith("data:"):
                    try:
                        yield json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        pass


def _absorb(ev: dict) -> None:
    """Store node output into current_outputs."""
    if ev.get("type") in ("node_completed", "node_failed"):
        st.session_state.current_outputs[ev.get("node") or ""] = ev.get("output") or {}


def _usd(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v >= 1e9: return f"${v/1e9:.1f}B"
    if v >= 1e6: return f"${v/1e6:.1f}M"
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


def _word_stream(text: str, delay: float = 0.028):
    """Word-by-word generator for st.write_stream()."""
    words = text.split()
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


# ── Renderers ──────────────────────────────────────────────────────────────────

def _render_tab(node: str, output: dict) -> None:
    """Render detailed agent output for one tab."""
    if node == "moonshot_evaluator":
        me = output.get("moonshot_evaluation") or {}
        if me.get("passes_moonshot_gate"):
            st.success("Passes all three criteria.")
        else:
            st.error(me.get("gate_failure_reason", "Failed."))
        c1, c2, c3 = st.columns(3)
        c1.metric("Problem Real?",     "Yes ✅" if me.get("problem_is_real")      else "No ❌")
        c2.metric("Solution Feasible?","Yes ✅" if me.get("solution_is_feasible") else "No ❌")
        c3.metric("Tech Available?",   "Yes ✅" if me.get("technology_is_available") else "No ❌")
        if me.get("confidence_level"):
            st.caption(f"Confidence: {me['confidence_level']}")

    elif node == "technical":
        t = output.get("technical_output") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("TRL",         f"{t.get('trl_level','—')} / 9")
        c2.metric("Confidence",  (t.get("confidence_level") or "—").title())
        c3.metric("Yrs to Prototype", t.get("time_to_prototype_years","—"))
        if t.get("trl_justification"):
            st.caption(t["trl_justification"])
        if t.get("key_blockers"):
            st.write("**Key blockers**")
            for b in t["key_blockers"]: st.write(f"- {b}")
        if t.get("required_breakthroughs"):
            with st.expander("Required breakthroughs"):
                for b in t["required_breakthroughs"]: st.write(f"- {b}")

    elif node == "market":
        m = output.get("market_output") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("TAM", _usd(m.get("tam_usd")))
        c2.metric("SAM", _usd(m.get("sam_usd")))
        c3.metric("SOM (5yr)", _usd(m.get("som_usd")))
        c4, c5 = st.columns(2)
        c4.metric("Growth", f"{m.get('market_growth_rate_pct','—')}%/yr")
        c5.metric("Time to Market", f"{m.get('time_to_market_years','—')} yrs")
        if m.get("competitive_moat"):
            st.info(f"**Moat:** {m['competitive_moat']}")
        if m.get("top_competitors"):
            st.write("**Top competitors**")
            for comp in m["top_competitors"]: st.write(f"- {comp}")

    elif node == "risk":
        r = output.get("risk_output") or {}
        level = r.get("overall_risk_level","—")
        badge = {"low":"🟢 Low","medium":"🟡 Medium","high":"🔴 High",
                 "very_high":"🔴 Very High"}.get(level, level)
        c1, c2 = st.columns(2)
        c1.metric("Overall Risk", badge)
        c2.metric("Confidence",   (r.get("confidence_level") or "—").title())
        if r.get("top_risk"):
            st.warning(f"**Top risk:** {r['top_risk']}")
        for cat, key in [("Technical","technical_risks"),("Regulatory","regulatory_risks"),
                         ("Financial","financial_risks"),("Market","market_risks")]:
            risks = r.get(key) or []
            if risks:
                with st.expander(f"{cat} risks ({len(risks)})"):
                    for risk in risks:
                        if isinstance(risk, dict):
                            dot = {"high":"🔴","medium":"🟡","low":"🟢"}.get(
                                risk.get("likelihood",""), "⚪")
                            st.write(f"{dot} **{risk.get('risk_name','')}** — "
                                     f"{risk.get('description','')}")
                            st.caption(f"Mitigation: {risk.get('mitigation','')}")

    elif node == "cost":
        c = output.get("cost_output") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("CAPEX",      _usd(c.get("capex_total_usd")))
        c2.metric("OPEX/yr",    _usd(c.get("opex_annual_usd")))
        c3.metric("Break-even", f"{c.get('break_even_years','—')} yrs")
        c4, c5 = st.columns(2)
        c4.metric("Unit Cost",
                  f"{_usd(c.get('unit_cost_usd'))} {c.get('unit_description','')}")
        c5.metric("Scale", c.get("production_scale","—"))
        lo, hi = c.get("capex_low_usd"), c.get("capex_high_usd")
        if lo and hi:
            st.caption(f"CAPEX range: {_usd(lo)} – {_usd(hi)}"
                       f"  ·  Confidence: {c.get('confidence_level','—')}")

    elif node == "rag":
        r = output.get("rag_output") or {}
        if r.get("grounded"):
            n = r.get("total_retrieved", 0)
            st.success(f"Grounded — {n} relevant chunks retrieved.")
            if r.get("query_used"):
                st.caption(f"Query: _{r['query_used']}_")
            if r.get("evidence_citations"):
                with st.expander("Sources"):
                    for src in r["evidence_citations"]: st.write(f"- `{src}`")
        else:
            st.warning("No matching documents in knowledge base.")
            st.caption("Run `python -m src.knowledge_base.ingestor --ingest-domain` to populate it.")

    elif node == "kill_shot_agent":
        k = output.get("kill_shot") or {}
        if k:
            st.markdown(f"**{k.get('critical_assumption','—')}**")
            st.caption(f"Why: {k.get('why_this_assumption','—')}")
            st.divider()
            st.write(k.get("experiment_description","—"))
            c1, c2 = st.columns(2)
            c1.metric("Cost",     _usd(k.get("estimated_cost_usd")))
            c2.metric("Duration", f"{k.get('estimated_duration_weeks','—')} weeks")
            col1, col2 = st.columns(2)
            with col1: st.success(f"**Success:** {k.get('success_criteria','—')}")
            with col2: st.error(f"**Failure:** {k.get('failure_criteria','—')}")
            if k.get("required_resources"):
                with st.expander("Resources"):
                    for r in k["required_resources"]: st.write(f"- {r}")
    else:
        st.info("No data.")


def _full_tabs(outputs: dict, include_kill_shot: bool = False) -> None:
    visible = [
        (lbl, node) for lbl, node in RESULT_TABS
        if node in outputs and (include_kill_shot or node != "kill_shot_agent")
    ]
    if not visible:
        return
    tabs = st.tabs([lbl for lbl, _ in visible])
    for tab, (_, node) in zip(tabs, visible):
        with tab:
            _render_tab(node, outputs.get(node) or {})


def _analysis_summary(outputs: dict, stream_kill_shot: bool = False) -> None:
    """
    Compact one-liner per agent, always visible.
    Kill shot is the hero — streams word by word on first render if stream_kill_shot=True.
    Full detail in a collapsible expander.
    """
    lines = []

    me = (outputs.get("moonshot_evaluator") or {}).get("moonshot_evaluation") or {}
    if me:
        icon = "✅" if me.get("passes_moonshot_gate") else "❌"
        reason = me.get("gate_failure_reason","")
        suffix = f" — {reason[:70]}" if reason and not me.get("passes_moonshot_gate") else ""
        lines.append(f"🌙 **Gate** {icon}{suffix}")

    t = (outputs.get("technical") or {}).get("technical_output") or {}
    if t:
        lines.append(
            f"⚙️ **Technical** — TRL {t.get('trl_level','—')}/9 · "
            f"{t.get('time_to_prototype_years','—')} yrs to prototype · "
            f"confidence {t.get('confidence_level','—')}"
        )

    m = (outputs.get("market") or {}).get("market_output") or {}
    if m:
        lines.append(
            f"📈 **Market** — TAM {_usd(m.get('tam_usd'))} · "
            f"SOM {_usd(m.get('som_usd'))} · {m.get('market_growth_rate_pct','—')}%/yr"
        )

    r = (outputs.get("risk") or {}).get("risk_output") or {}
    if r:
        level = r.get("overall_risk_level","—").replace("_"," ").title()
        lines.append(f"⚠️ **Risk** — {level}")

    c = (outputs.get("cost") or {}).get("cost_output") or {}
    if c:
        lines.append(
            f"💰 **Cost** — CAPEX {_usd(c.get('capex_total_usd'))} · "
            f"break-even {c.get('break_even_years','—')} yrs"
        )

    rag = (outputs.get("rag") or {}).get("rag_output") or {}
    if rag:
        n = rag.get("total_retrieved", 0)
        lines.append(
            f"📚 **Knowledge** — {'Grounded' if rag.get('grounded') else 'No match'} ({n} chunks)"
        )

    for line in lines:
        st.markdown(line)

    # Kill shot — the hero output
    k = (outputs.get("kill_shot_agent") or {}).get("kill_shot") or {}
    if k:
        st.divider()
        assumption = k.get("critical_assumption", "—")
        st.markdown(
            "<p style='font-size:0.65rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.1em;color:#ef4444;margin:0 0 4px'>Critical Assumption</p>",
            unsafe_allow_html=True,
        )
        if stream_kill_shot:
            st.write_stream(_word_stream(assumption))
        else:
            st.markdown(f"**{assumption}**")
        st.caption(f"Why: {k.get('why_this_assumption','—')}")
        st.write(f"**Experiment:** {k.get('experiment_description','—')}")
        c1, c2 = st.columns(2)
        c1.metric("Cost",     _usd(k.get("estimated_cost_usd")))
        c2.metric("Duration", f"{k.get('estimated_duration_weeks','—')} weeks")

    # Full details collapsible
    has_data = any(
        outputs.get(n) for n in
        ["technical","market","risk","cost","rag","moonshot_evaluator","kill_shot_agent"]
    )
    if has_data:
        st.write("")
        with st.expander("Full analysis details"):
            _full_tabs(outputs, include_kill_shot=bool(k))


# ── Live streaming ─────────────────────────────────────────────────────────────

def _stream(thread_id: str, label: str = "Analyzing…") -> tuple[str | None, bool]:
    """
    Stream pipeline events inside a st.status() widget.
    Detects chat vs analysis path from the intent classifier output.
    Returns (terminal_event_type, is_chat_path).
    """
    terminal = None
    is_chat = False

    with st.status(label, expanded=True) as sw:
        try:
            with requests.get(
                f"{API_URL}/evaluate/{thread_id}/stream", stream=True, timeout=300
            ) as resp:
                resp.raise_for_status()
                for ev in _parse_sse(resp):
                    ev_type = ev["type"]
                    node   = ev.get("node") or ""
                    output = ev.get("output") or {}
                    lbl    = NODE_LABELS.get(node, node) or ""

                    # Detect chat path from intent classifier result
                    if ev_type == "node_completed" and node == "intent_classifier":
                        if output.get("conversation_type") == "chat":
                            is_chat = True
                            sw.update(label="Reading your message…")

                    _absorb(ev)
                    if ev_type in TERMINAL:
                        # Merge terminal state into current_outputs
                        for k, v in (ev.get("output") or {}).items():
                            if k not in st.session_state.current_outputs:
                                st.session_state.current_outputs[k] = v

                    # Show pipeline progress only on the analysis path
                    if not is_chat:
                        if ev_type == "node_started" and lbl:
                            st.write(f"🔄  {lbl}…")
                        elif ev_type == "node_completed" and lbl:
                            st.write(f"✅  {lbl}")
                        elif ev_type == "node_failed" and lbl:
                            st.write(f"❌  {lbl} failed")

                    if ev_type in TERMINAL:
                        terminal = ev_type
                        break

            _sw = {
                "pipeline_complete": ("Complete", "complete"),
                "pipeline_rejected": ("Rejected",  "error"),
                "hitl_pause":        ("Ready for review", "complete"),
            }
            if not is_chat and terminal in _sw:
                sw.update(label=_sw[terminal][0], state=_sw[terminal][1])
            else:
                sw.update(label="Done", state="complete")

        except Exception as exc:
            st.error(f"Stream error: {exc}")
            sw.update(label="Error", state="error")

    return terminal, is_chat


# ── Stored message renderer ────────────────────────────────────────────────────

def _replay_messages() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            t = msg.get("type", "text")
            if msg["role"] == "user" or t == "chat":
                st.write(msg["content"])
            elif t in ("analysis", "rejected"):
                _analysis_summary(msg.get("outputs", {}), stream_kill_shot=False)
                if t == "rejected":
                    reason = msg.get("reason", "")
                    if reason:
                        st.error(f"**Rejected:** {reason}")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h3 style='font-size:1rem;font-weight:700;margin:0'>🔭 AgentLens</h3>"
        "<p style='font-size:0.73rem;color:#475569;margin:2px 0 0'>Moonshot idea analysis</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "<p style='font-size:0.73rem;color:#475569;line-height:1.55'>"
        "A LangGraph pipeline of 5 specialist agents: "
        "technical feasibility, market sizing, risk, cost estimation, "
        "and knowledge retrieval. Modeled after X Moonshot Factory's "
        "evaluation process. Human-in-the-loop before the Kill Shot experiment."
        "</p>",
        unsafe_allow_html=True,
    )

    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        st.divider()
        st.caption("THIS SESSION")
        for m in user_msgs[-8:]:
            preview = m["content"][:48] + ("…" if len(m["content"]) > 48 else "")
            st.markdown(
                f"<p style='font-size:0.76rem;color:#94a3b8;margin:3px 0;"
                f"padding:3px 8px;border-left:2px solid #1e2d4a'>💭 {preview}</p>",
                unsafe_allow_html=True,
            )
        st.write("")
        if st.button("Clear history", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = type(v)()
            st.rerun()

    st.divider()
    st.caption("PIPELINE STAGES")
    for stage in [
        "🛡️ Intent safety check",
        "🌙 Moonshot gate (3 criteria)",
        "⚙️ 📈 ⚠️ 💰 📚  5 agents in parallel",
        "👤 Human review (HITL)",
        "🎯 Kill Shot experiment",
    ]:
        st.markdown(
            f"<p style='font-size:0.73rem;color:#475569;margin:3px 0'>{stage}</p>",
            unsafe_allow_html=True,
        )


# ── Main ───────────────────────────────────────────────────────────────────────

phase = st.session_state.phase

# Replay stored conversation history
_replay_messages()

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.messages and phase == "input":
    st.markdown(
        "<h1 style='font-size:1.75rem;font-weight:700;margin-bottom:4px'>"
        "What's your moonshot?</h1>"
        "<p style='color:#475569;font-size:0.95rem;margin-bottom:1.5rem'>"
        "Describe an idea and I'll run a full techno-economic analysis "
        "— technical feasibility, market opportunity, risk, cost, and a "
        "Kill Shot experiment to test the critical assumption.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.73rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.08em;color:#475569;margin-bottom:8px'>Try one of these</p>",
        unsafe_allow_html=True,
    )
    for ex in EXAMPLES:
        if st.button(f"↗  {ex}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            try:
                r = requests.post(f"{API_URL}/evaluate", json={"idea": ex}, timeout=30)
                r.raise_for_status()
                st.session_state.thread_id = r.json()["thread_id"]
                st.session_state.current_outputs = {}
                st.session_state.phase = "streaming"
                st.rerun()
            except Exception as exc:
                st.session_state.messages.pop()
                st.error(f"Backend error: {exc}")

# ── Streaming phase ────────────────────────────────────────────────────────────
elif phase == "streaming":
    with st.chat_message("assistant"):
        terminal, is_chat = _stream(st.session_state.thread_id)

        if is_chat:
            # Stream the chat response word by word right into the chat bubble
            chat_text = (
                st.session_state.current_outputs.get("chat_responder") or {}
            ).get("chat_response", "")
            if chat_text:
                st.write_stream(_word_stream(chat_text))
            st.session_state.messages.append(
                {"role": "assistant", "type": "chat", "content": chat_text}
            )
            st.session_state.phase = "input"
            st.rerun()

        elif terminal == "hitl_pause":
            _analysis_summary(st.session_state.current_outputs)
            st.session_state.phase = "hitl"
            st.rerun()

        elif terminal == "pipeline_rejected":
            _analysis_summary(st.session_state.current_outputs)
            me = (
                st.session_state.current_outputs.get("moonshot_evaluator") or {}
            ).get("moonshot_evaluation") or {}
            reason = me.get("gate_failure_reason", "Idea rejected.")
            st.error(f"**Rejected:** {reason}")
            st.session_state.messages.append({
                "role": "assistant", "type": "rejected",
                "outputs": dict(st.session_state.current_outputs),
                "reason": reason,
            })
            st.session_state.phase = "input"
            st.rerun()

        elif terminal == "pipeline_complete":
            # Rare: complete without HITL (e.g. direct path)
            _analysis_summary(st.session_state.current_outputs, stream_kill_shot=True)
            st.session_state.messages.append({
                "role": "assistant", "type": "analysis",
                "outputs": dict(st.session_state.current_outputs),
            })
            st.session_state.phase = "input"
            st.rerun()

# ── HITL phase ─────────────────────────────────────────────────────────────────
elif phase == "hitl":
    # Show analysis from current_outputs (not yet in messages)
    with st.chat_message("assistant"):
        _analysis_summary(st.session_state.current_outputs)

    # The HITL decision arrives as the next assistant message
    with st.chat_message("assistant"):
        st.write(
            "**All 5 agents have completed.** "
            "Approve to design the Kill Shot experiment, or reject to stop here."
        )
        comment = st.text_input(
            "Domain correction or context (optional — passed to Kill Shot agent):",
            key="hitl_comment",
        )
        c1, c2, _ = st.columns([1, 1, 3])
        approved     = c1.button("Approve ✅", type="primary", use_container_width=True)
        rejected_btn = c2.button("Reject ❌",  use_container_width=True)

        if approved or rejected_btn:
            decision = "approved" if approved else "rejected"
            try:
                requests.post(
                    f"{API_URL}/evaluate/{st.session_state.thread_id}/resume",
                    json={"decision": decision, "comment": comment or None},
                    timeout=30,
                ).raise_for_status()
                st.session_state.phase = "resuming"
                st.rerun()
            except Exception as exc:
                st.error(f"Resume failed: {exc}")

# ── Resuming phase ─────────────────────────────────────────────────────────────
elif phase == "resuming":
    # Compact reminder of what was analyzed before the kill shot
    with st.chat_message("assistant"):
        _analysis_summary(st.session_state.current_outputs, stream_kill_shot=False)

    with st.chat_message("assistant"):
        terminal, _ = _stream(
            st.session_state.thread_id, label="Designing Kill Shot experiment…"
        )
        if terminal == "pipeline_complete":
            # Stream the kill shot assumption word by word — the money moment
            _analysis_summary(
                st.session_state.current_outputs, stream_kill_shot=True
            )
            st.session_state.messages.append({
                "role": "assistant", "type": "analysis",
                "outputs": dict(st.session_state.current_outputs),
            })
        else:
            st.error("Pipeline rejected after review.")
            st.session_state.messages.append({
                "role": "assistant", "type": "rejected",
                "outputs": dict(st.session_state.current_outputs),
                "reason": "Rejected after human review.",
            })
        st.session_state.phase = "input"
        st.rerun()

# ── Chat input (always last, only when ready for input) ────────────────────────
if phase == "input":
    if prompt := st.chat_input("Describe your moonshot idea, or just say hi…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            r = requests.post(f"{API_URL}/evaluate", json={"idea": prompt}, timeout=30)
            r.raise_for_status()
            st.session_state.thread_id = r.json()["thread_id"]
            st.session_state.current_outputs = {}
            st.session_state.phase = "streaming"
            st.rerun()
        except Exception as exc:
            st.session_state.messages.pop()
            st.error(f"Backend error: {exc}")
            st.rerun()
