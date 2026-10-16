from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

from numpy.ma import size
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# Project root
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Backend imports
# DO NOT change backend logic
# =========================================================

from src.agent.data_agent import run_agent
from src.cleaning.clean import clean_dataset
from src.rag.retriever import build_rag_index

from html import escape
from collections.abc import Iterable, Mapping


# =========================
# UI THEME
# =========================


INK = "#E6EDF7"
MUTED = "#8FA3BF"
LINE = "#22304A"
CANVAS = "#0B1220"
PAPER = "#121C30"
ACCENT = "#38BDF8"
ACCENT_HOVER = "#7DD3FC"
ACCENT_LIGHT = "#16314D"
RISK = "#F87171"
GOOD = "#34D399"


def inject_theme() -> None:

    st.markdown(
        f"""
        <style>

        :root {{
            --dp-ink:{INK};
            --dp-muted:{MUTED};
            --dp-line:{LINE};
            --dp-canvas:{CANVAS};
            --dp-paper:{PAPER};
            --dp-accent:{ACCENT};
            --dp-risk:{RISK};
            --dp-good:{GOOD};
        }}

        /* =====================================================
           PAGE
        ===================================================== */

        .stApp {{
            background: #0B1220 !important;
            color: #E6EDF7 !important;
        }}

        [data-testid="stAppViewContainer"] {{
            background: #0B1220 !important;
        }}

        [data-testid="stHeader"] {{
            background: #0B1220 !important;
        }}

        [data-testid="stToolbar"] {{
            background: transparent !important;
        }}

        [data-testid="stMain"] {{
            background: #0B1220 !important;
        }}

        [data-testid="stMainBlockContainer"] {{
            color: #E6EDF7 !important;
        }}

        .block-container {{
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }}

        
        html, body, [class*="css"] {{
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
            font-size: 17px !important;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stWidgetLabel"] p,
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div,
        .stMultiSelect div {{
            font-size: 16px !important;
            line-height: 1.6 !important;
        }}

        label, .stCaption {{
            font-size: 15px !important;
        }}

        h1 {{
            font-size: 2.3rem !important;
        }}

        h2 {{
            font-size: 1.7rem !important;
        }}

        h3 {{
            font-size: 1.3rem !important;
        }}

                
        h1, h2, h3 {{
            color: var(--dp-ink);
            letter-spacing: -0.02em;
        }}

        h2 {{
            margin-top: 1.8rem !important;
            margin-bottom: .65rem !important;
        }}

        p, li, label {{
            color:var(--dp-muted);
        }}

        .dp-dataset-intro {{
            color:var(--dp-muted);
            font-size:.88rem;
            line-height:1.5;
            margin:-.2rem 0 .7rem;
        }}

        .dp-dataset-meta {{
            color:var(--dp-muted);
            font-size:.72rem;
            margin-top:.15rem;
        }}

        [data-testid="stFileUploader"] section {{
            padding:.55rem .75rem !important;
        }}

        [data-testid="stFileUploader"] small {{
            color:var(--dp-muted);
        }}


        /* =====================================================
           HERO
        ===================================================== */

        .dp-hero {{
            padding:.35rem 0 1.7rem;
            border-bottom:1px solid var(--dp-line);
            margin-bottom:1.15rem;
        }}

        .dp-kicker {{
            color:var(--dp-accent);
            font-size:.68rem;
            font-weight:700;
            letter-spacing:.11em;
            text-transform:uppercase;
            margin-bottom:.55rem;
        }}

        .dp-section-kicker {{
            color:var(--dp-accent);
            font-size:.68rem;
            font-weight:700;
            letter-spacing:.1em;
            text-transform:uppercase;
            margin:1.5rem 0 .45rem;
        }}

        .dp-hero-title {{
            color:var(--dp-ink);
            font-size:clamp(2.25rem,5vw,3.5rem);
            line-height:1;
            font-weight:700;
            letter-spacing:-.045em;
            margin:0;
        }}

        .dp-hero-subtitle {{
            color:var(--dp-muted);
            font-size:1rem;
            line-height:1.6;
            max-width:720px;
            margin:.75rem 0 0;
        }}

        /* =====================================================
           SECTION LABELS
        ===================================================== */

        .stCaption {{
            color:var(--dp-muted) !important;
        }}

        /* =====================================================
           DATASET META
        ===================================================== */

        .dp-strip {{
            background:transparent;
            border:0;
            border-bottom:1px solid var(--dp-line);
            border-radius:0;
            padding:.15rem 0 .95rem;
            margin:.15rem 0 1.5rem;
        }}

        .dp-strip-grid {{
            display:grid;
            grid-template-columns:2fr repeat(5,1fr);
            gap:1.2rem;
            width:100%;
            align-items:end;
        }}

        .dp-strip-grid > div {{
            min-width:0;
        }}

        .dp-strip-label {{
            color:var(--dp-muted);
            font-size:.66rem;
            font-weight:600;
            text-transform:uppercase;
            letter-spacing:.07em;
            margin-bottom:.2rem;
        }}

        .dp-strip-value {{
            color:var(--dp-ink);
            font-size:.9rem;
            font-weight:600;
            overflow-wrap:anywhere;
        }}

        /* =====================================================
           STREAMLIT INPUTS
        ===================================================== */

        [data-testid="stFileUploader"] {{
    background: #121C30 !important;
    border: 1px solid #22304A !important;
    border-radius: 10px !important;
    padding: 8px !important;
}}

[data-testid="stFileUploader"] section {{
    background: #121C30 !important;
    border: 1px dashed #22304A !important;
    border-radius: 8px !important;
}}

[data-testid="stFileUploader"] section > div {{
    background: #121C30 !important;
}}

[data-testid="stFileUploader"] button {{
    background: #16314D !important;
    color: #7DD3FC !important;
    border: 1px solid #22304A !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
}}

[data-testid="stFileUploader"] button:hover {{
    background: #16314D !important;
    color: #06202E !important;
    border-color: #38BDF8 !important;
}}

[data-testid="stFileUploader"] small {{
    color: #8FA3BF !important;
}}

[data-testid="stFileUploader"] label {{
    color: #E6EDF7 !important;
}}

        [data-testid="stTextArea"] textarea {{
            background:var(--dp-paper);
            border:1px solid #22304A;
            border-radius:10px;
            color:var(--dp-ink);
            font-size:.95rem;
            line-height:1.55;
        }}

        [data-testid="stTextArea"] textarea:focus {{
            border-color:var(--dp-accent);
            box-shadow:none;
        }}

        [data-testid="stTextArea"] label {{
            color:var(--dp-muted);
        }}

        # /* Objective pills */
        # [data-testid="stPills"] {{
        #     margin:.35rem 0 .75rem;
        # }}

        # [data-testid="stPills"] button {{
        #     border:1px solid #22304A;
        #     background:var(--dp-paper);
        #     color:var(--dp-muted);
        #     border-radius:999px;
        #     font-size:.82rem;
        #     font-weight:600;
        #     padding:.35rem .8rem;
        # }}

        # [data-testid="stPills"] button:hover {{
        #     border-color:var(--dp-accent);
        #     color:var(--dp-accent);
        # }}

        # [data-testid="stPills"] button[aria-checked="true"] {{
        #     background:var(--dp-accent);
        #     border-color:var(--dp-accent);
        #     color:white;
        # }}

        /* =====================================================
           PRIMARY BUTTON
        ===================================================== */
        .stButton > button {{
            background: #38BDF8 !important;
            color: #121C30 !important;
            border: 1px solid #38BDF8 !important;
            border-radius: 8px !important;
            min-height: 44px !important;
            padding: 0.5rem 1.25rem !important;

            font-family: "IBM Plex Sans", sans-serif !important;
            font-size: 14px !important;
            font-weight: 600 !important;

            box-shadow: none !important;
            opacity: 1 !important;
        }}

        .stButton > button:hover {{
            background: #7DD3FC !important;
            color: #121C30 !important;
            border-color: #7DD3FC !important;
        }}

        .stButton > button:active {{
            background: #06202E !important;
            color: #121C30 !important;
        }}

        .stButton > button:focus {{
            background: #38BDF8 !important;
            color: #121C30 !important;
            border-color: #38BDF8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.22) !important;
        }}

        /* =====================================================
           STEPPER
        ===================================================== */

        .dp-stepper {{
            display:grid;
            grid-template-columns:repeat(5,minmax(0,1fr));
            gap:0;
            margin:1rem 0 1.7rem;
            border-top:1px solid var(--dp-line);
        }}

        .dp-step {{
            border-top:2px solid transparent;
            padding:.65rem .45rem 0 0;
            margin-top:-2px;
        }}

        .dp-step.done {{
            border-top-color:var(--dp-accent);
        }}

        .dp-step-number {{
            color:var(--dp-muted);
            font-size:.62rem;
            font-weight:700;
            letter-spacing:.08em;
        }}

        .dp-step.done .dp-step-number {{
            color:var(--dp-accent);
        }}

        .dp-step-name {{
            color:var(--dp-ink);
            font-size:.78rem;
            font-weight:650;
            margin-top:.12rem;
        }}

        /* =====================================================
           KPI STRIP
        ===================================================== */

        .dp-kpi-strip {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:0;
            margin:.5rem 0 1.35rem;
            background:var(--dp-paper);
            border:1px solid var(--dp-line);
            border-radius:11px;
            overflow:hidden;
        }}

        .dp-kpi {{
            background:var(--dp-paper);
            border:0;
            border-right:1px solid var(--dp-line);
            border-radius:0;
            padding:1rem 1.05rem;
            min-height:78px;
        }}

        .dp-kpi:last-child {{
            border-right:0;
        }}

        .dp-kpi-label {{
            color:var(--dp-muted);
            font-size:.68rem;
            font-weight:600;
            text-transform:uppercase;
            letter-spacing:.05em;
            margin-bottom:.35rem;
        }}

        .dp-kpi-value {{
            color:var(--dp-ink);
            font-size:1.08rem;
            font-weight:700;
            line-height:1.2;
        }}

        .dp-kpi-note {{
            color:var(--dp-muted);
            font-size:.68rem;
            margin-top:.28rem;
        }}

        .dp-good {{
            color:var(--dp-good);
        }}

        .dp-risk {{
            color:var(--dp-risk);
        }}

        /* =====================================================
           ANSWER
        ===================================================== */

        .st-key-answer {{
            background:var(--dp-paper);
            border:1px solid var(--dp-line);
            border-left:3px solid var(--dp-accent);
            border-radius:11px;
            padding:1.35rem 1.5rem;
            margin:0 0 .4rem;
        }}

        .st-key-answer [data-testid="stMarkdownContainer"] {{
            color:var(--dp-ink);
            font-family:"Source Serif 4",Georgia,serif;
            font-size:1.04rem;
            line-height:1.72;
        }}

        .st-key-answer h1,
        .st-key-answer h2,
        .st-key-answer h3 {{
            font-family:"IBM Plex Sans","Segoe UI",sans-serif;
            color:var(--dp-ink);
        }}

        .st-key-answer p,
        .st-key-answer li {{
            color:var(--dp-ink);
        }}

        .dp-note {{
            color:var(--dp-muted);
            font-size:.72rem;
            margin:0 0 1.25rem;
        }}

        /* =====================================================
           EXPANDERS
        ===================================================== */

        [data-testid="stExpander"] {{
            background:var(--dp-paper);
            border:1px solid var(--dp-line);
            border-radius:9px;
            margin-bottom:.55rem;
        }}

        [data-testid="stExpander"] summary {{
            color:var(--dp-ink);
            font-weight:650;
        }}

        [data-testid="stExpander"] summary:hover {{
            color:var(--dp-accent);
        }}

        /* =====================================================
           TOOL TRACE
        ===================================================== */

        .dp-tool-trace {{
            display:flex;
            flex-wrap:wrap;
            gap:.45rem;
            margin:.25rem 0 .3rem;
        }}

        .dp-tool {{
            border:1px solid var(--dp-line);
            border-radius:999px;
            padding:.38rem .7rem;
            background:var(--dp-paper);
            color:var(--dp-muted);
            font-size:.74rem;
        }}

        .dp-tool.used {{
            border-color:#38BDF8;
            color:var(--dp-accent);
            font-weight:650;
        }}

        /* =====================================================
           DATAFRAMES
        ===================================================== */

        [data-testid="stDataFrame"] {{
            border:1px solid var(--dp-line);
            border-radius:9px;
            overflow:hidden;
        }}

        /* =====================================================
           CHART AREA
        ===================================================== */

        .stPlotlyChart {{
            background:var(--dp-paper);
            border:1px solid var(--dp-line);
            border-radius:10px;
            padding:.25rem;
        }}

        /* =====================================================
           MISC
        ===================================================== */

        [data-testid="stStatusWidget"] {{
            border:1px solid var(--dp-line);
            border-radius:9px;
        }}

        hr {{
            border-color:var(--dp-line);
        }}

        @media (max-width:760px) {{

            .block-container {{
                padding-top:1.4rem;
            }}

            .dp-strip-grid {{
                grid-template-columns:repeat(2,minmax(0,1fr));
            }}

            .dp-kpi-strip {{
                grid-template-columns:repeat(2,minmax(0,1fr));
            }}

            .dp-kpi {{
                border-right:0;
                border-bottom:1px solid var(--dp-line);
            }}

            .dp-kpi:nth-child(odd) {{
                border-right:1px solid var(--dp-line);
            }}

            .dp-tool-trace {{
                grid-template-columns:1fr;
            }}

            .dp-stepper {{
                grid-template-columns:1fr;
                gap:.55rem;
                border-top:0;
            }}

            .dp-step {{
                border-top:1px solid var(--dp-line);
                padding-top:.45rem;
                margin-top:0;
            }}

            .dp-step.done {{
                border-top:2px solid var(--dp-accent);
            }}
        }}

                .plan-step {{
            display: flex;
            align-items: flex-start;
            gap: 14px;
            padding: 12px 0;
            border-bottom: 1px solid #22304A;
        }}

        .plan-step:last-child {{
            border-bottom: none;
        }}

        .plan-number {{
            min-width: 34px;
            font-size: 12px;
                        font-weight: 600;
            color: #38BDF8;
            letter-spacing: 0.04em;
        }}

        .plan-content {{
            flex: 1;
        }}

        .plan-label {{
            font-size: 14px;
            line-height: 1.55;
            color: #E6EDF7;
        }}

    

        div[data-testid="stPills"] button {{
            background: #121C30 !important;
            color: #8FA3BF !important;
            border: 1px solid #22304A !important;
            border-radius: 999px !important;
            padding: 0.45rem 0.9rem !important;
            font-family: "IBM Plex Sans", sans-serif !important;
            font-weight: 500 !important;
            opacity: 1 !important;
        }}

        div[data-testid="stPills"] button:hover {{
            background: #16314D !important;
            color: #7DD3FC !important;
            border-color: #38BDF8 !important;
        }}

        div[data-testid="stPills"] button[aria-pressed="true"] {{
            background: #38BDF8 !important;
            color: #121C30 !important;
            border: 1px solid #38BDF8 !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }}

                
                
                
        /* Run analysis button */
        div.stButton > button[kind="primary"] {{
            background-color: #38BDF8 !important;
            color: #06202E !important;
            border: 1px solid #38BDF8 !important;
            border-radius: 10px !important;
            min-height: 54px !important;
            padding: 0.7rem 1rem !important;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            opacity: 1 !important;
            box-shadow: none !important;
            transition: background-color 0.2s ease !important;
        }}

        div.stButton > button[kind="primary"] p {{
            color: #06202E !important;
            font-size: 18px !important;
            font-weight: 700 !important;
        }}

        div.stButton > button[kind="primary"]:hover {{
            background-color: #7DD3FC !important;
            color: #06202E !important;
            border-color: #7DD3FC !important;
        }}

        div.stButton > button[kind="primary"]:disabled {{
            background-color: #38BDF8 !important;
            color: #06202E !important;
            opacity: 1 !important;
        }}

                
        /* Dashboard typography */
        .dp-hero-title {{
            font-size: 2.5rem !important;
            line-height: 1.2 !important;
        }}

        .dp-hero-subtitle {{
            font-size: 1.15rem !important;
            line-height: 1.6 !important;
        }}

        .dp-kicker {{
            font-size: 0.95rem !important;
        }}

        .dp-kpi-label,
        .dp-strip-label {{
            font-size: 0.95rem !important;
        }}

        .dp-kpi-value {{
            font-size: 1.8rem !important;
            line-height: 1.3 !important;
        }}

        .dp-kpi-note,
        .dp-strip-value {{
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }}

        .dp-step-name {{
            font-size: 1rem !important;
        }}

        .dp-step-number {{
            font-size: 0.85rem !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state["_datapilot_theme_loaded"] = True

def hero(
    title: str = "DataPilot AI",
    subtitle: str = "Explainable, task-aware decision support for retail sales and inventory.",
) -> None:
    inject_theme()

    st.markdown(
        f"""
        <section class="dp-hero">
            <div class="dp-kicker">Retail decision support</div>
            <div class="dp-hero-title">{escape(title)}</div>
            <p class="dp-hero-subtitle">{escape(subtitle)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def dataset_strip(
    filename: str,
    rows: int,
    columns: int,
    date_range: str = "—",
    missing: int = 0,
    duplicates: int = 0,
) -> None:
    inject_theme()

    cells = [
        f'<div><div class="dp-strip-label">Dataset</div><div class="dp-strip-value">{escape(str(filename))}</div></div>',
        f'<div><div class="dp-strip-label">Rows</div><div class="dp-strip-value">{rows:,}</div></div>',
        f'<div><div class="dp-strip-label">Columns</div><div class="dp-strip-value">{columns}</div></div>',
        f'<div><div class="dp-strip-label">Missing</div><div class="dp-strip-value">{missing:,}</div></div>',
        f'<div><div class="dp-strip-label">Duplicates</div><div class="dp-strip-value">{duplicates:,}</div></div>',
        f'<div><div class="dp-strip-label">Date range</div><div class="dp-strip-value">{escape(str(date_range))}</div></div>',
    ]

    html = (
        '<section class="dp-strip">'
        '<div class="dp-strip-grid">'
        + "".join(cells)
        + "</div></section>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )

def kpi_cards(items: Iterable[Mapping[str, object]]) -> None:
    inject_theme()

    cards = []

    for item in items:
        tone = item.get("tone", "neutral")

        tone_class = (
            "dp-good"
            if tone == "good"
            else "dp-risk"
            if tone == "risk"
            else ""
        )

        label = escape(
            str(item.get("label", ""))
        )

        value = escape(
            str(item.get("value", "—"))
        )

        note = item.get("note")

        note_html = (
            '<div class="dp-kpi-note">'
            + escape(str(note))
            + "</div>"
            if note
            else ""
        )

        cards.append(
            '<div class="dp-kpi">'
            f'<div class="dp-kpi-label">{label}</div>'
            f'<div class="dp-kpi-value {tone_class}">{value}</div>'
            f'{note_html}'
            '</div>'
        )

    if cards:
        html = (
            '<div class="dp-kpi-strip">'
            + "".join(cards)
            + "</div>"
        )

        st.markdown(
            html,
            unsafe_allow_html=True,
        )


def stepper(
    steps: int = 5,
    current: int | None = None,
) -> None:
    inject_theme()

    labels = [
        "Profile",
        "Plan",
        "Analyze",
        "Retrieve",
        "Answer",
    ][:steps]

    done_to = (
        steps
        if current is None
        else max(0, min(current, steps))
    )

    blocks = []

    for index, label in enumerate(labels, start=1):
        done_class = " done" if index <= done_to else ""

        blocks.append(
            f'<div class="dp-step{done_class}">'
            f'<div class="dp-step-number">STEP {index}</div>'
            f'<div class="dp-step-name">{escape(label)}</div>'
            f'</div>'
        )

    html = (
        '<div class="dp-stepper">'
        + "".join(blocks)
        + "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )

def _tool_names(all_tools: object) -> list[str]:
    if isinstance(all_tools, Mapping):
        return [str(name) for name in all_tools.keys()]

    if isinstance(all_tools, (list, tuple, set)):
        names = []

        for item in all_tools:
            if isinstance(item, str):
                names.append(item)

            elif isinstance(item, Mapping):
                name = (
                    item.get("name")
                    or item.get("tool_name")
                    or item.get("id")
                )

                if name:
                    names.append(str(name))

            else:
                name = (
                    getattr(item, "name", None)
                    or getattr(item, "tool_name", None)
                )

                if name:
                    names.append(str(name))

        return names

    return []


def tool_trace(
    tools_used: Iterable[str],
    all_tools: object,
) -> None:
    inject_theme()

    used = {
        str(item)
        for item in (tools_used or [])
    }

    names = _tool_names(all_tools) or sorted(used)

    blocks = []

    for name in names:
        used_class = " used" if name in used else ""
        status_text = "Used" if name in used else "Not used"

        blocks.append(
            f'<div class="dp-tool{used_class}">'
            f'{escape(name)} · {status_text}'
            f'</div>'
        )

    if blocks:
        html = (
            '<div class="dp-tool-trace">'
            + "".join(blocks)
            + "</div>"
        )

        st.markdown(
            html,
            unsafe_allow_html=True,
        )
# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="DataPilot AI",
    page_icon=None,
    layout="wide",
)

# =========================================================
# Theme
# =========================================================

inject_theme()
hero()


# =========================================================
# Constants
# =========================================================

REQUIRED_COLUMNS = [
    "Date",
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Weather Condition",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Seasonality",
]


OBJECTIVES = {
    "Sales performance": (
        "Analyze sales performance and identify the most important "
        "sales pattern."
    ),
    "Inventory risk": (
        "Identify store-product combinations at high stock-out risk "
        "and explain the risk using inventory coverage."
    ),
    "Demand forecast": (
        "Forecast demand for the requested store and product and "
        "report the forecast."
    ),
    "Data quality": (
        "Assess data quality, including missing values, duplicates, "
        "invalid values, and key cleaning actions."
    ),
    "Business health": (
        "Give a concise business health summary using current sales, "
        "sales trend, category performance, and high-risk inventory."
    ),
}


# =========================================================
# Session state
# =========================================================

if "objective_choice" not in st.session_state:
    st.session_state.objective_choice = None

if "objective_text" not in st.session_state:
    st.session_state.objective_text = ""

if "last_objective_choice" not in st.session_state:
    st.session_state.last_objective_choice = None


if "agent_result" not in st.session_state:
    st.session_state.agent_result = None

if "last_run_objective" not in st.session_state:
    st.session_state.last_run_objective = None

if "dataset_signature" not in st.session_state:
    st.session_state.dataset_signature = None




# =========================================================
# Utility helpers
# =========================================================

def safe_read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read CSV/XLSX without exposing raw exceptions."""

    try:
        suffix = Path(
            uploaded_file.name
        ).suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(uploaded_file)

        if suffix == ".xlsx":
            return pd.read_excel(uploaded_file)

        st.error(
            "Use a CSV or Excel (.xlsx) file."
        )

    except Exception:
        st.error(
            "The dataset could not be read. "
            "Check the file format and try again."
        )

    return pd.DataFrame()


def dataset_date_range(
    df: pd.DataFrame,
) -> str:
    """Return a readable date range."""

    if "Date" not in df.columns:
        return "—"

    dates = pd.to_datetime(
        df["Date"],
        errors="coerce",
    ).dropna()

    if dates.empty:
        return "—"

    return (
        f"{dates.min():%d %b %Y} "
        f"to "
        f"{dates.max():%d %b %Y}"
    )


def friendly_tool_name(
    name: str,
) -> str:
    """Convert internal tool names into readable labels."""

    cleaned = (
        str(name)
        .replace("get_", "")
        .replace("search_", "")
        .replace("_", " ")
    )

    return cleaned.title()


def normalize_tool_results(
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert the existing Agent tool_results list into:

        {
            "tool_name": result
        }

    without changing backend behavior.
    """

    raw = (
        agent_result.get("tool_results")
        or agent_result.get("results")
        or agent_result.get("tool_evidence")
        or {}
    )

    if isinstance(raw, dict):

        if any(
            str(key).startswith("get_")
            or str(key).startswith("search_")
            for key in raw
        ):
            return raw

        if (
            "tool" in raw
            and (
                "result" in raw
                or "output" in raw
            )
        ):
            return {
                str(raw["tool"]):
                    raw.get(
                        "result",
                        raw.get("output")
                    )
            }

    normalized: dict[str, Any] = {}

    if isinstance(raw, list):

        for item in raw:

            if not isinstance(item, dict):
                continue

            name = (
                item.get("tool_name")
                or item.get("tool")
                or item.get("name")
            )

            if not name:
                continue

            if "result" in item:
                value = item["result"]
            elif "output" in item:
                value = item["output"]
            else:
                value = item.get("data")

            normalized[str(name)] = value

    return normalized


def _number(
    value: Any,
) -> float | None:
    """Safely convert a value to float."""

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)

    if isinstance(value, str):

        try:
            cleaned = (
                value
                .replace(",", "")
                .replace("₹", "")
                .replace("%", "")
                .strip()
            )

            return float(cleaned)

        except (TypeError, ValueError):
            return None

    return None


def _find_value(
    obj: Any,
    keys: tuple[str, ...],
) -> Any:
    """Recursively find the first matching key."""

    if isinstance(obj, dict):

        for key in keys:

            if key in obj:
                return obj[key]

        for value in obj.values():

            found = _find_value(
                value,
                keys,
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:

            found = _find_value(
                value,
                keys,
            )

            if found is not None:
                return found

    return None


def _records(
    obj: Any,
) -> list[dict[str, Any]]:
    """Normalize list/dict records for chart rendering."""

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(
            orient="records"
        )

    if isinstance(obj, list):
        return [
            item
            for item in obj
            if isinstance(item, dict)
        ]

    if isinstance(obj, dict):

        for key in (
            "rows",
            "records",
            "data",
            "results",
            "items",
            "high_risk_inventory",
            "category_sales",
            "sales_by_category",
            "forecast",
        ):

            value = obj.get(key)

            if isinstance(value, pd.DataFrame):

                return value.to_dict(
                    orient="records"
                )

            if (
                isinstance(value, list)
                and all(
                    isinstance(item, dict)
                    for item in value
                )
            ):
                return value

    return []


def _category_pairs(
    obj: Any,
) -> list[tuple[str, float]]:
    """Extract category-sales pairs from the tool result."""

    # Actual backend output:
    # pandas Series with category names as index
    if isinstance(obj, pd.Series):
        pairs = []

        for label, metric in obj.items():
            number = _number(metric)

            if number is not None:
                pairs.append(
                    (
                        str(label),
                        number,
                    )
                )

        return pairs

    # DataFrame output
    if isinstance(obj, pd.DataFrame):
        pairs = []

        for row in obj.to_dict(orient="records"):
            label = (
                row.get("Category")
                or row.get("category")
                or row.get("category_name")
            )

            value = _number(
                row.get("sales")
                or row.get("total_sales")
                or row.get("value")
            )

            if label is not None and value is not None:
                pairs.append(
                    (
                        str(label),
                        value,
                    )
                )

        return pairs

    # Dictionary output
    if isinstance(obj, dict):

        for key in (
            "category_sales",
            "sales_by_category",
            "categories",
        ):
            value = obj.get(key)

            if isinstance(value, dict):

                pairs = []

                for label, metric in value.items():
                    number = _number(metric)

                    if number is not None:
                        pairs.append(
                            (
                                str(label),
                                number,
                            )
                        )

                if pairs:
                    return pairs

        # Wrapped result
        for key in (
            "result",
            "output",
            "data",
            "value",
            "results",
        ):
            nested = obj.get(key)

            if nested is not None:
                pairs = _category_pairs(nested)

                if pairs:
                    return pairs

        # Direct dictionary
        pairs = []

        for label, metric in obj.items():
            number = _number(metric)

            if number is not None:
                pairs.append(
                    (
                        str(label),
                        number,
                    )
                )

        if pairs:
            return pairs

    # List of records
    if isinstance(obj, list):

        pairs = []

        for item in obj:

            if not isinstance(item, dict):
                continue

            label = (
                item.get("Category")
                or item.get("category")
                or item.get("category_name")
                or item.get("Region")
                or item.get("region")
                or item.get("region_name")
            )

            value = _number(
                item.get("Sales")
                or item.get("sales")
                or item.get("total_sales")
                or item.get("value")
            )

            if label is not None and value is not None:
                pairs.append(
                    (
                        str(label),
                        value,
                    )
                )

        return pairs
    return []
# =========================================================
# Dataset section
# =========================================================

st.subheader("Dataset")

st.markdown(
    '<div class="dp-dataset-intro">'
    "Upload the retail dataset used for analysis. "
    "Data quality checks and cleaning are applied automatically."
    "</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a retail CSV or Excel file",
    type=["csv", "xlsx"],
    help="CSV and XLSX files are supported.",
)


if uploaded_file is None:

    st.info(
        "Upload a retail CSV or Excel file to start."
    )

    st.stop()


raw_df = safe_read_uploaded_file(
    uploaded_file
)

if raw_df.empty:
    st.stop()


# =========================================================
# Required-column validation
# =========================================================

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in raw_df.columns
]

if missing_columns:

    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
        + "."
    )

    st.stop()


# =========================================================
# Cleaning
# =========================================================

try:

    cleaned_df, cleaning_log = (
        clean_dataset(raw_df)
    )

except Exception:

    st.error(
        "The dataset could not be cleaned. "
        "Check the required columns and data types."
    )

    st.stop()


# =========================================================
# Active dataset
# =========================================================

processed_dir = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

processed_dir.mkdir(
    parents=True,
    exist_ok=True,
)

active_dataset_path = (
    processed_dir
    / "active_retail_data.csv"
)

cleaned_df.to_csv(
    active_dataset_path,
    index=False,
)

os.environ[
    "DATAPILOT_DATASET_PATH"
] = str(
    active_dataset_path
)

dataset_signature = hashlib.sha256(
    uploaded_file.getvalue()
).hexdigest()

if (
    st.session_state.dataset_signature
    != dataset_signature
):

    try:

        build_rag_index()

        st.session_state.dataset_signature = (
            dataset_signature
        )

    except Exception:

        st.warning(
            "Dataset knowledge could not be indexed. "
            "Analysis tools are still available."
        )

# =========================================================
# Dataset strip
# =========================================================

missing_values = int(
    cleaned_df.isna().sum().sum()
)

duplicate_rows = int(
    cleaned_df.duplicated().sum()
)

dataset_strip(
    filename=uploaded_file.name,
    rows=len(cleaned_df),
    columns=len(cleaned_df.columns),
    date_range=dataset_date_range(
        cleaned_df
    ),
    missing=missing_values,
    duplicates=duplicate_rows,
)


# =========================================================
# Dataset detail expanders
# =========================================================

with st.expander(
    "Cleaning actions",
    expanded=False,
):

    if isinstance(
        cleaning_log,
        dict,
    ):

        for key, value in cleaning_log.items():

            st.write(
                f"**{key}:** {value}"
            )

    elif isinstance(
        cleaning_log,
        (list, tuple),
    ):

        for item in cleaning_log:

            st.write(
                str(item)
            )

    else:

        st.write(
            "No cleaning actions were returned."
        )


with st.expander(
    "Dataset profile",
    expanded=False,
):

    profile_summary = pd.DataFrame(
        [
            {
                "Metric": "Rows",
                "Value": str(len(cleaned_df)),
            },
            {
                "Metric": "Columns",
                "Value": str(len(
                    cleaned_df.columns
                )),
            },
            {
                "Metric": "Missing values",
                "Value": str(missing_values),
            },
            {
                "Metric": "Duplicate rows",
                "Value": str(duplicate_rows),
            },
            {
                "Metric": "Date range",
                "Value": dataset_date_range(
                    cleaned_df
                ),
            },
        ]
    )

    st.dataframe(
        profile_summary,
        width="stretch",
        hide_index=True,
    )

    profile_df = pd.DataFrame(
        {
            "Column": cleaned_df.columns,
            "Type": [
                str(dtype)
                for dtype in cleaned_df.dtypes
            ],
        }
    )

    st.dataframe(
        profile_df,
        width="stretch",
        hide_index=True,
    )


with st.expander(
    "Preview",
    expanded=False,
):

    st.dataframe(
        cleaned_df.head(10),
        width="stretch",
        hide_index=True,
    )


# =========================================================
# Objective
# =========================================================

st.subheader("Objective")

st.caption(
    "Choose a starting point, then edit the objective if needed."
)

selected_choice = st.pills(
    "Choose an objective",
    options=list(OBJECTIVES.keys()),
    key="objective_choice",
    label_visibility="collapsed",
)

if selected_choice is not None:
    if selected_choice != st.session_state.last_objective_choice:
        st.session_state.objective_text = OBJECTIVES[
            selected_choice
        ]
        st.session_state.last_objective_choice = selected_choice

objective_col, action_col = st.columns(
    [5, 1],
    gap="medium",
)

with objective_col:
    st.text_area(
        "Business objective",
        key="objective_text",
        height=95,
        placeholder="Describe what the manager needs to know.",
    )

with action_col:
    st.markdown(
        "<div style='height:1.85rem'></div>",
        unsafe_allow_html=True,
    )

    run_clicked = st.button(
        "Run analysis",
        type="primary",
        width="stretch",
    )


# =========================================================
# Agent execution
# =========================================================

if run_clicked:

    objective = st.session_state.objective_text.strip()

    if not objective:
        selected_choice = st.session_state.get("objective_choice")

        if selected_choice in OBJECTIVES:
            objective = OBJECTIVES[selected_choice]
            st.session_state.objective_text = objective

    if not objective:

        st.warning(
            "Enter a business objective before running the analysis."
        )

        st.stop()

    with st.status(
        "Running analysis…",
        expanded=False,
    ) as analysis_status:

        try:

            st.session_state.agent_result = (
                run_agent(objective)
            )

            st.session_state.last_run_objective = (
                objective
            )

            analysis_status.update(
                label="Analysis complete",
                state="complete",
            )

        except Exception:

            analysis_status.update(
                label="Analysis could not be completed",
                state="error",
            )

            st.error(
                "The analysis could not be completed. "
                "Add a Gemini API key to .env, then run again."
            )

            st.stop()


# =========================================================
# Result visibility
# Only show the stored result for the objective that was run.
# Editing the text area does not call the agent.
# =========================================================

agent_result = st.session_state.get(
    "agent_result"
)

current_objective = (
    st.session_state.objective_text.strip()
)

last_run_objective = (
    st.session_state.last_run_objective
)


if (
    not agent_result
    or not last_run_objective
    or current_objective != last_run_objective
):

    st.stop()


# =========================================================
# Normalize Agent result
# =========================================================

tool_results = normalize_tool_results(
    agent_result
)

tool_calls = (
    agent_result.get(
        "tool_calls"
    )
    or []
)

tools_used = []

for tool_call in tool_calls:

    if not isinstance(
        tool_call,
        dict,
    ):
        continue

    tool_name = (
        tool_call.get("tool")
        or tool_call.get("name")
    )

    if (
        tool_name
        and tool_name not in tools_used
    ):
        tools_used.append(
            str(tool_name)
        )


task_plan = (
    agent_result.get("task_plan")
    or agent_result.get("plan")
)

final_answer = (
    agent_result.get("answer")
    or agent_result.get("final_answer")
    or agent_result.get("response")
    or "No answer was returned."
)

retrieved_sources = (
    agent_result.get(
        "retrieved_sources"
    )
    or []
)


# =========================================================
# Results
# =========================================================

stepper(
    5,
    current=5,
)


# =========================================================
# KPI cards
# =========================================================

kpi_items = []


# ---------------------------------------------------------
# Total sales
# ---------------------------------------------------------

if "get_total_sales" in tool_results:

    total_sales = _number(
        _find_value(
            tool_results[
                "get_total_sales"
            ],
            (
                "total_sales",
                "sales",
                "total",
            ),
        )
    )

    if total_sales is not None:

        kpi_items.append(
            {
                "label": "Total sales",
                "value": (
                    f"₹{total_sales:,.2f}"
                ),
            }
        )


# ---------------------------------------------------------
# Sales trend
# ---------------------------------------------------------

if "get_sales_trend" in tool_results:

    trend_value = _number(
        _find_value(
            tool_results[
                "get_sales_trend"
            ],
            (
                "percentage_change",
                "trend_pct",
                "trend_percent",
                "sales_trend_pct",
                "change_pct",
            ),
        )
    )

    if trend_value is not None:

        kpi_items.append(
            {
                "label": "7-day sales trend",
                "value": (
                    f"{trend_value:+.2f}%"
                ),
                "tone": (
                    "good"
                    if trend_value >= 0
                    else "risk"
                ),
            }
        )


# ---------------------------------------------------------
# Top category
# ---------------------------------------------------------

if "get_sales_by_category" in tool_results:

    category_pairs = _category_pairs(
        tool_results[
            "get_sales_by_category"
        ]
    )

    if category_pairs:

        category_pairs.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        kpi_items.append(
            {
                "label": "Top category",
                "value": category_pairs[0][0],
            }
        )


# ---------------------------------------------------------
# High stock-out risk
# ---------------------------------------------------------

if (
    "get_high_risk_inventory"
    in tool_results
):

    risk_data = tool_results[
        "get_high_risk_inventory"
    ]

    risk_count = _number(
        _find_value(
            risk_data,
            (
                "total_high_risk_combinations",
                "high_risk_count",
                "risk_count",
                "total_high_risk",
            ),
        )
    )

    if risk_count is not None:

        kpi_items.append(
            {
                "label": "High stock-out risk",
                "value": f"{int(risk_count):,}",
                "tone": "risk",
                "note": "Coverage under 3 days",
            }
        )


kpi_cards(
    kpi_items
)


# =========================================================
# Decision summary
# =========================================================

st.markdown(
    '<div class="dp-section-kicker">Decision summary</div>',
    unsafe_allow_html=True,
)

answer_container = st.container(
    key="answer"
)

with answer_container:
    st.markdown(final_answer)

st.markdown(
    '<div class="dp-note">'
    "Numbers come from Python tools run on your dataset. "
    "The AI explains the results; it does not invent the numbers."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# How the agent got there
# =========================================================

with st.expander(
    "How the agent got there",
    expanded=False,
):
    if tools_used:
        st.caption(
            "Tools selected by the agent to support this answer."
        )

        tool_trace(
            tools_used,
            tools_used,
        )

    else:
        st.write(
            "No tool calls were recorded."
        )


# =========================================================
# AI task plan
# =========================================================

def render_plan_steps(
    plan: Any,
) -> list[str]:

    if not plan:
        return []

    text = str(plan).replace(
        "\r",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    matches = re.findall(
        r"(?is)"
        r"Step\s+\d+\s*[:.)-]\s*"
        r"(.*?)"
        r"(?=\s+Step\s+\d+\s*[:.)-]\s*|$)",
        text,
    )

    if matches:
        return [
            item.strip()
            for item in matches
            if item.strip()
        ]

    lines = [
        line.strip()
        for line in str(plan).splitlines()
        if line.strip()
    ]

    return lines


def clean_plan_label(
    text: str,
) -> str:
    """Convert raw agent plan steps into readable business labels."""

    cleaned = str(text).strip()

    # Remove numbering such as:
    # 1.
    # 2)
    # Step 1:
    cleaned = re.sub(
        r"^\s*(?:step\s*)?\d+\s*[\.\):\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    tool_match = re.search(
        r"(get_[a-zA-Z0-9_]+)",
        cleaned,
    )

    if tool_match:
        tool_name = tool_match.group(1)

        labels = {
            "get_total_sales": "Calculate total sales",
            "get_sales_by_category": "Compare sales by category",
            "get_sales_by_region": "Compare sales across regions",
            "get_sales_trend": "Measure recent sales trend",
            "get_high_risk_inventory": "Identify high stock-out risk",
            "get_demand_forecast": "Forecast demand",
        }

        return labels.get(
            tool_name,
            friendly_tool_name(tool_name),
        )

    # Remove leftover tool syntax if there is any.
    cleaned = re.sub(
        r"`?[A-Za-z_]+\([^)]*\)`?",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip(" -:")

    return cleaned


with st.expander(
    "AI task plan",
    expanded=False,
):
    st.caption(
        "The sequence of analysis steps selected for this objective."
    )

    plan_steps = render_plan_steps(
        task_plan
    )

    if plan_steps:

        for index, step in enumerate(
            plan_steps,
            start=1,
        ):

            label = clean_plan_label(
                step
            )

            st.markdown(
                f"""
                <div class="plan-step">
                    <div class="plan-number">
                        {index:02d}
                    </div>
                    <div class="plan-content">
                        <div class="plan-label">
                            {escape(label)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.write(
            "No task plan was returned."
        )

    with st.expander(
        "View plan details",
        expanded=False,
    ):

        if task_plan:

            st.code(
                str(task_plan),
                language="text",
            )

        else:

            st.write(
                "No plan details were returned."
            )


# =========================================================
# Tool evidence
# =========================================================

def render_evidence_value(
    result: Any,
) -> None:

    # DataFrame
    if isinstance(result, pd.DataFrame):

        st.dataframe(
            result,
            width="stretch",
            hide_index=True,
        )

        return

    # Series
    if isinstance(result, pd.Series):

        evidence_df = (
            result
            .rename("Value")
            .reset_index()
        )

        st.dataframe(
            evidence_df,
            width="stretch",
            hide_index=True,
        )

        return

    # List of dictionaries
    if isinstance(result, list):

        if result and all(
            isinstance(item, dict)
            for item in result
        ):

            evidence_df = pd.DataFrame(
                result
            )

            st.dataframe(
                evidence_df,
                width="stretch",
                hide_index=True,
            )

            return

        st.write(result)
        return

    # Dictionary
    if isinstance(result, dict):

        # Find a useful tabular value inside
        for value in result.values():

            if isinstance(value, pd.DataFrame):

                st.dataframe(
                    value,
                    width="stretch",
                    hide_index=True,
                )

                return

            if isinstance(value, pd.Series):

                evidence_df = (
                    value
                    .rename("Value")
                    .reset_index()
                )

                st.dataframe(
                    evidence_df,
                    width="stretch",
                    hide_index=True,
                )

                return

            if isinstance(value, list) and value:

                if all(
                    isinstance(item, dict)
                    for item in value
                ):

                    st.dataframe(
                        pd.DataFrame(value),
                        width="stretch",
                        hide_index=True,
                    )

                    return

        # Fallback for small dictionaries
        st.json(result)
        return

    # Simple scalar result
    st.metric(
        label="Result",
        value=str(result),
    )


with st.expander(
    "Tool evidence",
    expanded=False,
):

    if tool_results:

        for name, result in tool_results.items():

            with st.expander(
                friendly_tool_name(name),
                expanded=False,
            ):

                render_evidence_value(
                    result
                )

    else:

        st.write(
            "No tool evidence was returned."
        )

# =========================================================
# Retrieved dataset knowledge
# =========================================================

with st.expander(
    "Retrieved dataset knowledge",
    expanded=False,
):

    if retrieved_sources:

        for source in retrieved_sources:

            if not isinstance(
                source,
                dict,
            ):
                continue

            source_id = (
                source.get("id")
                or source.get("source_id")
                or source.get("chunk_id")
                or "dataset_source"
            )

            similarity = _number(
                source.get("score")
                or source.get("similarity")
                or source.get("similarity_score")
            )

            source_text = (
                source.get("text")
                or source.get("content")
                or "No source text was returned."
            )

            st.markdown(
                f"**{source_id}**"
            )

            if similarity is not None:

                st.caption(
                    "Similarity score: "
                    f"{similarity:.4f}"
                )

            st.write(
                source_text
            )

    else:

        st.write(
            "No dataset knowledge was retrieved for this objective."
        )


# =========================================================
# Charts
# =========================================================


def render_category_chart(result: Any) -> bool:
    """Render a clean category-sales chart."""

    def find_category_pairs(
        obj: Any,
    ) -> list[tuple[str, float]]:

        if isinstance(obj, pd.Series):
            pairs = []

            for label, value in obj.items():
                number = _number(value)

                if number is not None:
                    pairs.append(
                        (str(label), number)
                    )

            return pairs

        pairs = _category_pairs(obj)

        if pairs:
            return pairs

        if isinstance(obj, dict):

            for value in obj.values():

                pairs = find_category_pairs(value)

                if pairs:
                    return pairs

        if isinstance(obj, list):

            for value in obj:

                pairs = find_category_pairs(value)

                if pairs:
                    return pairs

        return []

    pairs = find_category_pairs(result)

    if not pairs:
        return False

    chart_df = pd.DataFrame(
        pairs,
        columns=["Category", "Sales"],
    )

    chart_df["Sales"] = pd.to_numeric(
        chart_df["Sales"],
        errors="coerce",
    )

    chart_df = chart_df.dropna(
        subset=["Sales"]
    )

    if chart_df.empty:
        return False

    chart_df = chart_df.sort_values(
        "Sales",
        ascending=False,
    )

    chart_df["Sales crore"] = (
        chart_df["Sales"] / 10_000_000
    )

    fig = px.bar(
        chart_df,
        x="Category",
        y="Sales crore",
        text="Sales crore",
    )
    fig.update_layout(
        paper_bgcolor="#121C30",
        plot_bgcolor="#121C30",
        font=dict(
            family="IBM Plex Sans, sans-serif",
            color="#E6EDF7",
            size=13,
        ),
        xaxis=dict(
            title=dict(
                font=dict(
                    color="#E6EDF7",
                    size=13,
                )
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=False,
            showline=True,
            linecolor="#22304A",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                font=dict(
                    color="#E6EDF7",
                    size=13,
                )
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=True,
            gridcolor="#22304A",
            showline=False,
            zeroline=False,
        ),
        margin=dict(
            l=55,
            r=25,
            t=20,
            b=55,
        ),
    )

    fig.update_traces(
        textfont=dict(
            color="#E6EDF7",
            size=12,
        ),
    )

    fig.update_traces(
        marker_color=ACCENT,
        texttemplate="₹%{text:.2f} Cr",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Sales: ₹%{customdata:,.2f}"
            "<extra></extra>"
        ),
        customdata=chart_df["Sales"],
    )

    fig.update_layout(
        height=330,
        showlegend=False,
        paper_bgcolor="#121C30",
        plot_bgcolor="#121C30",
        font=dict(
            family="IBM Plex Sans, Segoe UI, sans-serif",
            color="#E6EDF7",
            size=13,
        ),
        xaxis=dict(
            title=dict(
                text="Category",
                font=dict(
                    color="#E6EDF7",
                    size=13,
                ),
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=False,
            showline=True,
            linecolor="#22304A",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Sales (₹ crore)",
                font=dict(
                    color="#E6EDF7",
                    size=13,
                ),
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=True,
            gridcolor="#22304A",
            showline=False,
            zeroline=False,
            ticksuffix=" Cr",
        ),
        margin=dict(
            t=35,
            r=35,
            b=55,
            l=70,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )

    return True

def render_region_chart(result: Any) -> bool:
    """Render a clean regional-sales chart."""

    def find_region_pairs(
        obj: Any,
    ) -> list[tuple[str, float]]:

        if isinstance(obj, pd.Series):
            pairs = []

            for label, value in obj.items():
                number = _number(value)

                if number is not None:
                    pairs.append(
                        (str(label), number)
                    )

            return pairs

        pairs = _category_pairs(obj)

        if pairs:
            return pairs

        if isinstance(obj, dict):

            for value in obj.values():

                pairs = find_region_pairs(value)

                if pairs:
                    return pairs

        if isinstance(obj, list):

            for value in obj:

                pairs = find_region_pairs(value)

                if pairs:
                    return pairs

        return []

    pairs = find_region_pairs(result)

    if not pairs:
        return False

    chart_df = pd.DataFrame(
        pairs,
        columns=["Region", "Sales"],
    )

    chart_df["Sales"] = pd.to_numeric(
        chart_df["Sales"],
        errors="coerce",
    )

    chart_df = chart_df.dropna(
        subset=["Sales"]
    )

    if chart_df.empty:
        return False

    chart_df = chart_df.sort_values(
        "Sales",
        ascending=False,
    )

    chart_df["Sales crore"] = (
        chart_df["Sales"] / 10_000_000
    )

    fig = px.bar(
        chart_df,
        x="Region",
        y="Sales crore",
        text="Sales crore",
    )

    fig.update_layout(
        height=330,
        showlegend=False,
        paper_bgcolor="#121C30",
        plot_bgcolor="#121C30",
        font=dict(
            family="IBM Plex Sans, Segoe UI, sans-serif",
            color="#E6EDF7",
            size=13,
        ),
        xaxis=dict(
            title=dict(
                text="Region",
                font=dict(
                    color="#E6EDF7",
                    size=13,
                ),
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=False,
            showline=True,
            linecolor="#22304A",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Sales (₹ crore)",
                font=dict(
                    color="#E6EDF7",
                    size=13,
                ),
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=True,
            gridcolor="#22304A",
            showline=False,
            zeroline=False,
            ticksuffix=" Cr",
        ),
        margin=dict(
            t=35,
            r=35,
            b=55,
            l=70,
        ),
    )

    fig.update_traces(
        marker_color=ACCENT,
        texttemplate="₹%{text:.2f} Cr",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Sales: ₹%{customdata:,.2f}"
            "<extra></extra>"
        ),
        customdata=chart_df["Sales"],
    )

    fig.update_layout(
        height=330,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family=(
                "IBM Plex Sans, "
                "Segoe UI, sans-serif"
            ),
            color=INK,
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            linecolor=LINE,
        ),
        yaxis=dict(
            title="Sales (₹ crore)",
            showgrid=True,
            gridcolor="#22304A",
            zeroline=False,
            ticksuffix=" Cr",
        ),
        margin=dict(
            t=25,
            r=25,
            b=45,
            l=65,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )

    return True

def render_sales_trend_chart(
    raw: Any,
) -> None:

    recent = _number(
        _find_value(
            raw,
            (
                "recent_average_sales",
                "recent_7_day_average_sales",
                "recent_average",
            ),
        )
    )

    previous = _number(
        _find_value(
            raw,
            (
                "previous_average_sales",
                "previous_7_day_average_sales",
                "previous_average",
            ),
        )
    )

    if (
        recent is not None
        and previous is not None
    ):

        trend_df = pd.DataFrame(
            {
                "Period": [
                    "Previous 7 days",
                    "Recent 7 days",
                ],
                "Average daily sales": [
                    previous,
                    recent,
                ],
            }
        )

        fig = px.bar(
            trend_df,
            x="Period",
            y="Average daily sales",
            text="Average daily sales",
            template="plotly_white",
        )

        fig.update_traces(
            marker_color=ACCENT,
            texttemplate="₹%{text:,.0f}",
            textposition="outside",
            textfont=dict(
                color="#E6EDF7",
                size=12,
            ),
            cliponaxis=False,
        )

        fig.update_layout(
            title=None,
            showlegend=False,
            paper_bgcolor="#121C30",
            plot_bgcolor="#121C30",
            font=dict(
                family="IBM Plex Sans, Segoe UI, sans-serif",
                color="#E6EDF7",
                size=13,
            ),
            xaxis=dict(
                title=dict(
                    text="Period",
                    font=dict(
                        color="#E6EDF7",
                        size=13,
                    ),
                ),
                tickfont=dict(
                    color="#8FA3BF",
                    size=12,
                ),
                showgrid=False,
                showline=True,
                linecolor="#22304A",
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(
                    text="Average daily sales (₹)",
                    font=dict(
                        color="#E6EDF7",
                        size=13,
                    ),
                ),
                tickfont=dict(
                    color="#8FA3BF",
                    size=12,
                ),
                showgrid=True,
                gridcolor="#22304A",
                showline=False,
                zeroline=False,
                rangemode="tozero",
                tickprefix="₹",
            ),
            margin=dict(
                t=40,
                r=35,
                b=55,
                l=75,
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        return

    st.info(
        "Sales trend chart data was not returned."
    )


def _inventory_rows(
    raw: Any,
) -> list[dict[str, Any]]:

    # Actual backend output:
    # pandas DataFrame
    if isinstance(
        raw,
        pd.DataFrame,
    ):
        return raw.to_dict(
            orient="records"
        )

    if isinstance(
        raw,
        list,
    ):
        if all(
            isinstance(
                item,
                dict,
            )
            for item in raw
        ):
            return raw

    if isinstance(
        raw,
        dict,
    ):

        for key in (
            "top_20_highest_risk",
            "high_risk_inventory",
            "inventory_risk",
            "results",
            "data",
            "result",
            "output",
        ):

            value = raw.get(key)

            if isinstance(
                value,
                pd.DataFrame,
            ):
                return value.to_dict(
                    orient="records"
                )

            if isinstance(
                value,
                list,
            ):

                rows = [
                    item
                    for item in value
                    if isinstance(
                        item,
                        dict,
                    )
                ]

                if rows:
                    return rows

            if isinstance(
                value,
                dict,
            ):

                rows = _inventory_rows(
                    value
                )

                if rows:
                    return rows

        for value in raw.values():

            if isinstance(
                value,
                (dict, list, pd.DataFrame),
            ):

                rows = _inventory_rows(
                    value
                )

                if rows:
                    return rows

    return _records(raw)

def render_inventory_chart(
    raw: Any,
) -> None:

    source_rows = _inventory_rows(
        raw
    )

    rows = []

    for row in source_rows:

        store = (
            row.get("Store ID")
            or row.get("store_id")
            or row.get("store")
        )

        product = (
            row.get("Product ID")
            or row.get("product_id")
            or row.get("product")
        )

        coverage = _number(
            row.get("coverage_days")
            or row.get("coverage")
            or row.get("days_coverage")
            or row.get("Inventory Coverage")
            or row.get(
                "Inventory_Coverage_Days"
            )
        )

        if (
            store is not None
            and product is not None
            and coverage is not None
        ):

            rows.append(
                {
                    "Combination": (
                        f"{store} / {product}"
                    ),
                    "Coverage days": coverage,
                }
            )

    if not rows:

        st.info(
            "Inventory risk chart data was not returned."
        )

        return

    chart_df = (
        pd.DataFrame(rows)
        .sort_values(
            "Coverage days",
            ascending=True,
        )
        .head(10)
    )

    fig = px.bar(
        chart_df,
        x="Coverage days",
        y="Combination",
        orientation="h",
        text="Coverage days",
        template="plotly_white",
    )

    fig.update_traces(
        marker_color=RISK,
        texttemplate="%{text:.2f} days",
        textposition="outside",
        textfont=dict(
            color="#E6EDF7",
            size=12,
        ),
        cliponaxis=False,
    )

    fig.update_layout(
        title=None,
        showlegend=False,
        height=390,
        paper_bgcolor="#121C30",
        plot_bgcolor="#121C30",
        font=dict(
            family="IBM Plex Sans, Segoe UI, sans-serif",
            color="#E6EDF7",
            size=13,
        ),
        xaxis=dict(
            title=dict(
                text="Coverage (days)",
                font=dict(
                    color="#E6EDF7",
                    size=13,
                ),
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=True,
            gridcolor="#22304A",
            showline=False,
            zeroline=False,
            rangemode="tozero",
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(
                color="#E6EDF7",
                size=12,
            ),
            showgrid=False,
            showline=True,
            linecolor="#22304A",
            zeroline=False,
        ),
        margin=dict(
            t=25,
            r=75,
            b=60,
            l=125,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.caption(
        "High-risk combinations have coverage under 3 days."
    )


def render_forecast_chart(
    raw: Any,
) -> None:

    rows = []

    for row in _records(raw):

        date = (
            row.get("date")
            or row.get("Date")
            or row.get("forecast_date")
        )

        forecast = _number(
            row.get("forecast")
            or row.get("Forecast")
            or row.get("demand")
            or row.get("Forecast_Demand")
        )

        if (
            date is not None
            and forecast is not None
        ):

            rows.append(
                (
                    str(date),
                    forecast,
                )
            )

    if not rows:

        st.info(
            "Demand forecast chart data was not returned."
        )

        return

    forecast_df = pd.DataFrame(
        rows,
        columns=[
            "Date",
            "Units per day",
        ],
    )

    fig = px.line(
        forecast_df,
        x="Date",
        y="Units per day",
        markers=True,
        template="plotly_white",
    )

    fig.update_traces(
        line_color=ACCENT,
        marker_color=ACCENT,
        marker_size=7,
        line_width=3,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Forecast: %{y:.2f} units/day"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title=None,
        showlegend=False,
        height=350,
        paper_bgcolor="#121C30",
        plot_bgcolor="#121C30",
        font=dict(
            family="IBM Plex Sans, Segoe UI, sans-serif",
            color="#E6EDF7",
            size=13,
        ),
        xaxis=dict(
            title=dict(
                text="Date",
                font=dict(
                    color="#E6EDF7",
                    size=13,
                ),
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=False,
            showline=True,
            linecolor="#22304A",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Units per day",
                font=dict(
                    color="#E6EDF7",
                    size=13,
                ),
            ),
            tickfont=dict(
                color="#8FA3BF",
                size=12,
            ),
            showgrid=True,
            gridcolor="#22304A",
            showline=False,
            zeroline=False,
            rangemode="tozero",
        ),
        margin=dict(
            t=30,
            r=35,
            b=60,
            l=70,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


# =========================================================
# Render only charts for tools that actually ran
# =========================================================
active_charts = []

if (
    "get_sales_by_category"
    in tool_results
):
    active_charts.append(
        (
            "Category sales",
            render_category_chart,
            tool_results[
                "get_sales_by_category"
            ],
        )
    )
if (
    "get_sales_by_region"
    in tool_results
):
    active_charts.append(
        (
            "Regional sales",
            render_region_chart,
            tool_results[
                "get_sales_by_region"
            ],
        )
    )

if (
    "get_sales_trend"
    in tool_results
):
    active_charts.append(
        (
            "Sales trend",
            render_sales_trend_chart,
            tool_results,
        )
    )

if (
    "get_high_risk_inventory"
    in tool_results
):
    active_charts.append(
        (
            "Inventory risk",
            render_inventory_chart,
            tool_results[
                "get_high_risk_inventory"
            ],
        )
    )

if (
    "get_demand_forecast"
    in tool_results
):
    active_charts.append(
        (
            "Demand forecast",
            render_forecast_chart,
            tool_results[
                "get_demand_forecast"
            ],
        )
    )

if active_charts:
    st.subheader(
        "Charts"
    )

for index in range(
    0,
    len(active_charts),
    2,
):

    pair = active_charts[
        index:index + 2
    ]

    if len(pair) == 2:

        left, right = st.columns(
            2,
            gap="medium",
        )

        with left:

            st.markdown(
                f"**{pair[0][0]}**"
            )

            pair[0][1](
                pair[0][2]
            )

        with right:

            st.markdown(
                f"**{pair[1][0]}**"
            )

            pair[1][1](
                pair[1][2]
            )

    else:

        st.markdown(
            f"**{pair[0][0]}**"
        )

        pair[0][1](
            pair[0][2]
        )
