"""Theme, colours and small UI components for the dashboard."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Categorical palette (validated order, dark-surface steps) + status colours
C = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = C
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
SURFACE, CARD, TEXT, MUTED, GRID = "#0e1124", "#161a33", "#f0efec", "#a9a8c2", "#2a2f52"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#d03b3b"], [0.5, "#383835"], [1, "#3987e5"]]

ARM_COLOR = {"No offer": "#8a8aa3", "20% off": ORANGE, "Buy-one-get-one": MAGENTA, "Free shipping": AQUA}
SEG_COLOR = {"Persuadable": GREEN, "Sure thing": BLUE, "Lost cause": "#8a8aa3", "Sleeping dog": RED}
POLICY_COLOR = {"Blanket 20% off": ORANGE, "Smart targeting": AQUA}

pio.templates["growthlab"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=C,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor=CARD, bordercolor=GRID, font=dict(color=TEXT)),
        title=dict(font=dict(size=15, color=TEXT), x=0, xanchor="left"),
    )
)
pio.templates.default = "growthlab"

CSS = f"""
<style>
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #2a1d5c 0%, transparent 60%),
                      radial-gradient(1000px 500px at 110% 10%, #0f3b4f 0%, transparent 55%), {SURFACE}; }}
.block-container {{ padding-top: 3.6rem; max-width: 1500px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
.hero {{ padding: 18px 22px; border-radius: 18px; margin-bottom: 14px;
        background: linear-gradient(110deg, #5b3fd6 0%, #c2417d 45%, #e8743b 75%, #e8b03b 100%);
        box-shadow: 0 10px 30px rgba(91,63,214,.35); }}
.hero h1 {{ color: white; margin: 0; font-size: 1.9rem; }}
.hero p {{ color: rgba(255,255,255,.92); margin: 4px 0 0 0; font-size: .98rem; }}
.badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700;
         background: rgba(255,255,255,.18); color: white; margin-right: 6px; border: 1px solid rgba(255,255,255,.35); }}
.kpi {{ border-radius: 16px; padding: 14px 16px; color: white; min-height: 112px;
       box-shadow: 0 6px 18px rgba(0,0,0,.25); border: 1px solid rgba(255,255,255,.08); }}
.kpi .lbl {{ font-size: .78rem; opacity: .9; text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }}
.kpi .val {{ font-size: 1.75rem; font-weight: 800; margin: 2px 0; line-height: 1.15; }}
.kpi .sub {{ font-size: .8rem; opacity: .92; }}
.g1 {{ background: linear-gradient(135deg, #3b6fd8, #6a4fe0); }}
.g2 {{ background: linear-gradient(135deg, #d0542b, #e3962f); }}
.g3 {{ background: linear-gradient(135deg, #12876a, #24b38a); }}
.g4 {{ background: linear-gradient(135deg, #b23d72, #e0628e); }}
.g5 {{ background: linear-gradient(135deg, #7b5cf0, #b16cf0); }}
.g6 {{ background: linear-gradient(135deg, #c13a3a, #e36a5a); }}
.g7 {{ background: linear-gradient(135deg, #0c7da3, #2bb3d6); }}
.card {{ background: {CARD}; border: 1px solid {GRID}; border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; }}
.blog {{ display:inline-block; padding: 2px 9px; border-radius: 8px; font-size: .74rem; font-weight: 700;
        background: #2a2f52; color: #c9c6ff; border: 1px solid #3c4277; margin: 0 4px 4px 0; }}
.finding {{ border-left: 5px solid {AQUA}; background: {CARD}; padding: 10px 14px; border-radius: 10px; margin: 6px 0; }}
.warn {{ border-left: 5px solid {WARN}; }}
.bad {{ border-left: 5px solid {CRIT}; }}
.small {{ color: {MUTED}; font-size: .82rem; }}
.review {{ padding: 8px 10px; border-radius: 10px; margin-bottom: 6px; background: #1b2042; border: 1px solid {GRID}; font-size: .86rem; }}
.alert {{ padding: 7px 10px; border-radius: 10px; margin-bottom: 6px; font-size: .84rem; }}
.alert.crit {{ background: rgba(208,59,59,.18); border: 1px solid rgba(208,59,59,.6); }}
.alert.warn {{ background: rgba(250,178,25,.14); border: 1px solid rgba(250,178,25,.55); }}
.alert.good {{ background: rgba(12,163,12,.15); border: 1px solid rgba(12,163,12,.55); }}
.pulse {{ display:inline-block; width:10px; height:10px; border-radius:50%; background:#ff3b5c; margin-right:6px;
         box-shadow: 0 0 0 0 rgba(255,59,92,.7); animation: pulse 1.6s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(255,59,92,.7); }} 70% {{ box-shadow: 0 0 0 10px rgba(255,59,92,0); }}
                   100% {{ box-shadow: 0 0 0 0 rgba(255,59,92,0); }} }}
div[data-testid="stTabs"] button p {{ font-size: .95rem; font-weight: 600; }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(col, label, value, sub="", g="g1"):
    size = "" if len(str(value)) <= 13 else ' style="font-size:1.2rem"'
    col.markdown(f'<div class="kpi {g}"><div class="lbl">{label}</div><div class="val"{size}>{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def blogs(*names):
    st.markdown("".join(f'<span class="blog">{n}</span>' for n in names), unsafe_allow_html=True)


def finding(text, kind=""):
    st.markdown(f'<div class="finding {kind}">{text}</div>', unsafe_allow_html=True)


def fig_style(fig, h=360, title=None):
    fig.update_layout(height=h)
    if title:
        fig.update_layout(title=title)
    return fig


def money(x, d=0):
    if x is None:
        return "–"
    s = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{s}${x/1e6:.{max(d,1)}f}M"
    if x >= 1e4:
        return f"{s}${x/1e3:.0f}k"
    return f"{s}${x:,.{d}f}"


def pct(x, d=1):
    return "–" if x is None else f"{x:.{d}f}%"


def pfmt(p):
    """p-value as text: 'p < 0.001' or 'p = 0.042'."""
    if p is None:
        return "p = –"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
