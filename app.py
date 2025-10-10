# Greeno Big Three v1.6.6 — strict parser (TOTAL-aware bins, left-label) +
# collapsible ADs + reason totals (Missing + Attitude) + text-only Eric email +
# Categories: To-go Missing Complaints (To-Go/Delivery) + Attitude (all segments) + Other (placeholder)
# + High-contrast table styling (no GIF loader)
import io, os, re, base64, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── STYLING HELPERS (HIGH CONTRAST) ─────────────────
def style_table(df: pd.DataFrame, highlight_grand_total: bool = True) -> "pd.io.formats.style.Styler":
    """
    Alternating row shading + dark text so values are readable in dark mode.
    Also highlights the '— Grand Total —' row.
    """
    def zebra(series):
        return [
            "background-color: #F5F7FA" if i % 2 == 0 else "background-color: #E9EDF2"
            for i, _ in enumerate(series)
        ]

    sty = (
        df.style
        .set_properties(
            **{
                "color": "#111",
                "border-color": "#CCD3DB",
                "border-width": "0.5px",
                "border-style": "solid",
            }
        )
        .apply(zebra, axis=0)
    )

    if highlight_grand_total:
        def highlight_total(row):
            if str(row.name) == "— Grand Total —":
                return [
                    "background-color: #FFE39B; color: #111; font-weight: 700;"
                ] * len(row)
            return [""] * len(row)

        sty = sty.apply(highlight_total, axis=1)

    # Make index text dark too
    sty = sty.set_table_styles(
        [
            {
                "selector": "th.row_heading, th.blank",
                "props": [("color", "#111"), ("border-color", "#CCD3DB")],
            }
        ]
    )

    return sty

# ───────────────── HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.6.6", layout="wide")

logo_path = "greenosu.webp"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f'<img src="data:image/webp;base64,{logo_data}" width="240" style="border-radius:12px;">'
else:
    logo_html = '<div style="width:240px;height:240px;background:#fff;border-radius:12px;"></div>'

st.markdown(
    f"""
<div style="
    background-color:#0078C8; color:white; padding:2rem 2.5rem; border-radius:10px;
    display:flex; align-items:center; gap:2rem; box-shadow:0 4px 12px rgba(0,0,0,.2);
    position:sticky; top:0; z-index:50;
">
  {logo_html}
  <div style="display:flex; flex-direction:column; justify-content:center;">
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.6.6</h1>
      <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        Strict parsing + TOTAL-aware bins · Collapsible AD sections · Reason totals (Missing + Attitude)
        · Text-only Eric email · Category summaries · High-contrast tables
      </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ───────────────── SIDEBAR UPLOAD ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose the PDF report", type=["pdf"])
    st.caption(
        "Parses labels from the left side. Missing uses To-Go & Delivery; Attitude includes all segments."
    )

if not up:
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"]{
            outline:3px solid #2e7df6; box-shadow:0 0 0 4px rgba(46,125,246,.25);
            animation:pulse 1.2s ease-in-out infinite; border-radius:6px;
          }
          @keyframes pulse{0%{outline-color:#2e7df6}50%{outline-color:#90caf9}100%{outline-color:#2e7df6}}
        </style>
        <div style="text-align:center; margin-top:8vh;">
          <div style="font-size:3rem; line-height:1;">⬅️</div>
          <div style="font-size:1.25rem; font-weight:600; margin-top:.5rem;">
            Upload your PDF in the <em>left sidebar</em>
          </div>
          <div style="opacity:.85; margin-top:.25rem;">
            Click <strong>“Choose a PDF report”</strong> to begin.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = up.read()
if pdfplumber is None:
    st.error("pdfplumber is not installed. Run: pip install pdfplumber")
    st.stop()

# ───────────────── CONSTANTS, PARSER, AND ALL LOGIC ─────────────────
# (Use your working 1.6.5 file content here — unchanged — only the style_table is different)
# Keep your existing:
# - parsing functions
# - category summary
# - reason totals for Missing and Attitude
# - email generator (text-only Eric voice)
# - export sections
# - spinner “Roll Tide…”

# Example of keeping the same spinner:
with st.spinner("Roll Tide…"):
    header_x_map, raw_data, ordered_headers, pairs_debug = parse_pdf_build_ad_store_period_map(file_bytes)

# Everything else stays identical to your v1.6.5 file
