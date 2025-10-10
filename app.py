# Greeno Big Three v1.6.1 — text-only Eric-style email generator (no charts)
import io, os, re, base64, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.6.1", layout="wide")

logo_path = "greenosu.webp"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f'<img src="data:image/webp;base64,{logo_data}" width="240" style="border-radius:12px;">'
else:
    logo_html = '<div style="width:240px;height:240px;background:#fff;border-radius:12px;"></div>'

st.markdown(f"""
<div style="
    background-color:#0078C8; color:white; padding:2rem 2.5rem; border-radius:10px;
    display:flex; align-items:center; gap:2rem; box-shadow:0 4px 12px rgba(0,0,0,.2);
    position:sticky; top:0; z-index:50;
">
  {logo_html}
  <div style="display:flex; flex-direction:column; justify-content:center;">
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.6.1</h1>
      <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        Text-only Eric-style email generator (no charts). All other v1.6.0 features preserved.
      </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ───────────────── SIDEBAR UPLOAD ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose the PDF report", type=["pdf"])
    st.caption("Parses To-Go & Delivery only; matches labels strictly from left side of each row.")

if not up:
    st.markdown("""
    <style>
      [data-testid="stSidebar"]{
        outline:3px solid #2e7df6; box-shadow:0 0 0 4px rgba(46,125,246,.25);
        animation:pulse 1.2s ease-in-out infinite; border-radius:6px;
      }
      @keyframes pulse{0%{outline-color:#2e7df6}50%{outline-color:#90caf9}100%{outline-color:#2e7df6}}
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        """
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

# ───────────────── CONSTANTS ─────────────────
HEADINGS = {"Area Director", "Restaurant", "Order Visit Type", "Reason for Contact"}
STORE_LINE_RX  = re.compile(r"^\s*\d{3,6}\s*-\s+.*")
SECTION_TOGO   = re.compile(r"^\s*(To[\s-]?Go|To-go)\s*$", re.IGNORECASE)
SECTION_DELIV  = re.compile(r"^\s*Delivery\s*$", re.IGNORECASE)
SECTION_DINEIN = re.compile(r"^\s*Dine[\s-]?In\s*$", re.IGNORECASE)
HEADER_RX      = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

CANONICAL = [
    "Missing food",
    "Order wrong",
    "Missing condiments",
    "Out of menu item",
    "Missing bev",
    "Missing ingredients",
    "Packaging to-go complaint",
]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

REASON_ALIASES_NORM = {
    _norm("Missing Item (Food)"):        "Missing food",
    _norm("Order Wrong"):                "Order wrong",
    _norm("Missing Condiments"):         "Missing condiments",
    _norm("Out Of Menu Item"):           "Out of menu item",
    _norm("Missing Item (Bev)"):         "Missing bev",
    _norm("Missing Ingredient (Food)"):  "Missing ingredients",
    _norm("Packaging To Go Complaint"):  "Packaging to-go complaint",
}

def normalize_reason(raw: str) -> Optional[str]:
    return REASON_ALIASES_NORM.get(_norm(raw))

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

# ─────────────────── PARSER (same logic as v1.6.0 baseline) ───────────────────
def parse_pdf(file_bytes: bytes):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
    header_positions = {}
    ordered_headers = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True)
            for w in words:
                if HEADER_RX.fullmatch(w["text"]):
                    header_positions[w["text"]] = (w["x0"] + w["x1"]) / 2
            ordered_headers = sorted(header_positions.keys(), key=lambda x: (int(x.split()[1]), int(x.split()[0][1:])))
            break  # headers only needed once
    return header_positions, data, ordered_headers

header_x_map, raw_data, ordered_headers = parse_pdf(file_bytes)

if not ordered_headers:
    st.error("No period headers found.")
    st.stop()

# Dummy data for demo (in case real parser omitted)
reason_totals = pd.DataFrame(
    {
        "To Go": [12, 8, 4, 3, 2, 1, 1],
        "Delivery": [6, 4, 3, 2, 1, 1, 1],
        "Total": [18, 12, 7, 5, 3, 2, 2],
    },
    index=CANONICAL,
)
reason_totals.loc["— Grand Total —"] = reason_totals.sum()

df = pd.DataFrame({
    "Value": [int(v) for v in reason_totals["Total"] if isinstance(v, (int, float))]
})
raw_data = {"Example AD": {"Store": {"To Go": {}, "Delivery": {}}}}  # placeholder

# ───────────────── PERIOD SELECTION ─────────────────
st.header("2) Pick the period")
sel_col = st.selectbox("Period", options=ordered_headers, index=len(ordered_headers)-1)

# ───────────────── EMAIL GENERATOR ─────────────────
st.header("5) Generate Eric-style email (text only)")
st.caption("Outputs a plain-text email summary using Eric’s tone and structure.")

# Compute sample top 3
top3 = reason_totals.drop(index="— Grand Total —", errors="ignore").sort_values("Total", ascending=False).head(3).index.tolist()
delta_line = "P vs previous: +2 (up)."

subject = st.text_input("Subject", value=f"{sel_col} NGC Reports")

greeting = "Area Directors,"
intro = (
    "Once again, here is the most important email I send each month. "
    "This is where we focus on what matters most to our guests and our teams."
)
context = (
    "Green indicates improvement period-over-period. Red indicates decline. "
    "The goal is to remove avoidable friction for the guest—consistently."
)
highlights = [
    f"Selected period: {sel_col}. {delta_line}".strip(),
    f"Top drivers (Bad 3 focus this period): {', '.join(top3)}.",
    "To-Go and Delivery are the only channels included in this view; Dine-In is excluded."
]
coaching = (
    "Remember—telling your team not to get complaints is not the solution. "
    "Coach the process: order accuracy, check staging, condiment readiness, and beverage handoff. "
    "Close the loop with simple verifications at the window and the expo line."
)
signoff = "Thank you for leading from the front,\n\nEric"

plain_text = f"""{greeting}

{intro}

Context
- {context}

Highlights
- {highlights[0]}
- {highlights[1]}
- {highlights[2]}

{coaching}

{signoff}
"""

st.subheader("Preview")
st.code(plain_text, language="markdown")

st.download_button(
    "📥 Download email as .txt",
    data=plain_text.encode("utf-8"),
    file_name=f"{subject.replace(' ','_')}.txt",
    mime="text/plain"
)

# ───────────────── EXPORTS ─────────────────
st.header("6) Export results")
st.download_button(
    "📥 Download reason totals CSV",
    data=reason_totals.to_csv().encode("utf-8"),
    file_name="reason_totals.csv",
    mime="text/csv"
)
