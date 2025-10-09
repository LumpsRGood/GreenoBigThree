# app.py — Greeno Big Three v1.4 (Area Director extractor)

import io, os, re, base64
from collections import defaultdict
from typing import List, Tuple
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.4", layout="wide")

logo_path = "greenosu.webp"  # your local logo
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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.4</h1>
      <div style="height:5px; background-color:#F44336; width:220px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        Step 1: Parse the PDF and list every <strong>Area Director</strong>.
      </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ───────────────── SIDEBAR UPLOAD ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose a PDF report", type=["pdf"])
    st.caption("This step ignores numbers — we just find Area Director names reliably.")

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
    st.error("pdfplumber is not available. Install it and rerun.")
    st.stop()

# ───────────────── PARSING UTILITIES ─────────────────
HEADINGS = {
    "Area Director", "Restaurant", "Order Visit Type", "Reason for Contact"
}
SECTION_WORDS = {
    "Delivery", "Dine-In", "To Go", "To-Go", "To-go",
    "Total", "Delivery Total:", "Dine-In Total:", "To Go Total:", "To-go Total:"
}

STORE_LINE_RX = re.compile(r"^\s*\d{4,6}\s*-\s+.*")  # e.g., "5456 - Toledo OH (Talmadge Road)"

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

def extract_lines(page) -> List[Tuple[float, str]]:
    """
    Returns a list of (y_mid, line_text) from top to bottom for one page.
    Uses pdfplumber.extract_words and groups by y-mid to reconstruct lines.
    """
    words = page.extract_words(
        x_tolerance=1.2, y_tolerance=2.2,
        keep_blank_chars=False, use_text_flow=True
    )
    lines = defaultdict(list)
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[y_mid].append(w)
    out = []
    for y, items in sorted(lines.items(), key=lambda kv: kv[0]):
        items = sorted(items, key=lambda w: w["x0"])
        text = " ".join(it["text"].strip() for it in items if it["text"].strip())
        if text:
            out.append((y, text))
    return out

def is_heading_or_section(s: str) -> bool:
    s_clean = s.strip()
    if s_clean in HEADINGS:
        return True
    # Normalize basic "To-go" variants
    norm = s_clean.replace("—", "-").replace("–", "-")
    if norm in SECTION_WORDS:
        return True
    if norm.endswith(" Total:"):
        return True
    return False

def is_probable_name(s: str) -> bool:
    """
    Heuristic for a personal name: 2–4 words, starts with capital letters,
    not a store line, not a heading/section.
    """
    if STORE_LINE_RX.match(s):
        return False
    if is_heading_or_section(s):
        return False
    parts = [p for p in re.split(r"\s+", s.strip()) if p]
    if len(parts) < 2 or len(parts) > 5:
        return False
    # At least two parts should start with capital A-Z
    caps = sum(1 for p in parts if re.match(r"^[A-Z][a-zA-Z'\-]+$", p))
    return caps >= 2

def find_area_directors_from_lines(lines: List[Tuple[float, str]]) -> List[str]:
    """
    Strategy:
      - Identify store lines "#### - City (Location)".
      - For each store line, look backward to the nearest previous non-empty line
        that looks like a person's name (and isn't a heading/section).
      - Deduplicate while preserving first-seen order.
    """
    names_found: List[str] = []
    seen = set()

    for idx, (_, txt) in enumerate(lines):
        if STORE_LINE_RX.match(txt):
            # walk upward to find the closest plausible name
            j = idx - 1
            while j >= 0:
                s = lines[j][1].strip()
                if s and not is_heading_or_section(s):
                    if is_probable_name(s):
                        if s not in seen:
                            seen.add(s)
                            names_found.append(s)
                        break
                j -= 1
    return names_found

# ───────────────── RUN PARSER ─────────────────
with st.spinner("Scanning pages for Area Director names…"):
    all_lines_by_page: List[List[Tuple[float, str]]] = []
    names: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                lines = extract_lines(p)
                all_lines_by_page.append(lines)
                names.extend(find_area_directors_from_lines(lines))
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        st.stop()

# Deduplicate while preserving order (in case names repeat across pages)
seen = set()
names_unique = []
for n in names:
    if n not in seen:
        seen.add(n)
        names_unique.append(n)

# ───────────────── DISPLAY ─────────────────
st.success(f"Found {len(names_unique)} Area Director name(s).")

st.subheader("Area Directors")
if names_unique:
    for n in names_unique:
        st.markdown(f"- **{n}**")
else:
    st.warning("No Area Director names were detected. Expand the debug view below to inspect raw lines.")

# Quick copy + export
copy_text = "\n".join(names_unique)
st.text_area("Quick copy", value=copy_text, height=160)

df_names = pd.DataFrame({"Area Director": names_unique})
c1, c2 = st.columns(2)
with c1:
    csv = df_names.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, file_name="area_directors.csv", mime="text/csv")
with c2:
    import io as _io
    from pandas import ExcelWriter
    buff = _io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df_names.to_excel(writer, index=False, sheet_name="Area Directors")
    st.download_button("📥 Download Excel", data=buff.getvalue(),
                       file_name="area_directors.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ───────────────── DEBUG ─────────────────
with st.expander("🔎 Debug: view raw lines per page"):
    st.caption("If a name is missing/wrong, check how the PDF text is being read here.")
    pg = st.number_input("Page to preview", min_value=1, max_value=len(all_lines_by_page), value=1, step=1)
    lines_preview = all_lines_by_page[pg-1]
    st.write(pd.DataFrame([{"y": y, "text": t} for y, t in lines_preview]))
    st.caption("Tip: The parser looks for a store line like '5456 - City' and takes the nearest line above it that looks like a person’s name.")
