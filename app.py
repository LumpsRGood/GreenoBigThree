# app.py — Greeno Big Three v1.4.1 (Area Director extractor, anchored to heading)

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
st.set_page_config(page_title="Greeno Big Three v1.4.1", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.4.1</h1>
      <div style="height:5px; background-color:#F44336; width:220px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        Step 1 (clean): Parse the PDF and list every <strong>Area Director</strong> correctly.
      </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ───────────────── SIDEBAR UPLOAD ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose a PDF report", type=["pdf"])
    st.caption("This step anchors on the “Area Director” heading to find names.")

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
# Keywords that show up in “reason for contact” or titles — if any token appears, it’s not a name.
STOP_TOKENS = {
    "necessary","info","information","compensation","offered","restaurant","operational","issues",
    "missing","condiments","ingredient","food","bev","beverage","order","wrong","cold","slow",
    "unfriendly","manager","did","not","attempt","resolve","issue","appearance","packaging","to",
    "go","to-go","dine-in","delivery","total","guest","ticket","incorrect","understaffed","poor",
    "quality","presentation","overcooked","burnt","undercooked","host","server","greet","portion"
}

# headings we expect in the left rail
HEADINGS = {"Area Director","Restaurant","Order Visit Type","Reason for Contact"}

STORE_LINE_RX = re.compile(r"^\s*\d{4,6}\s*-\s+.*")  # "5456 - City (Location)"

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

def extract_words(page):
    return page.extract_words(
        x_tolerance=1.2, y_tolerance=2.2,
        keep_blank_chars=False, use_text_flow=True
    )

def group_lines(words):
    """
    Return list of dicts: {"y": y_mid, "x_min": min x0, "text": "joined text"}
    """
    from collections import defaultdict
    lines = defaultdict(list)
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[y_mid].append(w)
    out = []
    for y, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        ws = sorted(ws, key=lambda w: w["x0"])
        text = " ".join(w["text"].strip() for w in ws if w["text"].strip())
        if text:
            out.append({"y": y, "x_min": ws[0]["x0"], "text": text})
    return out

def looks_like_name(s: str) -> bool:
    """
    Stricter rule set:
      - 2–4 tokens
      - Each token starts with A–Z and is alphabetic/apostrophe/hyphen only
      - No digits, no parentheses/hyphens that indicate titles/locations
      - None of the STOP_TOKENS present (case-insensitive)
    """
    s_clean = s.strip()
    if STORE_LINE_RX.match(s_clean):
        return False
    if s_clean in HEADINGS:
        return False
    if any(ch.isdigit() for ch in s_clean):
        return False
    if "(" in s_clean or ")" in s_clean or " - " in s_clean or "—" in s_clean or "–" in s_clean:
        return False

    parts = [p for p in re.split(r"\s+", s_clean) if p]
    if len(parts) < 2 or len(parts) > 4:
        return False

    for p in parts:
        if not re.match(r"^[A-Z][a-zA-Z'\-]+$", p):
            return False

    toks = {t.lower() for t in re.split(r"[^\w]+", s_clean) if t}
    if toks & STOP_TOKENS:
        return False

    return True

def find_area_directors_on_page(lines: List[dict]) -> List[str]:
    """
    Anchor on the 'Area Director' heading.
    Strategy:
      - Find any line whose text == 'Area Director' (case-insensitive).
      - Take the next 1–5 lines; the first that looks_like_name() AND is left-aligned
        (x close to the page's left margin) is the Area Director for this page.
      - Fallback: if no heading anchor is found, use the first candidate above the
        first store line (but still apply strict rules).
    """
    names = []

    # Estimate left margin from the smallest x_min on page
    left_margin = min(l["x_min"] for l in lines) if lines else 0.0

    def is_left_aligned(x):
        return (x - left_margin) <= 20  # within ~20 px of left margin

    # 1) Anchor path
    idxs = [i for i, L in enumerate(lines) if L["text"].strip().lower() == "area director"]
    for i in idxs:
        for j in range(i+1, min(i+6, len(lines))):
            cand = lines[j]
            if looks_like_name(cand["text"]) and is_left_aligned(cand["x_min"]):
                names.append(cand["text"].strip())
                break  # one per anchor occurrence

    # 2) Fallback path (once per page): name above first store line
    if not names:
        first_store_idx = next((k for k, L in enumerate(lines) if STORE_LINE_RX.match(L["text"])), None)
        if first_store_idx is not None:
            for j in range(first_store_idx-1, max(first_store_idx-6, -1), -1):
                cand = lines[j]
                if looks_like_name(cand["text"]) and is_left_aligned(cand["x_min"]):
                    names.append(cand["text"].strip())
                    break

    return names

# ───────────────── RUN PARSER ─────────────────
with st.spinner("Scanning pages for Area Director names…"):
    all_lines_by_page: List[List[dict]] = []
    names: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                words = extract_words(p)
                lines = group_lines(words)
                all_lines_by_page.append(lines)
                names.extend(find_area_directors_on_page(lines))
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        st.stop()

# Deduplicate while preserving order
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
    st.warning("No Area Director names were detected. Expand the debug view to inspect the raw lines.")

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
    buff = _io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df_names.to_excel(writer, index=False, sheet_name="Area Directors")
    st.download_button("📥 Download Excel", data=buff.getvalue(),
                       file_name="area_directors.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ───────────────── DEBUG ─────────────────
with st.expander("🔎 Debug: raw lines per page (y, x_min, text)"):
    st.caption("If a name is missing/wrong, check how the PDF text is being read here.")
    page_num = st.number_input("Page to preview", min_value=1, max_value=len(all_lines_by_page), value=1, step=1)
    lines_preview = all_lines_by_page[page_num-1]
    st.dataframe(pd.DataFrame(lines_preview))
    st.caption("Parser rule: take the first left-aligned line AFTER 'Area Director' that looks like a name. "
               "Fallback: first name-like line immediately above the first store line on that page.")
