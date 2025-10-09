# Greeno Big Three v1.5 — AD → Store → (To Go, Delivery) → 7 reasons, per-period
import io, os, re, base64
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.5", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.5</h1>
      <div style="height:5px; background-color:#F44336; width:260px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        AD → Store totals for <strong>To Go</strong> & <strong>Delivery</strong> across the 7 targeted reasons, for the selected period.
      </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ───────────────── SIDEBAR UPLOAD ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose the PDF report", type=["pdf"])
    st.caption("This looks for headers like ‘P9 24’, ‘P1 25’ and maps numbers by x-position.")

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

# ───────────────── PARSING CONSTANTS ─────────────────
HEADINGS = {"Area Director", "Restaurant", "Order Visit Type", "Reason for Contact"}
STORE_LINE_RX = re.compile(r"^\s*\d{4,6}\s*-\s+.*")  # e.g., "5456 - City (Location)"
SECTION_TOGO = re.compile(r"^\s*(To[\s-]?Go|To-go)\s*$", re.IGNORECASE)
SECTION_DELIV = re.compile(r"^\s*Delivery\s*$", re.IGNORECASE)
SECTION_DINEIN = re.compile(r"^\s*Dine[\s-]?In\s*$", re.IGNORECASE)

# Period headers like "P9 24", "P10 24", "P1 25"
HEADER_RX = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

# Canonical reasons (and robust matchers for typical variants)
CANONICAL = [
    "Missing food",
    "Order wrong",
    "Missing condiments",
    "Out of menu item",
    "Missing bev",
    "Missing ingredients",
    "Packaging to-go complaint",
]
# map canonical -> list of regex patterns (case-insensitive) for matching raw label lines
REASON_PATTERNS = {
    "Missing food":              [re.compile(r"\bMissing (Item )?\(?Food\)?\b", re.IGNORECASE)],
    "Order wrong":               [re.compile(r"\bOrder wrong\b", re.IGNORECASE)],
    "Missing condiments":        [re.compile(r"\bMissing Condiment(s)?\b", re.IGNORECASE)],
    "Out of menu item":          [re.compile(r"\bOut Of Menu Item\b", re.IGNORECASE)],
    "Missing bev":               [re.compile(r"\bMissing (Item )?\(?Bev\)?\b", re.IGNORECASE),
                                  re.compile(r"\bMissing (Beverage|Drink)\b", re.IGNORECASE)],
    "Missing ingredients":       [re.compile(r"\bMissing Ingredient(s)?\b", re.IGNORECASE)],
    "Packaging to-go complaint": [re.compile(r"\bPackaging (To[\s-]?Go|to-go) Complaint\b", re.IGNORECASE)],
}

def normalize_reason(raw: str) -> Optional[str]:
    s = raw.strip()
    for canon, patterns in REASON_PATTERNS.items():
        for rx in patterns:
            if rx.search(s):
                return canon
    return None

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

# ───────────────── HEADER DETECTION ─────────────────
def find_period_headers(page) -> List[Tuple[str, float, float]]:
    """
    Returns [(header_text, x_center, y_mid)] for headers like 'P9 24'.
    """
    words = page.extract_words(x_tolerance=1.0, y_tolerance=2.0, keep_blank_chars=False, use_text_flow=True)
    lines = defaultdict(list)
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[y_mid].append(w)

    headers = []
    for ymid, ws in lines.items():
        ws = sorted(ws, key=lambda w: w["x0"])
        merged = []
        i = 0
        while i < len(ws):
            t = ws[i]["text"]; x0, x1 = ws[i]["x0"], ws[i]["x1"]
            cand, x1c = t, x1
            if i + 1 < len(ws):
                t2 = ws[i + 1]["text"]
                cand2 = f"{t} {t2}"
                if HEADER_RX.fullmatch(cand2):
                    x1c = ws[i + 1]["x1"]; cand = cand2; i += 2
                    merged.append((cand, (x0 + x1c)/2, ymid)); continue
            if HEADER_RX.fullmatch(cand):
                merged.append((cand, (x0 + x1)/2, ymid))
            i += 1
        if len(merged) >= 3:
            headers.extend(merged)

    # Deduplicate by text
    seen = {}
    for txt, xc, ym in sorted(headers, key=lambda h: (h[2], h[1])):
        seen.setdefault(txt, (txt, xc, ym))
    return list(seen.values())

def sort_headers(headers: List[str]) -> List[str]:
    def key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999, 999)
    return sorted(headers, key=key)

# ───────────────── LINE GROUPER ─────────────────
def extract_words_grouped(page):
    words = page.extract_words(
        x_tolerance=1.4, y_tolerance=2.4,
        keep_blank_chars=False, use_text_flow=True
    )
    lines = defaultdict(list)
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[y_mid].append(w)
    out = []
    for y, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        ws = sorted(ws, key=lambda w: w["x0"])
        text = " ".join(w["text"].strip() for w in ws if w["text"].strip())
        if text:
            out.append({"y": y, "x_min": ws[0]["x0"], "words": ws, "text": text})
    return out

# ───────────────── CORE PARSE ─────────────────
def parse_pdf_build_ad_store_period_map(file_bytes: bytes) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, Dict[str, int]]]], List[str]]:
    """
    Returns:
      header_positions: mapping header -> x_center
      data: {AD: {Store: {Section: {CanonicalReason: value_int}}}}
      ordered_headers: ordered list of period headers
    """
    header_positions: Dict[str, float] = {}
    ordered_headers: List[str] = []
    data: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        carry_headers = None
        for page in pdf.pages:
            headers = find_period_headers(page)
            if not headers and carry_headers:
                headers = carry_headers
            if not headers:
                continue
            carry_headers = headers[:]

            # Build header maps (x-centers)
            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))

            # Extract grouped lines
            lines = extract_words_grouped(page)

            # Walk lines: maintain current AD, Store, Section
            current_ad: Optional[str] = None
            current_store: Optional[str] = None
            current_section: Optional[str] = None

            # Find AD by anchoring "Area Director" heading
            for idx, L in enumerate(lines):
                if L["text"].strip().lower() == "area director":
                    # Next name-like, left-aligned line
                    left_margin = min(x["x0"] for x in L["words"]) if L["words"] else L["x_min"]
                    def is_left_aligned(x): return (x - left_margin) <= 20

                    for j in range(idx+1, min(idx+6, len(lines))):
                        cand = lines[j]
                        s = cand["text"].strip()
                        if is_left_aligned(cand["x_min"]) and looks_like_name(s):
                            current_ad = s
                            break
                    # continue scan after we found the AD; but do not break the page loop here

            # If we didn't find via anchor, fallback: pick first plausible name above first store
            if not current_ad:
                first_store_idx = next((k for k, L in enumerate(lines) if STORE_LINE_RX.match(L["text"])), None)
                if first_store_idx is not None:
                    for j in range(first_store_idx-1, max(first_store_idx-6, -1), -1):
                        s = lines[j]["text"].strip()
                        if looks_like_name(s):
                            current_ad = s
                            break

            # Now iterate lines to capture Store / Section / Reasons
            for L in lines:
                txt = L["text"].strip()

                # Store line
                if STORE_LINE_RX.match(txt):
                    current_store = txt
                    current_section = None
                    continue

                # Sections
                if SECTION_TOGO.match(txt):
                    current_section = "To Go"
                    continue
                if SECTION_DELIV.match(txt):
                    current_section = "Delivery"
                    continue
                if SECTION_DINEIN.match(txt):
                    current_section = "Dine-In"  # we will skip this section
                    continue

                # Skip headings
                if txt in HEADINGS:
                    continue

                # Only proceed if we have the hierarchy in place
                if not (current_ad and current_store and current_section in {"To Go", "Delivery"}):
                    continue

                # See if this line is one of the 7 reasons (robustly)
                canon = normalize_reason(txt)
                if not canon:
                    continue  # not one of our targets

                # Map numbers on this line to the nearest header by x-position
                # Then pull the value for the selected header later.
                # To keep it general, we store ALL header->value pairs for this reason.
                # We'll collapse to one (selected period) after the parse.
                # Initialize store slot
                _ = data[current_ad][current_store][current_section]  # ensure path

                for w in L["words"]:
                    token = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", token):
                        continue
                    xmid = (w["x0"] + w["x1"]) / 2
                    nearest_header = min(header_positions.items(), key=lambda kv: abs(kv[1] - xmid))[0]
                    # stash per-header values under a synthetic key so we can choose later
                    # We'll store in a temporary nested dict: data[AD][Store][Section][canon] as dict of header->int
                    # To avoid changing outer structure, keep a special key "__all__" that holds per-header map.
                    inner = data[current_ad][current_store][current_section]
                    if "__all__" not in inner:
                        inner["__all__"] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
                    inner["__all__"][canon][nearest_header] += int(token)

    return {h: header_positions[h] for h in ordered_headers}, data, ordered_headers

# Name rules reused
STOP_TOKENS = {
    "necessary","info","information","compensation","offered","restaurant","operational","issues",
    "missing","condiments","ingredient","food","bev","beverage","order","wrong","cold","slow",
    "unfriendly","manager","did","not","attempt","resolve","issue","appearance","packaging","to",
    "go","to-go","dine-in","delivery","total","guest","ticket","incorrect","understaffed","poor",
    "quality","presentation","overcooked","burnt","undercooked","host","server","greet","portion"
}
def looks_like_name(s: str) -> bool:
    s_clean = s.strip()
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

# ───────────────── RUN PARSE ─────────────────
with st.spinner("Parsing PDF…"):
    header_x_map, raw_data, ordered_headers = parse_pdf_build_ad_store_period_map(file_bytes)

if not ordered_headers:
    st.error("No period headers (e.g., ‘P9 24’) were detected.")
    st.stop()

# ───────────────── PERIOD SELECTION ─────────────────
st.header("2) Pick the period")
sel_col = st.selectbox("Period", options=ordered_headers, index=len(ordered_headers)-1)

# ───────────────── COLLAPSE TO SELECTED PERIOD & TOTALS ─────────────────
# Build flat rows: AD, Store, Section, Reason, Value
rows = []
for ad, stores in raw_data.items():
    for store, sections in stores.items():
        for section, reason_map in sections.items():
            if section not in {"To Go", "Delivery"}:
                continue
            # per-reason values
            all_per_header = reason_map.get("__all__", {})
            for canon in CANONICAL:
                # sum all tokens mapped to the chosen header for this reason
                v = 0
                if canon in all_per_header and sel_col in all_per_header[canon]:
                    v = int(all_per_header[canon][sel_col])
                rows.append({
                    "Area Director": ad,
                    "Store": store,
                    "Section": section,
                    "Reason": canon,
                    "Value": v,
                })

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No matching To Go/Delivery reasons were found for the selected period.")
    st.stop()

# Store totals (To Go + Delivery, just the 7 reasons)
store_totals = (
    df.groupby(["Area Director","Store"], as_index=False)["Value"].sum()
    .rename(columns={"Value":"Store Total"})
)

# AD totals (sum of their stores)
ad_totals = (
    store_totals.groupby("Area Director", as_index=False)["Store Total"].sum()
    .rename(columns={"Store Total":"AD Total"})
)

# Attach totals to detail for display
df_detail = df.merge(
    store_totals, on=["Area Director","Store"], how="left"
).merge(
    ad_totals, on="Area Director", how="left"
)

# ───────────────── DISPLAY ─────────────────
st.success("✅ Parsed successfully.")

st.subheader(f"Results for period: {sel_col}")
# Show nested by AD → Store
ads = df_detail["Area Director"].dropna().unique().tolist()

for ad in ads:
    sub = df_detail[df_detail["Area Director"]==ad].copy()
    st.markdown(f"### 👤 {ad}  —  AD Total: **{int(sub['AD Total'].iloc[0])}**")
    stores = sub["Store"].dropna().unique().tolist()
    for store in stores:
        substore = sub[sub["Store"]==store].copy()
        store_total = int(substore["Store Total"].iloc[0])
        st.markdown(f"**{store}**  — Store Total: **{store_total}**")

        # pivot Section × Reason for readability (To Go / Delivery in blocks)
        pivot = (
            substore.pivot_table(index="Reason", columns="Section", values="Value", aggfunc="sum", fill_value=0)
            .reindex(CANONICAL)  # fixed row order
        )
        # Add a total column across To Go + Delivery
        pivot["Total"] = pivot.sum(axis=1)
        st.dataframe(pivot, use_container_width=True)

# ───────────────── QUICK COPY & EXPORTS ─────────────────
st.header("3) Export")
# Flat export
csv = df_detail.to_csv(index=False)
st.download_button("📥 Download detail CSV", data=csv, file_name=f"ad_store_detail_{sel_col.replace(' ','_')}.csv", mime="text/csv")

buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    df_detail.to_excel(writer, index=False, sheet_name="Detail")
    store_totals.to_excel(writer, index=False, sheet_name="Store Totals")
    ad_totals.to_excel(writer, index=False, sheet_name="AD Totals")
st.download_button("📥 Download Excel (Detail + Totals)", data=buff.getvalue(),
                   file_name=f"ad_store_{sel_col.replace(' ','_')}.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ───────────────── DEBUG ─────────────────
with st.expander("🔎 Debug tips"):
    st.markdown("""
- If a reason you expect is missing, check the exact wording in the PDF:
  - **Missing bev** often appears as **“Missing Item (Bev)”** — covered by our patterns.
  - **Missing ingredients** may appear as **“Missing Ingredient (Food)”** — covered.
  - **Packaging to-go complaint** may be **“Packaging To Go Complaint”** — covered.
- This tool ignores **Dine-In** on purpose.
- “Store Total” is **To Go + Delivery** across the seven reasons for the selected period.
- “AD Total” is the sum of that Area Director’s Store Totals.
""")
