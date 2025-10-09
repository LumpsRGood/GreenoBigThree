# Greeno Big Three v1.5.8 — baseline v1.5.7 + collapsible AD sections (expand on demand)
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
st.set_page_config(page_title="Greeno Big Three v1.5.8", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.5.8</h1>
      <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        Baseline locked (strict 7-labels + TOTAL-aware bins) with collapsible Area Director details.
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
STORE_LINE_RX  = re.compile(r"^\s*\d{3,6}\s*-\s+.*")  # "5456 - City" or "0406 - City"
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

# Exact aliases → canonical (strict)
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

def looks_like_name(s: str) -> bool:
    STOP_TOKENS = {
        "necessary","info","information","compensation","offered","restaurant","operational","issues",
        "missing","condiments","ingredient","food","bev","beverage","order","wrong","cold","slow",
        "unfriendly","manager","did","not","attempt","resolve","issue","appearance","packaging","to",
        "go","to-go","dine-in","delivery","total","guest","ticket","incorrect","understaffed","poor",
        "quality","presentation","overcooked","burnt","undercooked","host","server","greet","portion"
    }
    s_clean = s.strip()
    if s_clean.lower() == "area director":
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

# ───────────────── HEADER HELPERS ─────────────────
def find_period_headers(page) -> List[Tuple[str, float, float]]:
    """Find 'P# YY' headers; return list of (text, x_center, y_mid)."""
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
    seen = {}
    for txt, xc, ym in sorted(headers, key=lambda h: (h[2], h[1])):
        seen.setdefault(txt, (txt, xc, ym))
    return list(seen.values())

def sort_headers(headers: List[str]) -> List[str]:
    def key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999, 999)
    return sorted(headers, key=key)

def find_total_header_x(page, header_y: float) -> Optional[float]:
    """x-center of 'Total' on the header line, if present."""
    words = page.extract_words(
        x_tolerance=1.0, y_tolerance=2.0,
        keep_blank_chars=False, use_text_flow=True
    )
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        if abs(y_mid - header_y) <= 2.5 and w["text"].strip().lower() == "total":
            return (w["x0"] + w["x1"]) / 2
    return None

def build_header_bins(header_positions: Dict[str, float], total_x: Optional[float]) -> List[Tuple[str, float, float]]:
    """
    Half-interval bins for each period; last bin ends halfway to TOTAL if present.
    Returns [ (header, left_bound, right_bound) ].
    """
    def _key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999, 999)
    items = sorted(header_positions.items(), key=lambda kv: _key(kv[0]))
    headers = [h for h, _ in items]
    xs = [x for _, x in items]

    if len(xs) >= 2:
        gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
        med_gap = statistics.median(gaps)
    else:
        med_gap = 60.0

    bins = []
    for i, (h, x) in enumerate(zip(headers, xs)):
        left = (xs[i-1] + x)/2 if i > 0 else x - 0.5*med_gap
        if i < len(xs) - 1:
            right = (x + xs[i+1])/2
        else:
            right = (x + total_x)/2 if total_x is not None else x + 0.5*med_gap
        bins.append((h, left, right))
    return bins

def map_x_to_header(header_bins: List[Tuple[str, float, float]], xmid: float) -> Optional[str]:
    for h, left, right in header_bins:
        if left <= xmid < right:
            return h
    return None

# ───────────────── LINE GROUPING ─────────────────
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

def find_ad_for_store(lines: List[dict], store_idx: int, left_margin: float, back_limit: int = 12) -> Optional[str]:
    """Walk upward from the store line to find the nearest left-aligned name-like line."""
    def is_left_aligned(x): return (x - left_margin) <= 24
    for j in range(store_idx - 1, max(store_idx - back_limit, -1), -1):
        cand = lines[j]
        s = cand["text"].strip()
        if is_left_aligned(cand["x_min"]) and looks_like_name(s):
            return s
    for j in range(store_idx - back_limit - 1, -1, -1):
        cand = lines[j]
        s = cand["text"].strip()
        if looks_like_name(s):
            return s
    return None

# ───────────────── PARSER ─────────────────
def parse_pdf_build_ad_store_period_map(file_bytes: bytes):
    """
    Returns:
      header_positions: {header -> x_center}
      data: {AD: {Store: {Section: {"__all__": {Reason: {Header: int}}}}}}
      ordered_headers: [headers...]
      pairs_debug: list of (AD, Store)
    """
    header_positions: Dict[str, float] = {}
    ordered_headers: List[str] = []
    pairs_debug: List[Tuple[str, str]] = []

    data: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        carry_headers = None
        carry_total_x = None
        for page in pdf.pages:
            headers = find_period_headers(page) or carry_headers
            if not headers:
                continue
            carry_headers = headers[:]

            # header positions & order
            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)

            # TOTAL column x (if present) and bins for this page
            total_x = find_total_header_x(page, header_y) or carry_total_x
            carry_total_x = total_x
            header_bins = build_header_bins({h: header_positions[h] for h in ordered_headers}, total_x)

            # label area (everything left of the first period)
            first_period_x = min(header_positions[h] for h in ordered_headers)
            label_right_edge = first_period_x - 12  # small padding

            lines = extract_words_grouped(page)
            if not lines:
                continue

            left_margin = min(L["x_min"] for L in lines)
            current_ad: Optional[str] = None
            current_store: Optional[str] = None
            current_section: Optional[str] = None

            for idx, L in enumerate(lines):
                txt = L["text"].strip()

                # Store detection → resolve AD per store (look upward)
                if STORE_LINE_RX.match(txt):
                    ad_for_this_store = find_ad_for_store(lines, idx, left_margin)
                    if ad_for_this_store:
                        current_ad = ad_for_this_store
                    current_store = txt
                    current_section = None
                    if current_ad:
                        pairs_debug.append((current_ad, current_store))
                    continue

                # Sections
                if SECTION_TOGO.match(txt):
                    current_section = "To Go";   continue
                if SECTION_DELIV.match(txt):
                    current_section = "Delivery"; continue
                if SECTION_DINEIN.match(txt):
                    current_section = "Dine-In";  continue  # ignored later

                # Skip headings
                if txt in HEADINGS:
                    continue
                if not (current_ad and current_store and current_section in {"To Go","Delivery"}):
                    continue

                # LEFT-SIDE LABEL ONLY (strict alias match)
                label_tokens = [w["text"].strip() for w in L["words"] if w["x1"] <= label_right_edge]
                label_text = " ".join(t for t in label_tokens if t)
                canon = normalize_reason(label_text)
                if not canon:
                    continue

                # Accumulate numbers under period bins (ignore TOTAL/whitespace)
                sect = data[current_ad].setdefault(current_store, {}).setdefault(current_section, {})
                per_header = sect.setdefault("__all__", defaultdict(lambda: defaultdict(int)))
                for w in L["words"]:
                    token = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", token):
                        continue
                    if w["x0"] <= label_right_edge:
                        continue  # ignore any digits in the label area
                    xmid = (w["x0"] + w["x1"]) / 2
                    mapped = map_x_to_header(header_bins, xmid)
                    if mapped is None:
                        continue  # outside any period bin (e.g., under TOTAL)
                    if mapped in ordered_headers:
                        per_header[canon][mapped] += int(token)

    return {h: header_positions[h] for h in ordered_headers}, data, ordered_headers, pairs_debug

# ───────────────── RUN ─────────────────
with st.spinner("Roll Tide…"):
    header_x_map, raw_data, ordered_headers, pairs_debug = parse_pdf_build_ad_store_period_map(file_bytes)

if not ordered_headers:
    st.error("No period headers (like ‘P9 24’) found.")
    st.stop()

# ───────────────── PERIOD SELECTION ─────────────────
st.header("2) Pick the period")
sel_col = st.selectbox("Period", options=ordered_headers, index=len(ordered_headers)-1)

# ───────────────── BUILD RESULTS ─────────────────
rows = []
for ad, stores in raw_data.items():
    for store, sections in stores.items():
        for section, reason_map in sections.items():
            if section not in {"To Go", "Delivery"}:
                continue
            all_per_header = reason_map.get("__all__", {})
            for canon in CANONICAL:
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
    st.warning("No matching To Go/Delivery reasons found for the selected period.")
    st.stop()

# ---- Reason totals (copy-friendly) ----
def _order_series(s: pd.Series) -> pd.Series:
    return s.reindex(CANONICAL)

tot_to_go = (
    df[df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)
tot_delivery = (
    df[df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)
tot_overall = (
    df.groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)

reason_totals = pd.DataFrame({
    "To Go": _order_series(tot_to_go),
    "Delivery": _order_series(tot_delivery),
    "Total": _order_series(tot_overall),
}).fillna(0).astype(int)
reason_totals.loc["— Grand Total —"] = reason_totals.sum(numeric_only=True)

# Store & AD totals for detail views
store_totals = (
    df.groupby(["Area Director","Store"], as_index=False)["Value"].sum()
      .rename(columns={"Value":"Store Total"})
)
ad_totals = (
    store_totals.groupby("Area Director", as_index=False)["Store Total"].sum()
      .rename(columns={"Store Total":"AD Total"})
)
df_detail = df.merge(store_totals, on=["Area Director","Store"], how="left") \
              .merge(ad_totals, on="Area Director", how="left")

# ───────────────── DISPLAY (collapsible AD sections) ─────────────────
st.success("✅ Parsed with strict labels & TOTAL-aware bins.")
st.subheader(f"Results for period: {sel_col}")

col1, col2 = st.columns([1, 3])
with col1:
    expand_all = st.toggle("Expand all Area Directors", value=False, help="Show all stores & reason pivots for each AD")

# Quick AD totals glance table (optional compact summary)
with col2:
    ad_summary = ad_totals.rename(columns={"Store Total": "AD Total"}).sort_values("Area Director")
    st.dataframe(ad_summary, use_container_width=True, height=min(400, 60 + 28 * max(2, len(ad_summary))))

ads = df_detail["Area Director"].dropna().unique().tolist()
for ad in ads:
    sub = df_detail[df_detail["Area Director"]==ad].copy()
    ad_total_val = int(sub['AD Total'].iloc[0])
    with st.expander(f"👤 {ad} — AD Total: {ad_total_val}", expanded=expand_all):
        stores = sub["Store"].dropna().unique().tolist()
        for store in stores:
            substore = sub[sub["Store"]==store].copy()
            store_total = int(substore["Store Total"].iloc[0])
            st.markdown(f"**{store}**  — Store Total: **{store_total}**")
            pivot = (
                substore.pivot_table(index="Reason", columns="Section", values="Value", aggfunc="sum", fill_value=0)
                        .reindex(CANONICAL)
            )
            pivot["Total"] = pivot.sum(axis=1)
            st.dataframe(pivot, use_container_width=True)

# ───────────────── REASON TOTALS (copy-friendly) ─────────────────
st.header("4) Reason totals (selected period)")
st.caption("Use this table to copy values into your spreadsheet (e.g., your new P9 column).")
st.dataframe(reason_totals, use_container_width=True)

# ───────────────── DRILL-DOWN / EVIDENCE ─────────────────
with st.expander("🔎 Drill-down: show exact values per header for any AD & Store"):
    if raw_data:
        ad_pick = st.selectbox("Area Director", sorted(raw_data.keys()))
        store_pick = st.selectbox("Store", sorted(raw_data[ad_pick].keys()))
        sec_pick = st.radio("Section", ["To Go", "Delivery"], horizontal=True)

        per = raw_data[ad_pick][store_pick].get(sec_pick, {}).get("__all__", {})
        mat = []
        for r in CANONICAL:
            row = {"Reason": r}
            for h in ordered_headers:
                row[h] = int(per.get(r, {}).get(h, 0))
            mat.append(row)
        df_mat = pd.DataFrame(mat).set_index("Reason")
        st.dataframe(df_mat, use_container_width=True)
        st.caption("Verifies the exact numbers parsed under each period header for your chosen store.")
    else:
        st.caption("No data parsed.")

# ───────────────── EXPORTS ─────────────────
st.header("5) Export results")
csv = df_detail.to_csv(index=False)
st.download_button("📥 Download detail CSV", data=csv, file_name=f"ad_store_detail_{sel_col.replace(' ','_')}.csv", mime="text/csv")

# Excel with all sheets including Reason Totals
buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    df_detail.to_excel(writer, index=False, sheet_name="Detail")
    store_totals.to_excel(writer, index=False, sheet_name="Store Totals")
    ad_totals.to_excel(writer, index=False, sheet_name="AD Totals")
    reason_totals.to_excel(writer, sheet_name="Reason Totals")
st.download_button(
    "📥 Download Excel (Detail + Totals + Reason Totals)",
    data=buff.getvalue(),
    file_name=f"ad_store_{sel_col.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# CSV just for Reason Totals (quick paste into your sheet)
rt_csv = reason_totals.to_csv()
st.download_button(
    "📥 Download reason totals CSV (selected period)",
    data=rt_csv,
    file_name=f"reason_totals_{sel_col.replace(' ','_')}.csv",
    mime="text/csv"
)

# ───────────────── DEBUG / VERIFICATION ─────────────────
with st.expander("🧪 Debug: AD ↔ Store pairs detected this run"):
    if pairs_debug:
        st.dataframe(pd.DataFrame(pairs_debug, columns=["Area Director","Store"]))
    else:
        st.caption("No pairs captured (unexpected).")
