# Greeno Big Three v1.6.9 — strict parser (TOTAL-aware bins, left-label) +
# collapsible ADs + reason totals (Missing + Attitude + Other) + Period Change Summary (text) +
# Historical context (lower = better) + Single Excel export
# Categories: To-go Missing Complaints (To-Go/Delivery) + Attitude (all segments) + Other (all segments)
import io, os, re, base64, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st
# Greeno Big Three v1.7.0 — strict parser (TOTAL-aware bins, left-label) +
# Quick Glance scoreboard + reason totals (Missing + Attitude + Other) +
# Period Change Summary (text, dynamic height) +
# Historical context (lower = better) with highlighted Current cell +
# Collapsible ADs + High-contrast tables + Single Excel export
# Categories: To-go Missing Complaints (To-Go/Delivery) + Attitude (all segments) + Other (all segments)
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
                return ["background-color: #FFE39B; color: #111; font-weight: 700;"] * len(row)
            return [""] * len(row)
        sty = sty.apply(highlight_total, axis=1)

    # Make index header text dark too
    sty = sty.set_table_styles(
        [{"selector": "th.row_heading, th.blank", "props": [("color", "#111"), ("border-color", "#CCD3DB")]}]
    )
    return sty

# ───────────────── HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.7.0", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.7.0</h1>
      <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ───────────────── SIDEBAR UPLOAD ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose the PDF report", type=["pdf"])
    st.caption("Parses labels from the left side. Missing = To-Go & Delivery; Attitude/Other = all segments.")

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

# ───────────────── CONSTANTS ─────────────────
HEADINGS = {"Area Director", "Restaurant", "Order Visit Type", "Reason for Contact"}
STORE_LINE_RX  = re.compile(r"^\s*\d{3,6}\s*-\s+.*")
SECTION_TOGO   = re.compile(r"^\s*(To[\s-]?Go|To-go)\s*$", re.IGNORECASE)
SECTION_DELIV  = re.compile(r"^\s*Delivery\s*$", re.IGNORECASE)
SECTION_DINEIN = re.compile(r"^\s*Dine[\s-]?In\s*$", re.IGNORECASE)
HEADER_RX      = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

# Canonical reasons — To-go Missing Complaints (7)
MISSING_REASONS = [
    "Missing food",
    "Order wrong",
    "Missing condiments",
    "Out of menu item",
    "Missing bev",
    "Missing ingredients",
    "Packaging to-go complaint",
]

# Canonical reasons — Attitude (7)
ATTITUDE_REASONS = [
    "Unprofessional/Unfriendly",
    "Manager directly involved",
    "Manager not available",
    "Manager did not visit",
    "Negative mgr-employee exchange",
    "Manager did not follow up",
    "Argued with guest",
]

# Canonical reasons — Other (7, all segments)
OTHER_REASONS = [
    "Long hold/no answer",
    "No/insufficient compensation offered",
    "Did not attempt to resolve",
    "Guest left without ordering",
    "Unknowledgeable",
    "Did not open on time",
    "No/poor apology",
]

ALL_CANONICAL = MISSING_REASONS + ATTITUDE_REASONS + OTHER_REASONS

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

# Exact aliases → canonical (strict)
ALIASES_MISSING = {
    _norm("Missing Item (Food)"):        "Missing food",
    _norm("Order Wrong"):                "Order wrong",
    _norm("Missing Condiments"):         "Missing condiments",
    _norm("Out Of Menu Item"):           "Out of menu item",
    _norm("Missing Item (Bev)"):         "Missing bev",
    _norm("Missing Ingredient (Food)"):  "Missing ingredients",
    _norm("Packaging To Go Complaint"):  "Packaging to-go complaint",
}
ALIASES_ATTITUDE = {
    _norm("Unprofessional Behavior"):                 "Unprofessional/Unfriendly",
    _norm("Unfriendly Attitude"):                     "Unprofessional/Unfriendly",
    _norm("Manager Directly Involved In Complaint"):  "Manager directly involved",
    _norm("Management Not Available"):                "Manager not available",
    _norm("Manager Did Not Visit"):                   "Manager did not visit",
    _norm("Negative Manager-Employee Interaction"):   "Negative mgr-employee exchange",
    _norm("Manager Did Not Follow Up"):               "Manager did not follow up",
    _norm("Argued With Guest"):                       "Argued with guest",
}
ALIASES_OTHER = {
    _norm("Long Hold/No Answer/Hung Up"):                            "Long hold/no answer",
    _norm("No/Unsatisfactory Compensation Offered By Restaurant"):   "No/insufficient compensation offered",
    _norm("Did Not Attempt To Resolve Issue"):                       "Did not attempt to resolve",
    _norm("Guest Left Without Dining or Ordering"):                  "Guest left without ordering",
    _norm("Unknowledgeable"):                                        "Unknowledgeable",
    _norm("Didn't Open/close On Time"):                              "Did not open on time",
    _norm("No/Poor Apology"):                                        "No/poor apology",
}

REASON_ALIASES_NORM = {**ALIASES_MISSING, **ALIASES_ATTITUDE, **ALIASES_OTHER}}

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
    words = page.extract_words(x_tolerance=1.0, y_tolerance=2.0, keep_blank_chars=False, use_text_flow=True)
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        if abs(y_mid - header_y) <= 2.5 and w["text"].strip().lower() == "total":
            return (w["x0"] + w["x1"]) / 2
    return None

def build_header_bins(header_positions: Dict[str, float], total_x: Optional[float]) -> List[Tuple[str, float, float]]:
    def _key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999, 999)
    items = sorted(header_positions.items(), key=lambda kv: _key(kv[0]))
    headers = [h for h, _ in items]
    xs = [x for _, x in items]
    med_gap = statistics.median([xs[i+1]-xs[i] for i in range(len(xs)-1)]) if len(xs) >= 2 else 60.0
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

            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)

            total_x = find_total_header_x(page, header_y) or carry_total_x
            carry_total_x = total_x
            header_bins = build_header_bins({h: header_positions[h] for h in ordered_headers}, total_x)

            first_period_x = min(header_positions[h] for h in ordered_headers)
            label_right_edge = first_period_x - 12  # keep labels strictly left of first period

            lines = extract_words_grouped(page)
            if not lines:
                continue

            left_margin = min(L["x_min"] for L in lines)
            current_ad: Optional[str] = None
            current_store: Optional[str] = None
            current_section: Optional[str] = None

            for idx, L in enumerate(lines):
                txt = L["text"].strip()

                if STORE_LINE_RX.match(txt):
                    ad_for_this_store = find_ad_for_store(lines, idx, left_margin)
                    if ad_for_this_store:
                        current_ad = ad_for_this_store
                    current_store = txt
                    current_section = None
                    if current_ad:
                        pairs_debug.append((current_ad, current_store))
                    continue

                if SECTION_TOGO.match(txt):
                    current_section = "To Go";   continue
                if SECTION_DELIV.match(txt):
                    current_section = "Delivery"; continue
                if SECTION_DINEIN.match(txt):
                    current_section = "Dine-In";  continue

                if txt in HEADINGS:
                    continue
                if not (current_ad and current_store and current_section in {"To Go","Delivery","Dine-In"}):
                    continue

                # LEFT LABEL ONLY (strict)
                label_tokens = [w["text"].strip() for w in L["words"] if w["x1"] <= label_right_edge]
                label_text = " ".join(t for t in label_tokens if t)
                canon = normalize_reason(label_text)
                if not canon:
                    continue

                sect = data[current_ad].setdefault(current_store, {}).setdefault(current_section, {})
                per_header = sect.setdefault("__all__", defaultdict(lambda: defaultdict(int)))
                for w in L["words"]:
                    token = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", token):
                        continue
                    if w["x0"] <= label_right_edge:
                        continue
                    xmid = (w["x0"] + w["x1"]) / 2
                    mapped = map_x_to_header(header_bins, xmid)
                    if mapped is None:
                        continue
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

# ───────────────── QUICK GLANCE SCOREBOARD ─────────────────
st.markdown("### Quick glance")

def _total_for(period_label: str, reasons: list[str], sections: set[str]) -> int:
    if not period_label:
        return 0
    total = 0
    for ad, stores in raw_data.items():
        for store, sects in stores.items():
            for sec_name, reason_map in sects.items():
                if sec_name not in sections:
                    continue
                per = reason_map.get("__all__", {})
                for r in reasons:
                    total += int(per.get(r, {}).get(period_label, 0))
    return int(total)

# Prior period (if any)
try:
    cur_idx = ordered_headers.index(sel_col)
    prior_label = ordered_headers[cur_idx - 1] if cur_idx > 0 else None
except ValueError:
    prior_label = None

# Category scopes
missing_sections = {"To Go", "Delivery"}
att_sections     = {"To Go", "Delivery", "Dine-In"}
other_sections   = {"To Go", "Delivery", "Dine-In"}

# Totals (current and prior)
tot_missing_cur = _total_for(sel_col,      MISSING_REASONS,  missing_sections)
tot_missing_prv = _total_for(prior_label,  MISSING_REASONS,  missing_sections)
tot_att_cur     = _total_for(sel_col,      ATTITUDE_REASONS, att_sections)
tot_att_prv     = _total_for(prior_label,  ATTITUDE_REASONS, att_sections)
tot_other_cur   = _total_for(sel_col,      OTHER_REASONS,    other_sections)
tot_other_prv   = _total_for(prior_label,  OTHER_REASONS,    other_sections)

overall_cur = tot_missing_cur + tot_att_cur + tot_other_cur
overall_prv = (tot_missing_prv + tot_att_prv + tot_other_prv) if prior_label else 0

def fmt_delta(a: int, b: int, has_prior: bool) -> str:
    if not has_prior:
        return "n/a"
    diff = a - b
    return f"{diff:+d}"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Overall (all categories)", overall_cur, fmt_delta(overall_cur, overall_prv, prior_label is not None))
with c2:
    st.metric("To-go Missing Complaints", tot_missing_cur, fmt_delta(tot_missing_cur, tot_missing_prv, prior_label is not None))
with c3:
    st.metric("Attitude",                tot_att_cur,     fmt_delta(tot_att_cur,     tot_att_prv,     prior_label is not None))
with c4:
    st.metric("Other",                   tot_other_cur,   fmt_delta(tot_other_cur,   tot_other_prv,   prior_label is not None))

# ───────────────── BUILD RESULTS ─────────────────
rows = []
for ad, stores in raw_data.items():
    for store, sections in stores.items():
        for section, reason_map in sections.items():
            if section not in {"To Go", "Delivery", "Dine-In"}:
                continue
            all_per_header = reason_map.get("__all__", {})
            for canon in ALL_CANONICAL:
                v = int(all_per_header.get(canon, {}).get(sel_col, 0))
                rows.append({
                    "Area Director": ad,
                    "Store": store,
                    "Section": section,
                    "Reason": canon,
                    "Value": v,
                })

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No matching reasons found for the selected period.")
    st.stop()

# ───────────────── CATEGORY MAPPING ─────────────────
CATEGORY_TOGO_MISSING = "To-go Missing Complaints"
CATEGORY_ATTITUDE     = "Attitude"
CATEGORY_OTHER        = "Other"

CATEGORY_MAP = {r: CATEGORY_TOGO_MISSING for r in MISSING_REASONS}
CATEGORY_MAP.update({r: CATEGORY_ATTITUDE for r in ATTITUDE_REASONS})
CATEGORY_MAP.update({r: CATEGORY_OTHER for r in OTHER_REASONS})
df["Category"] = df["Reason"].map(CATEGORY_MAP).fillna("Unassigned")

# ───────────────── DISPLAY (collapsible AD sections) ─────────────────
st.success("✅ Parsed with strict labels & TOTAL-aware bins.")
st.subheader(f"Results for period: {sel_col}")

col1, col2 = st.columns([1, 3])
with col1:
    expand_all = st.toggle("Expand all Area Directors", value=False, help="Show all stores & reason pivots for each AD")

with col2:
    ad_totals = (
        df.groupby(["Area Director","Store"], as_index=False)["Value"].sum()
          .groupby("Area Director", as_index=False)["Value"].sum()
          .rename(columns={"Value":"AD Total"})
    )
    st.dataframe(style_table(ad_totals, highlight_grand_total=False), use_container_width=True,
                 height=min(400, 60 + 28 * max(2, len(ad_totals))))

# per-store + per-AD totals (all rows)
store_totals = (
    df.groupby(["Area Director","Store"], as_index=False)["Value"].sum()
      .rename(columns={"Value":"Store Total"})
)
df_detail = df.merge(store_totals, on=["Area Director","Store"], how="left") \
              .merge(ad_totals, on="Area Director", how="left")

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
            show_reasons = MISSING_REASONS + ATTITUDE_REASONS + OTHER_REASONS
            pivot = (
                substore[substore["Reason"].isin(show_reasons)]
                .pivot_table(index="Reason", columns="Section", values="Value", aggfunc="sum", fill_value=0)
                .reindex(show_reasons)
            )
            pivot["Total"] = pivot.sum(axis=1)
            st.dataframe(style_table(pivot, highlight_grand_total=False), use_container_width=True)

# ───────────────── REASON TOTALS — Missing ─────────────────
st.header("4) Reason totals — To-go Missing Complaints (selected period)")
st.caption("To-Go and Delivery only, for the seven Missing reasons.")

missing_df = df[df["Reason"].isin(MISSING_REASONS) & df["Section"].isin({"To Go","Delivery"})]

def _order_series_missing(s: pd.Series) -> pd.Series:
    return s.reindex(MISSING_REASONS)

tot_to_go = (
    missing_df[missing_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
tot_delivery = (
    missing_df[missing_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
tot_overall = (
    missing_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)
)

reason_totals_missing = pd.DataFrame({
    "To Go": _order_series_missing(tot_to_go),
    "Delivery": _order_series_missing(tot_delivery),
    "Total": _order_series_missing(tot_overall),
}).fillna(0).astype(int)
reason_totals_missing.loc["— Grand Total —"] = reason_totals_missing.sum(numeric_only=True)

st.dataframe(style_table(reason_totals_missing), use_container_width=True)

# ───────────────── REASON TOTALS — Attitude ─────────────────
st.header("4b) Reason totals — Attitude (selected period)")
st.caption("All segments (Dine-In, To Go, Delivery) for the seven Attitude reasons.")

att_df = df[df["Reason"].isin(ATTITUDE_REASONS)]

def _order_series_att(s: pd.Series) -> pd.Series:
    return s.reindex(ATTITUDE_REASONS)

att_dinein = (
    att_df[att_df["Section"] == "Dine-In"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
att_togo = (
    att_df[att_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
att_delivery = (
    att_df[att_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
att_total = (
    att_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)
)

reason_totals_attitude = pd.DataFrame({
    "Dine-In": _order_series_att(att_dinein),
    "To Go": _order_series_att(att_togo),
    "Delivery": _order_series_att(att_delivery),
    "Total": _order_series_att(att_total),
}).fillna(0).astype(int)
reason_totals_attitude.loc["— Grand Total —"] = reason_totals_attitude.sum(numeric_only=True)

st.dataframe(style_table(reason_totals_attitude), use_container_width=True)

# ───────────────── REASON TOTALS — Other ─────────────────
st.header("4c) Reason totals — Other (selected period)")
st.caption("All segments (Dine-In, To Go, Delivery) for the seven Other reasons.")

oth_df = df[df["Reason"].isin(OTHER_REASONS)]

def _order_series_other(s: pd.Series) -> pd.Series:
    return s.reindex(OTHER_REASONS)

oth_dinein = (
    oth_df[oth_df["Section"] == "Dine-In"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
oth_togo = (
    oth_df[oth_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
oth_delivery = (
    oth_df[oth_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
oth_total = (
    oth_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)
)

reason_totals_other = pd.DataFrame({
    "Dine-In": _order_series_other(oth_dinein),
    "To Go": _order_series_other(oth_togo),
    "Delivery": _order_series_other(oth_delivery),
    "Total": _order_series_other(oth_total),
}).fillna(0).astype(int)
reason_totals_other.loc["— Grand Total —"] = reason_totals_other.sum(numeric_only=True)

st.dataframe(style_table(reason_totals_other), use_container_width=True)

# ───────────────── CATEGORY SUMMARIES ─────────────────
def category_summary_block(number_label: str, category_name: str, allowed_sections: set):
    st.header(f"{number_label}) Category summary — {category_name}")
    subset = df[(df["Category"] == category_name) & (df["Section"].isin(allowed_sections))]
    if subset.empty:
        st.info(f"No rows currently mapped to “{category_name}”.")
        return None, None, 0
    cat_store_totals = (
        subset.groupby(["Area Director", "Store"], as_index=False)["Value"]
              .sum().rename(columns={"Value": "Category Total"})
    )
    cat_ad_totals = (
        cat_store_totals.groupby("Area Director", as_index=False)["Category Total"]
                        .sum().rename(columns={"Category Total": "AD Category Total"})
    )
    cat_grand_total = int(cat_store_totals["Category Total"].sum())
    colA, colB = st.columns([1, 3])
    with colA:
        st.metric("Grand Total (Category)", cat_grand_total)
    with colB:
        st.dataframe(style_table(cat_ad_totals, highlight_grand_total=False),
                     use_container_width=True,
                     height=min(400, 60 + 28 * max(2, len(cat_ad_totals))))
    st.subheader("Per-Store Category Totals")
    st.caption(f"Each store’s total for “{category_name}” in the selected period.")
    st.dataframe(style_table(cat_store_totals, highlight_grand_total=False), use_container_width=True)
    return cat_ad_totals, cat_store_totals, cat_grand_total

# 5a — To-go Missing Complaints (To-Go + Delivery only)
tgc_ad_totals, tgc_store_totals, tgc_grand = category_summary_block("5a", "To-go Missing Complaints", {"To Go","Delivery"})
# 5b — Attitude (all segments)
att_ad_totals, att_store_totals, att_grand = category_summary_block("5b", "Attitude", {"To Go","Delivery","Dine-In"})
# 5c — Other (all segments)
oth_ad_totals, oth_store_totals, oth_grand = category_summary_block("5c", "Other", {"To Go","Delivery","Dine-In"})

# ───────────────── 6) PERIOD CHANGE SUMMARY (vs previous) ─────────────────
st.header("6) Period change summary (vs previous period)")
if not prior_label:
    st.info("No earlier period available to compare against.")
else:
    def totals_by_reason_for(period_label: str, reasons: list[str], allowed_sections: set[str]) -> pd.Series:
        sums = {r: 0 for r in reasons}
        for ad, stores in raw_data.items():
            for store, sections in stores.items():
                for section, reason_map in sections.items():
                    if section not in allowed_sections:
                        continue
                    per = reason_map.get("__all__", {})
                    for r in reasons:
                        sums[r] += int(per.get(r, {}).get(period_label, 0))
        return pd.Series(sums).astype(int)

    missing_sections = {"To Go", "Delivery"}
    attitude_sections = {"To Go", "Delivery", "Dine-In"}
    other_sections    = {"To Go", "Delivery", "Dine-In"}

    cur_missing = totals_by_reason_for(sel_col,    MISSING_REASONS,  missing_sections)
    prv_missing = totals_by_reason_for(prior_label,MISSING_REASONS,  missing_sections)
    cur_att     = totals_by_reason_for(sel_col,    ATTITUDE_REASONS, attitude_sections)
    prv_att     = totals_by_reason_for(prior_label,ATTITUDE_REASONS, attitude_sections)
    cur_other   = totals_by_reason_for(sel_col,    OTHER_REASONS,    other_sections)
    prv_other   = totals_by_reason_for(prior_label,OTHER_REASONS,    other_sections)

    delta_missing = (cur_missing - prv_missing).astype(int)
    delta_att     = (cur_att - prv_att).astype(int)
    delta_other   = (cur_other - prv_other).astype(int)

    total_missing_cur = int(cur_missing.sum()); total_missing_prv = int(prv_missing.sum())
    total_att_cur     = int(cur_att.sum());     total_att_prv     = int(prv_att.sum())
    total_other_cur   = int(cur_other.sum());   total_other_prv   = int(prv_other.sum())

    def fmt_delta(n: int) -> str:
        return f"{n:+d}"

    lines = []
    lines.append(f"Selected period: {sel_col}   •   Prior: {prior_label}")
    lines.append("")
    lines.append("To-go Missing Complaints (To-Go + Delivery)")
    lines.append(f"- Overall: {total_missing_cur} ({fmt_delta(total_missing_cur - total_missing_prv)} vs prior)")
    any_change_missing = False
    for r in MISSING_REASONS:
        d = int(delta_missing.get(r, 0))
        if d != 0:
            lines.append(f"  • {r}: {int(cur_missing[r])} ({fmt_delta(d)})")
            any_change_missing = True
    if not any_change_missing:
        lines.append("  • No change by reason.")
    lines.append("")
    lines.append("Attitude (All segments)")
    lines.append(f"- Overall: {total_att_cur} ({fmt_delta(total_att_cur - total_att_prv)} vs prior)")
    any_change_att = False
    for r in ATTITUDE_REASONS:
        d = int(delta_att.get(r, 0))
        if d != 0:
            lines.append(f"  • {r}: {int(cur_att[r])} ({fmt_delta(d)})")
            any_change_att = True
    if not any_change_att:
        lines.append("  • No change by reason.")
    lines.append("")
    lines.append("Other (All segments)")
    lines.append(f"- Overall: {total_other_cur} ({fmt_delta(total_other_cur - total_other_prv)} vs prior)")
    any_change_other = False
    for r in OTHER_REASONS:
        d = int(delta_other.get(r, 0))
        if d != 0:
            lines.append(f"  • {r}: {int(cur_other[r])} ({fmt_delta(d)})")
            any_change_other = True
    if not any_change_other:
        lines.append("  • No change by reason.")

    summary_text = "\n".join(lines)
    num_lines = summary_text.count("\n") + 2
    dyn_height = max(160, min(700, 24 * num_lines))
    st.text_area("Copy to clipboard", summary_text, height=dyn_height)
    st.download_button(
        "📥 Download summary as .txt",
        data=summary_text.encode("utf-8"),
        file_name=f"period_change_summary_{sel_col.replace(' ','_')}.txt",
        mime="text/plain",
    )

# ───────────────── 7) Historical context — highs/lows vs all periods ─────────────────
st.header("7) Historical context — highs/lows vs all periods (lower = better)")

def build_reason_period_matrix(reasons: list[str], allowed_sections: set[str]) -> dict[str, dict[str, int]]:
    """
    Returns: {period -> {reason -> total}} across all ADs/stores for allowed sections.
    """
    mat = {p: {r: 0 for r in reasons} for p in ordered_headers}
    for ad, stores in raw_data.items():
        for store, sections_map in stores.items():
            for sec, reason_map in sections_map.items():
                if sec not in allowed_sections:
                    continue
                per = reason_map.get("__all__", {})
                for r in reasons:
                    pr = per.get(r, {})
                    for p in ordered_headers:
                        mat[p][r] += int(pr.get(p, 0))
    return mat

def build_highlow_tables(reasons: list[str], allowed_sections: set[str], title: str):
    st.subheader(title)

    mat = build_reason_period_matrix(reasons, allowed_sections)

    rows = []
    num_periods = len(ordered_headers)
    best_vals = {}
    worst_vals = {}
    for r in reasons:
        series = [(p, mat[p][r]) for p in ordered_headers]
        # LOWER = BETTER
        best_period, best_val   = min(series, key=lambda kv: kv[1])  # lowest value
        worst_period, worst_val = max(series, key=lambda kv: kv[1])  # highest value
        best_vals[r] = best_val
        worst_vals[r] = worst_val
        # Rank 1 = lowest value
        sorted_asc = sorted(series, key=lambda kv: kv[1])
        rank = next(i+1 for i,(p,v) in enumerate(sorted_asc) if p == sel_col)
        cur_val = mat[sel_col][r]
        rows.append({
            "Reason": r,
            "Current": cur_val,
            "Rank": f"{rank}/{num_periods}",
            "Best (Period)": f"{best_val} @ {best_period}",
            "Worst (Period)": f"{worst_val} @ {worst_period}",
        })

    df_reasons = pd.DataFrame(rows).set_index("Reason")

    # Style: highlight Current cell if it equals best (green) or worst (red)
    def highlight_current(col: pd.Series):
        styles = []
        for reason, val in col.items():
            if val == best_vals.get(reason, None):
                styles.append("background-color: #B7F0B1; color:#111; font-weight:600;")
            elif val == worst_vals.get(reason, None):
                styles.append("background-color: #F7B0A5; color:#111; font-weight:600;")
            else:
                styles.append("")
        return styles

    # Category totals across periods (lower = better)
    totals_by_period = {p: sum(mat[p][r] for r in reasons) for p in ordered_headers}
    best_p, best_total   = min(totals_by_period.items(), key=lambda kv: kv[1])
    worst_p, worst_total = max(totals_by_period.items(), key=lambda kv: kv[1])
    current_total = totals_by_period.get(sel_col, 0)
    rank_list = sorted(totals_by_period.items(), key=lambda kv: kv[1])  # ascending
    current_rank_idx = next(i+1 for i,(p,v) in enumerate(rank_list) if p == sel_col)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Current total (lower is better)", current_total)
    with colB:
        st.metric("Best total (lowest)", f"{best_total}", help=f"Period: {best_p}")
    with colC:
        st.metric("Worst total (highest)", f"{worst_total}", help=f"Period: {worst_p}")
    st.caption(f"Current period rank: {current_rank_idx}/{num_periods} (1 = lowest/best)")

    sty = style_table(df_reasons, highlight_grand_total=False).apply(highlight_current, subset=["Current"])
    st.dataframe(sty, use_container_width=True)

# Missing → To-Go & Delivery only
build_highlow_tables(MISSING_REASONS, {"To Go","Delivery"}, "7a) To-go Missing Complaints (To-Go + Delivery) — highs/lows")

# Attitude → All segments
build_highlow_tables(ATTITUDE_REASONS, {"Dine-In","To Go","Delivery"}, "7b) Attitude (All segments) — highs/lows")

# Other → All segments
build_highlow_tables(OTHER_REASONS, {"Dine-In","To Go","Delivery"}, "7c) Other (All segments) — highs/lows")

# ───────────────── 8) EXPORT — Excel only (All Sheets) ─────────────────
st.header("8) Export results")

buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    # Detail + rollups
    df_detail.to_excel(writer, index=False, sheet_name="Detail")
    ad_totals.to_excel(writer, index=False, sheet_name="AD Totals")
    store_totals.to_excel(writer, index=False, sheet_name="Store Totals")
    # Reason totals
    reason_totals_missing.to_excel(writer, sheet_name="Reason Totals (Missing)")
    reason_totals_attitude.to_excel(writer, sheet_name="Reason Totals (Attitude)")
    reason_totals_other.to_excel(writer, sheet_name="Reason Totals (Other)")
    # Category sheets
    (tgc_ad_totals if tgc_ad_totals is not None else pd.DataFrame(columns=["Area Director","AD Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-ToGoMissing AD Totals")
    (tgc_store_totals if tgc_store_totals is not None else pd.DataFrame(columns=["Area Director","Store","Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-ToGoMissing Store")
    (att_ad_totals if att_ad_totals is not None else pd.DataFrame(columns=["Area Director","AD Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Attitude AD Totals")
    (att_store_totals if att_store_totals is not None else pd.DataFrame(columns=["Area Director","Store","Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Attitude Store")
    (oth_ad_totals if oth_ad_totals is not None else pd.DataFrame(columns=["Area Director","AD Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Other AD Totals")
    (oth_store_totals if oth_store_totals is not None else pd.DataFrame(columns=["Area Director","Store","Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Other Store")

st.download_button(
    "📥 Download Excel (All Sheets)",
    data=buff.getvalue(),
    file_name=f"ad_store_{sel_col.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ───────────────── DEBUG / VERIFICATION ─────────────────
with st.expander("🧪 Debug: AD ↔ Store pairs detected this run"):
    if pairs_debug:
        st.dataframe(style_table(pd.DataFrame(pairs_debug, columns=["Area Director","Store"]), highlight_grand_total=False),
                     use_container_width=True)
    else:
        st.caption("No pairs captured (unexpected).")
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
                return ["background-color: #FFE39B; color: #111; font-weight: 700;"] * len(row)
            return [""] * len(row)
        sty = sty.apply(highlight_total, axis=1)

    # Make index header text dark too
    sty = sty.set_table_styles(
        [{"selector": "th.row_heading, th.blank", "props": [("color", "#111"), ("border-color", "#CCD3DB")]}]
    )
    return sty

# ───────────────── HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.6.9", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.6.9</h1>
      <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        Strict parsing · Collapsible ADs · Reason totals (Missing + Attitude + Other) · Period change summary · Historical highs/lows (lower = better) · Excel (All Sheets)
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
    st.caption("Parses labels from the left side. Missing = To-Go & Delivery; Attitude/Other = all segments.")

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

# ───────────────── CONSTANTS ─────────────────
HEADINGS = {"Area Director", "Restaurant", "Order Visit Type", "Reason for Contact"}
STORE_LINE_RX  = re.compile(r"^\s*\d{3,6}\s*-\s+.*")
SECTION_TOGO   = re.compile(r"^\s*(To[\s-]?Go|To-go)\s*$", re.IGNORECASE)
SECTION_DELIV  = re.compile(r"^\s*Delivery\s*$", re.IGNORECASE)
SECTION_DINEIN = re.compile(r"^\s*Dine[\s-]?In\s*$", re.IGNORECASE)
HEADER_RX      = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

# Canonical reasons — To-go Missing Complaints (7)
MISSING_REASONS = [
    "Missing food",
    "Order wrong",
    "Missing condiments",
    "Out of menu item",
    "Missing bev",
    "Missing ingredients",
    "Packaging to-go complaint",
]

# Canonical reasons — Attitude (7)
ATTITUDE_REASONS = [
    "Unprofessional/Unfriendly",
    "Manager directly involved",
    "Manager not available",
    "Manager did not visit",
    "Negative mgr-employee exchange",
    "Manager did not follow up",
    "Argued with guest",
]

# Canonical reasons — Other (7, all segments)
OTHER_REASONS = [
    "Long hold/no answer",
    "No/insufficient compensation offered",
    "Did not attempt to resolve",
    "Guest left without ordering",
    "Unknowledgeable",
    "Did not open on time",
    "No/poor apology",
]

ALL_CANONICAL = MISSING_REASONS + ATTITUDE_REASONS + OTHER_REASONS

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

# Exact aliases → canonical (strict)
ALIASES_MISSING = {
    _norm("Missing Item (Food)"):        "Missing food",
    _norm("Order Wrong"):                "Order wrong",
    _norm("Missing Condiments"):         "Missing condiments",
    _norm("Out Of Menu Item"):           "Out of menu item",
    _norm("Missing Item (Bev)"):         "Missing bev",
    _norm("Missing Ingredient (Food)"):  "Missing ingredients",
    _norm("Packaging To Go Complaint"):  "Packaging to-go complaint",
}
ALIASES_ATTITUDE = {
    _norm("Unprofessional Behavior"):                 "Unprofessional/Unfriendly",
    _norm("Unfriendly Attitude"):                     "Unprofessional/Unfriendly",
    _norm("Manager Directly Involved In Complaint"):  "Manager directly involved",
    _norm("Management Not Available"):                "Manager not available",
    _norm("Manager Did Not Visit"):                   "Manager did not visit",
    _norm("Negative Manager-Employee Interaction"):   "Negative mgr-employee exchange",
    _norm("Manager Did Not Follow Up"):               "Manager did not follow up",
    _norm("Argued With Guest"):                       "Argued with guest",
}
ALIASES_OTHER = {
    _norm("Long Hold/No Answer/Hung Up"):                            "Long hold/no answer",
    _norm("No/Unsatisfactory Compensation Offered By Restaurant"):   "No/insufficient compensation offered",
    _norm("Did Not Attempt To Resolve Issue"):                       "Did not attempt to resolve",
    _norm("Guest Left Without Dining or Ordering"):                  "Guest left without ordering",
    _norm("Unknowledgeable"):                                        "Unknowledgeable",
    _norm("Didn't Open/close On Time"):                              "Did not open on time",
    _norm("No/Poor Apology"):                                        "No/poor apology",
}

REASON_ALIASES_NORM = {**ALIASES_MISSING, **ALIASES_ATTITUDE, **ALIASES_OTHER}

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
    words = page.extract_words(x_tolerance=1.0, y_tolerance=2.0, keep_blank_chars=False, use_text_flow=True)
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        if abs(y_mid - header_y) <= 2.5 and w["text"].strip().lower() == "total":
            return (w["x0"] + w["x1"]) / 2
    return None

def build_header_bins(header_positions: Dict[str, float], total_x: Optional[float]) -> List[Tuple[str, float, float]]:
    def _key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999, 999)
    items = sorted(header_positions.items(), key=lambda kv: _key(kv[0]))
    headers = [h for h, _ in items]
    xs = [x for _, x in items]
    med_gap = statistics.median([xs[i+1]-xs[i] for i in range(len(xs)-1)]) if len(xs) >= 2 else 60.0
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

            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)

            total_x = find_total_header_x(page, header_y) or carry_total_x
            carry_total_x = total_x
            header_bins = build_header_bins({h: header_positions[h] for h in ordered_headers}, total_x)

            first_period_x = min(header_positions[h] for h in ordered_headers)
            label_right_edge = first_period_x - 12  # keep labels strictly left of first period

            lines = extract_words_grouped(page)
            if not lines:
                continue

            left_margin = min(L["x_min"] for L in lines)
            current_ad: Optional[str] = None
            current_store: Optional[str] = None
            current_section: Optional[str] = None

            for idx, L in enumerate(lines):
                txt = L["text"].strip()

                if STORE_LINE_RX.match(txt):
                    ad_for_this_store = find_ad_for_store(lines, idx, left_margin)
                    if ad_for_this_store:
                        current_ad = ad_for_this_store
                    current_store = txt
                    current_section = None
                    if current_ad:
                        pairs_debug.append((current_ad, current_store))
                    continue

                if SECTION_TOGO.match(txt):
                    current_section = "To Go";   continue
                if SECTION_DELIV.match(txt):
                    current_section = "Delivery"; continue
                if SECTION_DINEIN.match(txt):
                    current_section = "Dine-In";  continue

                if txt in HEADINGS:
                    continue
                if not (current_ad and current_store and current_section in {"To Go","Delivery","Dine-In"}):
                    continue

                # LEFT LABEL ONLY (strict)
                label_tokens = [w["text"].strip() for w in L["words"] if w["x1"] <= label_right_edge]
                label_text = " ".join(t for t in label_tokens if t)
                canon = normalize_reason(label_text)
                if not canon:
                    continue

                sect = data[current_ad].setdefault(current_store, {}).setdefault(current_section, {})
                per_header = sect.setdefault("__all__", defaultdict(lambda: defaultdict(int)))
                for w in L["words"]:
                    token = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", token):
                        continue
                    if w["x0"] <= label_right_edge:
                        continue
                    xmid = (w["x0"] + w["x1"]) / 2
                    mapped = map_x_to_header(header_bins, xmid)
                    if mapped is None:
                        continue
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
            if section not in {"To Go", "Delivery", "Dine-In"}:
                continue
            all_per_header = reason_map.get("__all__", {})
            for canon in ALL_CANONICAL:
                v = int(all_per_header.get(canon, {}).get(sel_col, 0))
                rows.append({
                    "Area Director": ad,
                    "Store": store,
                    "Section": section,
                    "Reason": canon,
                    "Value": v,
                })

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No matching reasons found for the selected period.")
    st.stop()

# ───────────────── CATEGORY MAPPING ─────────────────
CATEGORY_TOGO_MISSING = "To-go Missing Complaints"
CATEGORY_ATTITUDE     = "Attitude"
CATEGORY_OTHER        = "Other"

CATEGORY_MAP = {r: CATEGORY_TOGO_MISSING for r in MISSING_REASONS}
CATEGORY_MAP.update({r: CATEGORY_ATTITUDE for r in ATTITUDE_REASONS})
CATEGORY_MAP.update({r: CATEGORY_OTHER for r in OTHER_REASONS})
df["Category"] = df["Reason"].map(CATEGORY_MAP).fillna("Unassigned")

# ───────────────── DISPLAY (collapsible AD sections) ─────────────────
st.success("✅ Parsed with strict labels & TOTAL-aware bins.")
st.subheader(f"Results for period: {sel_col}")

col1, col2 = st.columns([1, 3])
with col1:
    expand_all = st.toggle("Expand all Area Directors", value=False, help="Show all stores & reason pivots for each AD")

with col2:
    ad_totals = (
        df.groupby(["Area Director","Store"], as_index=False)["Value"].sum()
          .groupby("Area Director", as_index=False)["Value"].sum()
          .rename(columns={"Value":"AD Total"})
    )
    st.dataframe(style_table(ad_totals, highlight_grand_total=False), use_container_width=True,
                 height=min(400, 60 + 28 * max(2, len(ad_totals))))

# per-store + per-AD totals (all rows)
store_totals = (
    df.groupby(["Area Director","Store"], as_index=False)["Value"].sum()
      .rename(columns={"Value":"Store Total"})
)
df_detail = df.merge(store_totals, on=["Area Director","Store"], how="left") \
              .merge(ad_totals, on="Area Director", how="left")

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
            show_reasons = MISSING_REASONS + ATTITUDE_REASONS + OTHER_REASONS
            pivot = (
                substore[substore["Reason"].isin(show_reasons)]
                .pivot_table(index="Reason", columns="Section", values="Value", aggfunc="sum", fill_value=0)
                .reindex(show_reasons)
            )
            pivot["Total"] = pivot.sum(axis=1)
            st.dataframe(style_table(pivot, highlight_grand_total=False), use_container_width=True)

# ───────────────── REASON TOTALS — Missing ─────────────────
st.header("4) Reason totals — To-go Missing Complaints (selected period)")
st.caption("To-Go and Delivery only, for the seven Missing reasons.")

missing_df = df[df["Reason"].isin(MISSING_REASONS) & df["Section"].isin({"To Go","Delivery"})]

def _order_series_missing(s: pd.Series) -> pd.Series:
    return s.reindex(MISSING_REASONS)

tot_to_go = (
    missing_df[missing_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
tot_delivery = (
    missing_df[missing_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
tot_overall = (
    missing_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)
)

reason_totals_missing = pd.DataFrame({
    "To Go": _order_series_missing(tot_to_go),
    "Delivery": _order_series_missing(tot_delivery),
    "Total": _order_series_missing(tot_overall),
}).fillna(0).astype(int)
reason_totals_missing.loc["— Grand Total —"] = reason_totals_missing.sum(numeric_only=True)

st.dataframe(style_table(reason_totals_missing), use_container_width=True)

# ───────────────── REASON TOTALS — Attitude ─────────────────
st.header("4b) Reason totals — Attitude (selected period)")
st.caption("All segments (Dine-In, To Go, Delivery) for the seven Attitude reasons.")

att_df = df[df["Reason"].isin(ATTITUDE_REASONS)]

def _order_series_att(s: pd.Series) -> pd.Series:
    return s.reindex(ATTITUDE_REASONS)

att_dinein = (
    att_df[att_df["Section"] == "Dine-In"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
att_togo = (
    att_df[att_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
att_delivery = (
    att_df[att_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
att_total = (
    att_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)
)

reason_totals_attitude = pd.DataFrame({
    "Dine-In": _order_series_att(att_dinein),
    "To Go": _order_series_att(att_togo),
    "Delivery": _order_series_att(att_delivery),
    "Total": _order_series_att(att_total),
}).fillna(0).astype(int)
reason_totals_attitude.loc["— Grand Total —"] = reason_totals_attitude.sum(numeric_only=True)

st.dataframe(style_table(reason_totals_attitude), use_container_width=True)

# ───────────────── REASON TOTALS — Other ─────────────────
st.header("4c) Reason totals — Other (selected period)")
st.caption("All segments (Dine-In, To Go, Delivery) for the seven Other reasons.")

oth_df = df[df["Reason"].isin(OTHER_REASONS)]

def _order_series_other(s: pd.Series) -> pd.Series:
    return s.reindex(OTHER_REASONS)

oth_dinein = (
    oth_df[oth_df["Section"] == "Dine-In"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
oth_togo = (
    oth_df[oth_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
oth_delivery = (
    oth_df[oth_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"].sum().astype(int)
)
oth_total = (
    oth_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)
)

reason_totals_other = pd.DataFrame({
    "Dine-In": _order_series_other(oth_dinein),
    "To Go": _order_series_other(oth_togo),
    "Delivery": _order_series_other(oth_delivery),
    "Total": _order_series_other(oth_total),
}).fillna(0).astype(int)
reason_totals_other.loc["— Grand Total —"] = reason_totals_other.sum(numeric_only=True)

st.dataframe(style_table(reason_totals_other), use_container_width=True)

# ───────────────── CATEGORY SUMMARIES ─────────────────
def category_summary_block(number_label: str, category_name: str, allowed_sections: set):
    st.header(f"{number_label}) Category summary — {category_name}")
    subset = df[(df["Category"] == category_name) & (df["Section"].isin(allowed_sections))]
    if subset.empty:
        st.info(f"No rows currently mapped to “{category_name}”.")
        return None, None, 0
    cat_store_totals = (
        subset.groupby(["Area Director", "Store"], as_index=False)["Value"]
              .sum().rename(columns={"Value": "Category Total"})
    )
    cat_ad_totals = (
        cat_store_totals.groupby("Area Director", as_index=False)["Category Total"]
                        .sum().rename(columns={"Category Total": "AD Category Total"})
    )
    cat_grand_total = int(cat_store_totals["Category Total"].sum())
    colA, colB = st.columns([1, 3])
    with colA:
        st.metric("Grand Total (Category)", cat_grand_total)
    with colB:
        st.dataframe(style_table(cat_ad_totals, highlight_grand_total=False),
                     use_container_width=True,
                     height=min(400, 60 + 28 * max(2, len(cat_ad_totals))))
    st.subheader("Per-Store Category Totals")
    st.caption(f"Each store’s total for “{category_name}” in the selected period.")
    st.dataframe(style_table(cat_store_totals, highlight_grand_total=False), use_container_width=True)
    return cat_ad_totals, cat_store_totals, cat_grand_total

# 5a — To-go Missing Complaints (To-Go + Delivery only)
tgc_ad_totals, tgc_store_totals, tgc_grand = category_summary_block("5a", "To-go Missing Complaints", {"To Go","Delivery"})
# 5b — Attitude (all segments)
att_ad_totals, att_store_totals, att_grand = category_summary_block("5b", "Attitude", {"To Go","Delivery","Dine-In"})
# 5c — Other (all segments)
oth_ad_totals, oth_store_totals, oth_grand = category_summary_block("5c", "Other", {"To Go","Delivery","Dine-In"})

# ───────────────── 6) PERIOD CHANGE SUMMARY (vs previous) ─────────────────
st.header("6) Period change summary (vs previous period)")
try:
    cur_idx = ordered_headers.index(sel_col)
    prior_label = ordered_headers[cur_idx - 1] if cur_idx > 0 else None
except ValueError:
    prior_label = None

if not prior_label:
    st.info("No earlier period available to compare against.")
else:
    def totals_by_reason_for(period_label: str, reasons: list[str], allowed_sections: set[str]) -> pd.Series:
        sums = {r: 0 for r in reasons}
        for ad, stores in raw_data.items():
            for store, sections in stores.items():
                for section, reason_map in sections.items():
                    if section not in allowed_sections:
                        continue
                    per = reason_map.get("__all__", {})
                    for r in reasons:
                        sums[r] += int(per.get(r, {}).get(period_label, 0))
        return pd.Series(sums).astype(int)

    missing_sections = {"To Go", "Delivery"}
    attitude_sections = {"To Go", "Delivery", "Dine-In"}
    other_sections    = {"To Go", "Delivery", "Dine-In"}

    # Current vs prior totals per reason
    cur_missing = totals_by_reason_for(sel_col, MISSING_REASONS, missing_sections)
    prv_missing = totals_by_reason_for(prior_label, MISSING_REASONS, missing_sections)
    cur_att     = totals_by_reason_for(sel_col, ATTITUDE_REASONS, attitude_sections)
    prv_att     = totals_by_reason_for(prior_label, ATTITUDE_REASONS, attitude_sections)
    cur_other   = totals_by_reason_for(sel_col, OTHER_REASONS, other_sections)
    prv_other   = totals_by_reason_for(prior_label, OTHER_REASONS, other_sections)

    # Deltas (plain +/- numbers)
    delta_missing = (cur_missing - prv_missing).astype(int)
    delta_att     = (cur_att - prv_att).astype(int)
    delta_other   = (cur_other - prv_other).astype(int)

    # Overall totals
    total_missing_cur = int(cur_missing.sum()); total_missing_prv = int(prv_missing.sum())
    total_att_cur     = int(cur_att.sum());     total_att_prv     = int(prv_att.sum())
    total_other_cur   = int(cur_other.sum());   total_other_prv   = int(prv_other.sum())

    def fmt_delta(n: int) -> str:
        return f"{n:+d}"

    # Build concise text
    lines = []
    lines.append(f"Selected period: {sel_col}   •   Prior: {prior_label}")
    lines.append("")
    # Missing
    lines.append("To-go Missing Complaints (To-Go + Delivery)")
    lines.append(f"- Overall: {total_missing_cur} ({fmt_delta(total_missing_cur - total_missing_prv)} vs prior)")
    any_change_missing = False
    for r in MISSING_REASONS:
        d = int(delta_missing.get(r, 0))
        if d != 0:
            lines.append(f"  • {r}: {int(cur_missing[r])} ({fmt_delta(d)})")
            any_change_missing = True
    if not any_change_missing:
        lines.append("  • No change by reason.")
    lines.append("")
    # Attitude
    lines.append("Attitude (All segments)")
    lines.append(f"- Overall: {total_att_cur} ({fmt_delta(total_att_cur - total_att_prv)} vs prior)")
    any_change_att = False
    for r in ATTITUDE_REASONS:
        d = int(delta_att.get(r, 0))
        if d != 0:
            lines.append(f"  • {r}: {int(cur_att[r])} ({fmt_delta(d)})")
            any_change_att = True
    if not any_change_att:
        lines.append("  • No change by reason.")
    lines.append("")
    # Other
    lines.append("Other (All segments)")
    lines.append(f"- Overall: {total_other_cur} ({fmt_delta(total_other_cur - total_other_prv)} vs prior)")
    any_change_other = False
    for r in OTHER_REASONS:
        d = int(delta_other.get(r, 0))
        if d != 0:
            lines.append(f"  • {r}: {int(cur_other[r])} ({fmt_delta(d)})")
            any_change_other = True
    if not any_change_other:
        lines.append("  • No change by reason.")

    summary_text = "\n".join(lines)

    # Dynamic height: ~24px per line, bounded
    num_lines = summary_text.count("\n") + 2
    dyn_height = max(160, min(700, 24 * num_lines))

    st.text_area("Copy to clipboard", summary_text, height=dyn_height)
    st.download_button(
        "📥 Download summary as .txt",
        data=summary_text.encode("utf-8"),
        file_name=f"period_change_summary_{sel_col.replace(' ','_')}.txt",
        mime="text/plain",
    )

# ───────────────── 7) Historical context — highs/lows vs all periods ─────────────────
st.header("7) Historical context — highs/lows vs all periods")

def build_reason_period_matrix(reasons: list[str], allowed_sections: set[str]) -> dict[str, dict[str, int]]:
    """
    Returns: {period -> {reason -> total}} across all ADs/stores for allowed sections.
    """
    mat = {p: {r: 0 for r in reasons} for p in ordered_headers}
    for ad, stores in raw_data.items():
        for store, sections_map in stores.items():
            for sec, reason_map in sections_map.items():
                if sec not in allowed_sections:
                    continue
                per = reason_map.get("__all__", {})
                for r in reasons:
                    pr = per.get(r, {})
                    for p in ordered_headers:
                        mat[p][r] += int(pr.get(p, 0))
    return mat

def build_highlow_tables(reasons: list[str], allowed_sections: set[str], title: str):
    st.subheader(title)

    mat = build_reason_period_matrix(reasons, allowed_sections)

    rows = []
    num_periods = len(ordered_headers)
    for r in reasons:
        series = [(p, mat[p][r]) for p in ordered_headers]
        # LOWER = BETTER
        best_period, best_val   = min(series, key=lambda kv: kv[1])  # lowest value
        worst_period, worst_val = max(series, key=lambda kv: kv[1])  # highest value
        # Rank 1 = lowest value
        sorted_asc = sorted(series, key=lambda kv: kv[1])
        rank = next(i+1 for i,(p,v) in enumerate(sorted_asc) if p == sel_col)
        cur_val = mat[sel_col][r]
        rows.append({
            "Reason": r,
            "Current": cur_val,
            "Rank": f"{rank}/{num_periods}",
            "Best (Period)": f"{best_val} @ {best_period}",
            "Worst (Period)": f"{worst_val} @ {worst_period}",
        })

    df_reasons = pd.DataFrame(rows).set_index("Reason")

    # Category totals across periods (lower = better)
    totals_by_period = {p: sum(mat[p][r] for r in reasons) for p in ordered_headers}
    best_p, best_total   = min(totals_by_period.items(), key=lambda kv: kv[1])
    worst_p, worst_total = max(totals_by_period.items(), key=lambda kv: kv[1])
    current_total = totals_by_period.get(sel_col, 0)
    rank_list = sorted(totals_by_period.items(), key=lambda kv: kv[1])  # ascending
    current_rank_idx = next(i+1 for i,(p,v) in enumerate(rank_list) if p == sel_col)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Current total (lower is better)", current_total)
    with colB:
        st.metric("Best total (lowest)", f"{best_total}", help=f"Period: {best_p}")
    with colC:
        st.metric("Worst total (highest)", f"{worst_total}", help=f"Period: {worst_p}")
    st.caption(f"Current period rank: {current_rank_idx}/{num_periods} (1 = lowest/best)")

    st.dataframe(style_table(df_reasons, highlight_grand_total=False), use_container_width=True)

# Missing → To-Go & Delivery only
build_highlow_tables(MISSING_REASONS, {"To Go","Delivery"}, "7a) To-go Missing Complaints (To-Go + Delivery) — highs/lows")

# Attitude → All segments
build_highlow_tables(ATTITUDE_REASONS, {"Dine-In","To Go","Delivery"}, "7b) Attitude (All segments) — highs/lows")

# Other → All segments
build_highlow_tables(OTHER_REASONS, {"Dine-In","To Go","Delivery"}, "7c) Other (All segments) — highs/lows")

# ───────────────── 8) EXPORT — Excel only (All Sheets) ─────────────────
st.header("8) Export results")

buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    # Detail + rollups
    df_detail.to_excel(writer, index=False, sheet_name="Detail")
    ad_totals.to_excel(writer, index=False, sheet_name="AD Totals")
    store_totals.to_excel(writer, index=False, sheet_name="Store Totals")
    # Reason totals
    reason_totals_missing.to_excel(writer, sheet_name="Reason Totals (Missing)")
    reason_totals_attitude.to_excel(writer, sheet_name="Reason Totals (Attitude)")
    reason_totals_other.to_excel(writer, sheet_name="Reason Totals (Other)")
    # Category sheets
    (tgc_ad_totals if tgc_ad_totals is not None else pd.DataFrame(columns=["Area Director","AD Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-ToGoMissing AD Totals")
    (tgc_store_totals if tgc_store_totals is not None else pd.DataFrame(columns=["Area Director","Store","Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-ToGoMissing Store")
    (att_ad_totals if att_ad_totals is not None else pd.DataFrame(columns=["Area Director","AD Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Attitude AD Totals")
    (att_store_totals if att_store_totals is not None else pd.DataFrame(columns=["Area Director","Store","Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Attitude Store")
    (oth_ad_totals if oth_ad_totals is not None else pd.DataFrame(columns=["Area Director","AD Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Other AD Totals")
    (oth_store_totals if oth_store_totals is not None else pd.DataFrame(columns=["Area Director","Store","Category Total"])) \
        .to_excel(writer, index=False, sheet_name="Cat-Other Store")

st.download_button(
    "📥 Download Excel (All Sheets)",
    data=buff.getvalue(),
    file_name=f"ad_store_{sel_col.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ───────────────── DEBUG / VERIFICATION ─────────────────
with st.expander("🧪 Debug: AD ↔ Store pairs detected this run"):
    if pairs_debug:
        st.dataframe(style_table(pd.DataFrame(pairs_debug, columns=["Area Director","Store"]), highlight_grand_total=False),
                     use_container_width=True)
    else:
        st.caption("No pairs captured (unexpected).")
