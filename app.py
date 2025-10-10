# Greeno Big Three v1.6.4 — strict parser (TOTAL-aware bins, left-label) +
# collapsible ADs + reason totals (Missing + Attitude) + text-only Eric email +
# Categories: To-go Missing Complaints (To-Go/Delivery) + Attitude (all segments) + Other (placeholder)
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
st.set_page_config(page_title="Greeno Big Three v1.6.4", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.6.4</h1>
      <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
      <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
        Strict parsing + TOTAL-aware bins · Collapsible AD sections · Reason totals (Missing + Attitude) · Text-only Eric email · Category summaries
      </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ───────────────── SIDEBAR UPLOAD ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose the PDF report", type=["pdf"])
    st.caption("Parses labels from the left side. Missing uses To-Go & Delivery; Attitude includes all segments.")

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

ALL_CANONICAL = MISSING_REASONS + ATTITUDE_REASONS

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
REASON_ALIASES_NORM = {**ALIASES_MISSING, **ALIASES_ATTITUDE}

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
CATEGORY_OTHER        = "Other"      # placeholder

CATEGORY_MAP = {r: CATEGORY_TOGO_MISSING for r in MISSING_REASONS}
CATEGORY_MAP.update({r: CATEGORY_ATTITUDE for r in ATTITUDE_REASONS})
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
    st.dataframe(ad_totals.sort_values("Area Director"), use_container_width=True, height=min(400, 60 + 28 * max(2, len(ad_totals))))

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
            show_reasons = MISSING_REASONS + ATTITUDE_REASONS
            pivot = (
                substore[substore["Reason"].isin(show_reasons)]
                .pivot_table(index="Reason", columns="Section", values="Value", aggfunc="sum", fill_value=0)
                .reindex(show_reasons)
            )
            pivot["Total"] = pivot.sum(axis=1)
            st.dataframe(pivot, use_container_width=True)

# ───────────────── REASON TOTALS — Missing ─────────────────
st.header("4) Reason totals — To-go Missing Complaints (selected period)")
st.caption("To-Go and Delivery only, for the seven Missing reasons (matches your spreadsheet flow).")

missing_df = df[df["Reason"].isin(MISSING_REASONS) & df["Section"].isin({"To Go","Delivery"})]

def _order_series_missing(s: pd.Series) -> pd.Series:
    return s.reindex(MISSING_REASONS)

tot_to_go = (
    missing_df[missing_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)
tot_delivery = (
    missing_df[missing_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)
tot_overall = (
    missing_df.groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)

reason_totals_missing = pd.DataFrame({
    "To Go": _order_series_missing(tot_to_go),
    "Delivery": _order_series_missing(tot_delivery),
    "Total": _order_series_missing(tot_overall),
}).fillna(0).astype(int)
reason_totals_missing.loc["— Grand Total —"] = reason_totals_missing.sum(numeric_only=True)

st.dataframe(reason_totals_missing, use_container_width=True)

# ───────────────── REASON TOTALS — Attitude ─────────────────
st.header("4b) Reason totals — Attitude (selected period)")
st.caption("All segments (Dine-In, To Go, Delivery) for the seven Attitude reasons.")

att_df = df[df["Reason"].isin(ATTITUDE_REASONS)]

def _order_series_att(s: pd.Series) -> pd.Series:
    return s.reindex(ATTITUDE_REASONS)

att_dinein = (
    att_df[att_df["Section"] == "Dine-In"]
      .groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)
att_togo = (
    att_df[att_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)
att_delivery = (
    att_df[att_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)
att_total = (
    att_df.groupby("Reason", as_index=True)["Value"]
      .sum()
      .astype(int)
)

reason_totals_attitude = pd.DataFrame({
    "Dine-In": _order_series_att(att_dinein),
    "To Go": _order_series_att(att_togo),
    "Delivery": _order_series_att(att_delivery),
    "Total": _order_series_att(att_total),
}).fillna(0).astype(int)
reason_totals_attitude.loc["— Grand Total —"] = reason_totals_attitude.sum(numeric_only=True)

st.dataframe(reason_totals_attitude, use_container_width=True)

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
        st.dataframe(
            cat_ad_totals.sort_values("Area Director"),
            use_container_width=True,
            height=min(400, 60 + 28 * max(2, len(cat_ad_totals)))
        )
    st.subheader("Per-Store Category Totals")
    st.caption(f"Each store’s total for “{category_name}” in the selected period.")
    st.dataframe(
        cat_store_totals.sort_values(["Area Director", "Store"]),
        use_container_width=True,
    )
    return cat_ad_totals, cat_store_totals, cat_grand_total

# 4c — To-go Missing Complaints (To-Go + Delivery only)
tgc_ad_totals, tgc_store_totals, tgc_grand = category_summary_block("4c", "To-go Missing Complaints", {"To Go","Delivery"})
# 4d — Attitude (all segments)
att_ad_totals, att_store_totals, att_grand = category_summary_block("4d", "Attitude", {"To Go","Delivery","Dine-In"})
# 4e — Other (placeholder)
oth_ad_totals, oth_store_totals, oth_grand = category_summary_block("4e", "Other", {"To Go","Delivery","Dine-In"})

# ───────────────── DRILL-DOWN / EVIDENCE ─────────────────
with st.expander("🔎 Drill-down: show exact values per header for any AD & Store"):
    if raw_data:
        ad_pick = st.selectbox("Area Director", sorted(raw_data.keys()))
        store_pick = st.selectbox("Store", sorted(raw_data[ad_pick].keys()))
        sec_pick = st.radio("Section", ["To Go", "Delivery", "Dine-In"], horizontal=True)

        per = raw_data[ad_pick][store_pick].get(sec_pick, {}).get("__all__", {})
        mat = []
        for r in MISSING_REASONS + ATTITUDE_REASONS:
            row = {"Reason": r}
            for h in ordered_headers:
                row[h] = int(per.get(r, {}).get(h, 0))
            mat.append(row)
        df_mat = pd.DataFrame(mat).set_index("Reason")
        st.dataframe(df_mat, use_container_width=True)
        st.caption("Verifies the exact numbers parsed under each period header for your chosen store/section.")
    else:
        st.caption("No data parsed.")

# ───────────────── EMAIL GENERATOR (text-only Eric voice) ─────────────────
st.header("5) Generate Eric-style email (text only)")
st.caption("Outputs a plain-text email summary using Eric’s tone and structure.")

def compute_delta_vs_prior(sel: str) -> Optional[Tuple[str, int]]:
    try:
        idx = ordered_headers.index(sel)
    except ValueError:
        return None
    if idx == 0:
        return None
    prior = ordered_headers[idx - 1]
    # Email focuses on Missing (To Go + Delivery)
    cur_total = int(missing_df["Value"].sum())
    rows_prior = []
    # rebuild prior under same scope
    for ad, stores in raw_data.items():
        for store, sections in stores.items():
            for section, reason_map in sections.items():
                if section not in {"To Go", "Delivery"}:
                    continue
                all_per_header = reason_map.get("__all__", {})
                for canon in MISSING_REASONS:
                    rows_prior.append(int(all_per_header.get(canon, {}).get(prior, 0)))
    prior_total = int(sum(rows_prior)) if rows_prior else 0
    return prior, cur_total - prior_total

top3 = (
    reason_totals_missing.drop(index="— Grand Total —", errors="ignore")
    .sort_values("Total", ascending=False)
    .head(3)
    .index.tolist()
)

delta_info = compute_delta_vs_prior(sel_col)
delta_line = ""
if delta_info:
    prior_label, delta_val = delta_info
    arrow = "down" if delta_val < 0 else "up" if delta_val > 0 else "flat"
    delta_line = f"P vs {prior_label}: {delta_val:+d} ({arrow})."

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
    "To-Go and Delivery are the only channels included in this Missing category view. Attitude includes all segments."
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

# Detail CSV (all rows captured)
csv = df_detail.to_csv(index=False)
st.download_button("📥 Download detail CSV", data=csv, file_name=f"ad_store_detail_{sel_col.replace(' ','_')}.csv", mime="text/csv")

# Reason totals CSVs
st.download_button(
    "📥 Download Reason Totals — Missing (CSV)",
    data=reason_totals_missing.to_csv().encode("utf-8"),
    file_name=f"reason_totals_missing_{sel_col.replace(' ','_')}.csv",
    mime="text/csv"
)
st.download_button(
    "📥 Download Reason Totals — Attitude (CSV)",
    data=reason_totals_attitude.to_csv().encode("utf-8"),
    file_name=f"reason_totals_attitude_{sel_col.replace(' ','_')}.csv",
    mime="text/csv"
)

# Category CSVs
def maybe_dl(df_in, label, fname):
    if df_in is not None:
        st.download_button(label, data=df_in.to_csv(index=False).encode("utf-8"), file_name=fname, mime="text/csv")

maybe_dl(tgc_ad_totals,    "📥 Category AD Totals — To-go Missing (CSV)", f"category_togo_missing_ad_{sel_col.replace(' ','_')}.csv")
maybe_dl(tgc_store_totals, "📥 Category Store Totals — To-go Missing (CSV)", f"category_togo_missing_store_{sel_col.replace(' ','_')}.csv")
maybe_dl(att_ad_totals,    "📥 Category AD Totals — Attitude (CSV)", f"category_attitude_ad_{sel_col.replace(' ','_')}.csv")
maybe_dl(att_store_totals, "📥 Category Store Totals — Attitude (CSV)", f"category_attitude_store_{sel_col.replace(' ','_')}.csv")
maybe_dl(oth_ad_totals,    "📥 Category AD Totals — Other (CSV)", f"category_other_ad_{sel_col.replace(' ','_')}.csv")
maybe_dl(oth_store_totals, "📥 Category Store Totals — Other (CSV)", f"category_other_store_{sel_col.replace(' ','_')}.csv")

# Excel (All + Reason Totals + Category sheets)
buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    df_detail.to_excel(writer, index=False, sheet_name="Detail")
    ad_totals.to_excel(writer, index=False, sheet_name="AD Totals")
    store_totals.to_excel(writer, index=False, sheet_name="Store Totals")
    reason_totals_missing.to_excel(writer, sheet_name="Reason Totals (Missing)")
    reason_totals_attitude.to_excel(writer, sheet_name="Reason Totals (Attitude)")

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
        st.dataframe(pd.DataFrame(pairs_debug, columns=["Area Director","Store"]))
    else:
        st.caption("No pairs captured (unexpected).")
