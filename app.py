# Greeno Big Three v1.8.4 — trigger-based matching + adjacent-line fallback
# - pdfplumber parser (no fitz)
# - Short, unique triggers for metrics (robust to wraps/variation)
# - Compensation special-case (captures prev/next lines)
# - NEW: generic adjacent-line fallback for all reasons if trigger line has no numbers
# - Keeps scoreboard, category totals, highs/lows, exports, debug

import io, os, re, base64, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── STYLING HELPERS ─────────────────
def style_table(df: pd.DataFrame, highlight_grand_total: bool = True) -> "pd.io.formats.style.Styler":
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
    sty = sty.set_table_styles(
        [{"selector": "th.row_heading, th.blank", "props": [("color", "#111"), ("border-color", "#CCD3DB")]}]
    )
    return sty

# ───────────────── HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.8.4", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.8.4</h1>
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
    st.caption("Missing = To-Go/Delivery (except ‘Out of menu item’ includes Dine-In). Attitude/Other = all segments.")
    debug_mode = st.checkbox("🔍 Enable Debug Mode", value=False)

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

MISSING_REASONS = [
    "Missing food","Order wrong","Missing condiments","Out of menu item",
    "Missing bev","Missing ingredients","Packaging to-go complaint",
]
ATTITUDE_REASONS = [
    "Unprofessional/Unfriendly","Manager directly involved","Manager not available",
    "Manager did not visit","Negative mgr-employee exchange","Manager did not follow up",
    "Argued with guest",
]
OTHER_REASONS = [
    "Long hold/no answer","No/insufficient compensation offered","Did not attempt to resolve",
    "Guest left without ordering","Unknowledgeable","Did not open on time","No/poor apology",
]
ALL_CANONICAL = MISSING_REASONS + ATTITUDE_REASONS + OTHER_REASONS

# Short, robust triggers (lowercase substring matches)
KEYWORD_TRIGGERS = {
    # TO-GO MISSING
    "Missing food": ["missing food"],
    "Order wrong": ["order wrong"],
    "Missing condiments": ["condiments"],
    "Out of menu item": ["out of menu"],
    "Missing bev": ["missing bev"],
    "Missing ingredients": ["ingredient"],
    "Packaging to-go complaint": ["packaging"],

    # ATTITUDE (all segments)
    "Unprofessional/Unfriendly": ["unfriendly"],
    "Manager directly involved": ["directly involved", "involved"],
    "Manager not available": ["manager not available"],
    "Manager did not visit": ["did not visit", "no visit"],
    "Negative mgr-employee exchange": ["manager-employee", "exchange"],
    "Manager did not follow up": ["follow up"],
    "Argued with guest": ["argued"],

    # OTHER (all segments)
    "Long hold/no answer": ["hold", "no answer", "hung up"],
    "No/insufficient compensation offered": ["compensation"],
    "Did not attempt to resolve": ["resolve"],
    "Guest left without ordering": ["without ordering"],
    "Unknowledgeable": ["unknowledgeable"],
    "Did not open on time": ["open on time"],
    "No/poor apology": ["apology"],
}

SPECIAL_REASON_SECTIONS = {
    # Out of menu item counts across *all* segments
    "Out of menu item": {"To Go", "Delivery", "Dine-In"}
}

COMP_CANON = "No/insufficient compensation offered"

def _lc(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def _matches_keyword(label_text_lc: str) -> Optional[str]:
    for canon, triggers in KEYWORD_TRIGGERS.items():
        for trig in triggers:
            if trig in label_text_lc:
                return canon
    return None

def _is_comp_line(label_lc: str) -> bool:
    # any piece in the 2–3 line wrap OR "compensation" itself
    return (
        "no/unsatisfactory" in label_lc or
        "compensation offered by" in label_lc or
        label_lc.strip() == "restaurant" or
        "compensation" in label_lc
    )

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

def is_structural_total(label_text_lc: str) -> bool:
    return (
        label_text_lc.endswith(" total:") or
        label_text_lc == "dine-in total:" or
        label_text_lc == "to go total:" or
        label_text_lc == "delivery total:" or
        label_text_lc == "total:"
    )

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
            right = (x + total_x)/2 if total_x is not None else x + 0.6*med_gap
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
def parse_pdf_build_ad_store_period_map(file_bytes: bytes, debug: bool = False):
    header_positions: Dict[str, float] = {}
    ordered_headers: List[str] = []
    pairs_debug: List[Tuple[str, str]] = []

    data: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    debug_log = {
        "unmatched_labels": [],
        "ignored_tokens": [],
        "token_trace": [],
        "events": [],
        "header_bins": [],
        "raw_layout": [],
        "facts": [],
    }

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        carry_headers = None
        carry_total_x = None
        for page in pdf.pages:
            headers = find_period_headers(page) or carry_headers
            if not headers:
                continue
            if find_period_headers(page):
                carry_headers = headers[:]
                carry_total_x = None
            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)

            total_x = find_total_header_x(page, header_y) or carry_total_x
            if total_x is not None:
                carry_total_x = total_x
            header_bins = build_header_bins({h: header_positions[h] for h in ordered_headers}, total_x)

            if debug:
                debug_log["header_bins"].append({
                    "page": page.page_number,
                    "headers": [h for (h, _, _) in headers],
                    "header_positions": {h: header_positions[h] for h in ordered_headers},
                    "total_x": total_x,
                    "bins": [{"period": h, "left": left, "right": right} for (h, left, right) in header_bins],
                })

            first_period_x = min(header_positions[h] for h in ordered_headers)
            label_right_edge = first_period_x - 10

            lines = extract_words_grouped(page)
            if not lines:
                continue

            left_margin = min(L["x_min"] for L in lines)
            current_ad: Optional[str] = None
            current_store: Optional[str] = None
            current_section: Optional[str] = None

            def left_label_text(line):
                return " ".join(
                    w["text"].strip()
                    for w in line["words"]
                    if w["x1"] <= label_right_edge and w["text"].strip()
                ).strip()

            def consume_words(line_obj, canon_reason: str):
                y_band = line_obj["y"]
                for w in line_obj["words"]:
                    token = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", token):
                        continue
                    if w["x0"] <= label_right_edge:
                        continue
                    w_y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
                    if abs(w_y_mid - y_band) > 0.01:
                        continue
                    xmid = (w["x0"] + w["x1"]) / 2
                    if total_x is not None and xmid >= (total_x - 1.0):
                        if debug:
                            debug_log["ignored_tokens"].append({
                                "page": page.page_number,
                                "token": token,
                                "xmid": xmid,
                                "reason": f"{canon_reason} (>= TOTAL cutoff)",
                                "store": current_store,
                                "section": current_section,
                            })
                        continue
                    mapped = map_x_to_header(header_bins, xmid)
                    if mapped is None or mapped not in ordered_headers:
                        if debug:
                            debug_log["ignored_tokens"].append({
                                "page": page.page_number,
                                "token": token,
                                "xmid": xmid,
                                "reason": f"{canon_reason} (no header bin)",
                                "store": current_store,
                                "section": current_section,
                            })
                        continue
                    sect = data[current_ad].setdefault(current_store, {}).setdefault(current_section, {})
                    per_header = sect.setdefault("__all__", defaultdict(lambda: defaultdict(int)))
                    per_header[canon_reason][mapped] += int(token)

                    if debug:
                        debug_log["token_trace"].append({
                            "page": page.page_number,
                            "ad": current_ad,
                            "store": current_store,
                            "section": current_section,
                            "reason": canon_reason,
                            "period": mapped,
                            "value": int(token),
                        })
                        debug_log["facts"].append({
                            "Area Director": current_ad, "Store": current_store, "Section": current_section,
                            "Reason": canon_reason, "Period": mapped, "Value": int(token),
                        })

            # NEW: wrapper that returns how many tokens we captured
            def consume_and_count(line_obj, canon_reason: str) -> int:
                if debug:
                    before = len(debug_log["token_trace"])
                    consume_words(line_obj, canon_reason)
                    after = len(debug_log["token_trace"])
                    return max(0, after - before)
                else:
                    # Heuristic when debug is off: count numeric tokens to the right of label edge
                    cnt = 0
                    y_band = line_obj["y"]
                    for w in line_obj["words"]:
                        token = w["text"].strip()
                        if not re.fullmatch(r"-?\d+", token):
                            continue
                        if w["x0"] <= label_right_edge:
                            continue
                        w_y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
                        if abs(w_y_mid - y_band) <= 0.01:
                            cnt += 1
                    consume_words(line_obj, canon_reason)
                    return cnt

            idx = 0
            while idx < len(lines):
                L = lines[idx]
                txt = L["text"].strip()

                if debug:
                    for w in L["words"]:
                        debug_log["raw_layout"].append({
                            "page": page.page_number,
                            "y": L["y"],
                            "x0": w["x0"],
                            "x1": w["x1"],
                            "xmid": (w["x0"] + w["x1"]) / 2,
                            "text": w["text"],
                            "ad": current_ad,
                            "store": current_store,
                            "section": current_section,
                        })

                # Store detection
                if STORE_LINE_RX.match(txt):
                    ad_for_this_store = find_ad_for_store(lines, idx, left_margin)
                    if ad_for_this_store:
                        current_ad = ad_for_this_store
                    current_store = txt
                    current_section = None
                    if current_ad:
                        pairs_debug.append((current_ad, current_store))
                    idx += 1
                    continue

                # Section markers
                if SECTION_TOGO.match(txt):
                    current_section = "To Go";   idx += 1; continue
                if SECTION_DELIV.match(txt):
                    current_section = "Delivery"; idx += 1; continue
                if SECTION_DINEIN.match(txt):
                    current_section = "Dine-In";  idx += 1; continue

                if txt in HEADINGS:
                    idx += 1
                    continue
                if not (current_ad and current_store and current_section in {"To Go", "Delivery", "Dine-In"}):
                    idx += 1
                    continue

                label_text = left_label_text(L)
                label_lc = _lc(label_text)

                if is_structural_total(label_lc):
                    idx += 1
                    continue

                # Special-case: compensation (capture current, prev, next lines)
                canon = _matches_keyword(label_lc)
                if canon == COMP_CANON or _is_comp_line(label_lc):
                    canon = COMP_CANON
                    consume_words(L, canon)
                    if idx > 0:
                        prev_lc = _lc(left_label_text(lines[idx - 1]))
                        if _is_comp_line(prev_lc):
                            consume_words(lines[idx - 1], canon)
                    if (idx + 1) < len(lines):
                        next_lc = _lc(left_label_text(lines[idx + 1]))
                        if _is_comp_line(next_lc):
                            consume_words(lines[idx + 1], canon)
                    idx += 1
                    continue

                # Default handling with ADJACENT-LINE FALLBACK
                if not canon:
                    if debug:
                        debug_log["unmatched_labels"].append({
                            "page": page.page_number,
                            "text": label_text,
                            "ad": current_ad,
                            "store": current_store,
                            "section": current_section,
                        })
                    idx += 1
                    continue

                got = consume_and_count(L, canon)

                # If nothing captured on the trigger line, try immediate previous/next (wrapped labels)
                if got == 0:
                    tried = False
                    if idx > 0:
                        prev_L = lines[idx - 1]
                        prev_label_lc = _lc(left_label_text(prev_L))
                        if not is_structural_total(prev_label_lc):
                            got += consume_and_count(prev_L, canon)
                            tried = True
                    if got == 0 and (idx + 1) < len(lines):
                        next_L = lines[idx + 1]
                        next_label_lc = _lc(left_label_text(next_L))
                        if not is_structural_total(next_label_lc):
                            got += consume_and_count(next_L, canon)
                            tried = True
                    if debug and got > 0:
                        debug_log["events"].append({
                            "type": "adjacent_capture",
                            "reason": canon,
                            "page": page.page_number,
                            "ad": current_ad,
                            "store": current_store,
                            "section": current_section,
                            "note": "numbers on adjacent line due to wrap"
                        })
                idx += 1

    return {h: header_positions[h] for h in ordered_headers}, data, ordered_headers, pairs_debug, debug_log

# ───────────────── RUN ─────────────────
with st.spinner("Roll Tide…"):
    header_x_map, raw_data, ordered_headers, pairs_debug, debug_log = parse_pdf_build_ad_store_period_map(
        file_bytes, debug=debug_mode
    )

if debug_mode:
    st.info(
        f"Debug — Unmatched labels: {len(debug_log['unmatched_labels'])} • "
        f"Ignored tokens: {len(debug_log['ignored_tokens'])}"
    )

if not ordered_headers:
    st.error("No period headers (like ‘P9 25’) found.")
    st.stop()

# ───────────────── PERIOD PICKER ─────────────────
st.header("2) Pick the period")
sel_col = st.selectbox("Period", options=ordered_headers, index=len(ordered_headers)-1)

# ───────────────── AGG HELPERS ─────────────────
def _allowed_sections_for_reason(reason: str, default_sections: set[str]) -> set[str]:
    return SPECIAL_REASON_SECTIONS.get(reason, default_sections)

def _total_for(period_label: Optional[str], reasons: list[str], default_sections: set[str]) -> int:
    if not period_label:
        return 0
    total = 0
    for ad, stores in raw_data.items():
        for store, sects in stores.items():
            for sec_name, reason_map in sects.items():
                per = reason_map.get("__all__", {})
                for r in reasons:
                    allowed = _allowed_sections_for_reason(r, default_sections)
                    if sec_name not in allowed:
                        continue
                    total += int(per.get(r, {}).get(period_label, 0))
    return int(total)

def _totals_by_period(reasons: list[str], default_sections: set[str]) -> Dict[str, int]:
    res = {p: 0 for p in ordered_headers}
    for ad, stores in raw_data.items():
        for store, sects in stores.items():
            for sec_name, reason_map in sects.items():
                per = reason_map.get("__all__", {})
                for r in reasons:
                    allowed = _allowed_sections_for_reason(r, default_sections)
                    if sec_name not in allowed:
                        continue
                    for p in ordered_headers:
                        res[p] += int(per.get(r, {}).get(p, 0))
    return res

# ───────────────── QUICK GLANCE (vs previous) ─────────────────
st.markdown("### Quick glance")

try:
    cur_idx = ordered_headers.index(sel_col)
    prior_label = ordered_headers[cur_idx - 1] if cur_idx > 0 else None
except ValueError:
    prior_label = None

missing_sections = {"To Go", "Delivery"}
att_sections     = {"To Go", "Delivery", "Dine-In"}
other_sections   = {"To Go", "Delivery", "Dine-In"}

tot_missing_cur = _total_for(sel_col,     MISSING_REASONS,  missing_sections)
tot_att_cur     = _total_for(sel_col,     ATTITUDE_REASONS, att_sections)
tot_other_cur   = _total_for(sel_col,     OTHER_REASONS,    other_sections)
tot_missing_prv = _total_for(prior_label, MISSING_REASONS,  missing_sections)
tot_att_prv     = _total_for(prior_label, ATTITUDE_REASONS, att_sections)
tot_other_prv   = _total_for(prior_label, OTHER_REASONS,    other_sections)

overall_cur = tot_missing_cur + tot_att_cur + tot_other_cur
overall_prv = (tot_missing_prv + tot_att_prv + tot_other_prv) if prior_label else 0

def diff_val(cur, prv, has_prior): 
    return (cur - prv) if has_prior else None
def fmt_diff(d): 
    return "n/a" if d is None else f"{d:+d}"
def cls_from_delta(d):
    if d is None: return ""
    return " best" if d < 0 else (" worst" if d > 0 else "")

overall_diff = diff_val(overall_cur, overall_prv, prior_label is not None)
miss_diff    = diff_val(tot_missing_cur, tot_missing_prv, prior_label is not None)
att_diff     = diff_val(tot_att_cur,     tot_att_prv,     prior_label is not None)
oth_diff     = diff_val(tot_other_cur,   tot_other_prv,   prior_label is not None)

overall_cls = cls_from_delta(overall_diff)
missing_cls = cls_from_delta(miss_diff)
att_cls     = cls_from_delta(att_diff)
other_cls   = cls_from_delta(oth_diff)

def diff_class(d):
    if d is None: return ""
    return "neg" if d < 0 else ("pos" if d > 0 else "zero")

prior_text = prior_label or "n/a"

score_css = """
<style>
.score-wrap{display:flex;gap:16px;margin:10px 0 8px 0}
.score{flex:1;background:#1113;border:2px solid #38414a;border-radius:14px;padding:18px 20px;text-align:center}
.score.best{border-color:#66BB6A; box-shadow:0 0 0 1px rgba(102,187,106,.55) inset}
.score.worst{border-color:#EF5350; box-shadow:0 0 0 1px rgba(239,83,80,.55) inset}
.score h4{margin:0 0 8px 0;font-weight:700;font-size:1.15rem;color:#cfd8e3}
.score .num{font-size:3rem;line-height:1.1;font-weight:800;color:#fff;margin:2px 0 2px}
.score .delta{margin-top:6px;font-size:1.05rem}
.delta.neg{color:#66BB6A}
.delta.pos{color:#EF5350}
.delta.zero{color:#9fb3c8}
.delta .vs{opacity:.85;margin-left:8px}
@media (max-width:900px){.score-wrap{flex-direction:column}}
</style>
"""
st.markdown(score_css, unsafe_allow_html=True)

score_html = f"""
<div class="score-wrap">
  <div class="score{overall_cls}">
    <h4>Overall (all categories)</h4>
    <div class="num">{overall_cur}</div>
    <div class="delta {diff_class(overall_diff)}">{fmt_diff(overall_diff)}<span class="vs">vs {prior_text}</span></div>
  </div>
  <div class="score{missing_cls}">
    <h4>To-go Missing Complaints</h4>
    <div class="num">{tot_missing_cur}</div>
    <div class="delta {diff_class(miss_diff)}">{fmt_diff(miss_diff)}<span class="vs">vs {prior_text}</span></div>
  </div>
  <div class="score{att_cls}">
    <h4>Attitude</h4>
    <div class="num">{tot_att_cur}</div>
    <div class="delta {diff_class(att_diff)}">{fmt_diff(att_diff)}<span class="vs">vs {prior_text}</span></div>
  </div>
  <div class="score{other_cls}">
    <h4>Other</h4>
    <div class="num">{tot_other_cur}</div>
    <div class="delta {diff_class(oth_diff)}">{fmt_diff(oth_diff)}<span class="vs">vs {prior_text}</span></div>
  </div>
</div>
"""
st.markdown(score_html, unsafe_allow_html=True)
if prior_label:
    st.caption(f"Δ shows change vs previous period ({prior_label}). Lower is better.")
else:
    st.caption("No previous period available — deltas shown as n/a. Lower is better.")

# ───────────────── BUILD ROWS (for summaries & export) ─────────────────
rows = []
for ad, stores in raw_data.items():
    for store, sections in stores.items():
        for section, reason_map in sections.items():
            if section not in {"To Go", "Delivery", "Dine-In"}:
                continue
            all_per_header = reason_map.get("__all__", {})
            for canon in ALL_CANONICAL:
                vdict = all_per_header.get(canon, {})
                for period, val in vdict.items():
                    rows.append({
                        "Area Director": ad, "Store": store, "Section": section,
                        "Reason": canon, "Period": period, "Value": int(val),
                    })

df_all = pd.DataFrame(rows)
df = df_all[df_all["Period"] == sel_col].copy()

if df.empty:
    st.warning("No matching reasons found for the selected period.")
    st.stop()

# For export only (not displayed)
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

# ───────────────── CATEGORY MAPPING ─────────────────
CATEGORY_TOGO_MISSING = "To-go Missing Complaints"
CATEGORY_ATTITUDE     = "Attitude"
CATEGORY_OTHER        = "Other"

CATEGORY_MAP = {r: CATEGORY_TOGO_MISSING for r in MISSING_REASONS}
CATEGORY_MAP.update({r: CATEGORY_ATTITUDE for r in ATTITUDE_REASONS})
CATEGORY_MAP.update({r: CATEGORY_OTHER for r in OTHER_REASONS})
df["Category"] = df["Reason"].map(CATEGORY_MAP).fillna("Unassigned")

# ───────────────── REASON TOTALS — Missing ─────────────────
st.header("4) Reason totals — To-go Missing Complaints (selected period)")
st.caption("To-Go and Delivery columns shown; Total for “Out of menu item” includes Dine-In as well.")

missing_df = df_all[(df_all["Reason"].isin(MISSING_REASONS)) & (df_all["Period"] == sel_col)]

def _order_series_missing(s: pd.Series) -> pd.Series:
    return s.reindex(MISSING_REASONS)

tot_togo = (
    missing_df[missing_df["Section"] == "To Go"]
      .groupby("Reason", as_index=True)["Value"].sum()
)
tot_delivery = (
    missing_df[missing_df["Section"] == "Delivery"]
      .groupby("Reason", as_index=True)["Value"].sum()
)
tot_dinein = (
    missing_df[missing_df["Section"] == "Dine-In"]
      .groupby("Reason", as_index=True)["Value"].sum()
)

total_series = tot_togo.add(tot_delivery, fill_value=0)
if "Out of menu item" in set(total_series.index).union(set(tot_dinein.index)):
    total_series.loc["Out of menu item"] = (
        float(total_series.get("Out of menu item", 0)) + float(tot_dinein.get("Out of menu item", 0))
    )

reason_totals_missing = pd.DataFrame({
    "To Go": _order_series_missing(tot_togo).fillna(0).astype(int),
    "Delivery": _order_series_missing(tot_delivery).fillna(0).astype(int),
    "Total": _order_series_missing(total_series).fillna(0).astype(int),
})

cat_grand_total_missing = int(reason_totals_missing["Total"].sum())
st.metric("Category Grand Total — To-go Missing Complaints", cat_grand_total_missing)

reason_totals_missing.loc["— Grand Total —"] = reason_totals_missing.sum(numeric_only=True)
st.dataframe(style_table(reason_totals_missing), use_container_width=True)

# ───────────────── REASON TOTALS — Attitude ─────────────────
st.header("4b) Reason totals — Attitude (selected period)")
st.caption("All segments (Dine-In, To Go, Delivery).")

att_df = df_all[(df_all["Reason"].isin(ATTITUDE_REASONS)) & (df_all["Period"] == sel_col)]

def _order_series_att(s: pd.Series) -> pd.Series:
    return s.reindex(ATTITUDE_REASONS)

att_dinein = att_df[att_df["Section"] == "Dine-In"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
att_togo = att_df[att_df["Section"] == "To Go"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
att_delivery = att_df[att_df["Section"] == "Delivery"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
att_total = att_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)

reason_totals_attitude = pd.DataFrame({
    "Dine-In": _order_series_att(att_dinein),
    "To Go": _order_series_att(att_togo),
    "Delivery": _order_series_att(att_delivery),
    "Total": _order_series_att(att_total),
}).fillna(0).astype(int)

cat_grand_total_att = int(reason_totals_attitude["Total"].sum())
st.metric("Category Grand Total — Attitude", cat_grand_total_att)

reason_totals_attitude.loc["— Grand Total —"] = reason_totals_attitude.sum(numeric_only=True)
st.dataframe(style_table(reason_totals_attitude), use_container_width=True)

# ───────────────── REASON TOTALS — Other ─────────────────
st.header("4c) Reason totals — Other (selected period)")
st.caption("All segments (Dine-In, To Go, Delivery).")

oth_df = df_all[(df_all["Reason"].isin(OTHER_REASONS)) & (df_all["Period"] == sel_col)]

def _order_series_other(s: pd.Series) -> pd.Series:
    return s.reindex(OTHER_REASONS)

oth_dinein = oth_df[oth_df["Section"] == "Dine-In"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
oth_togo = oth_df[oth_df["Section"] == "To Go"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
oth_delivery = oth_df[oth_df["Section"] == "Delivery"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
oth_total = oth_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)

reason_totals_other = pd.DataFrame({
    "Dine-In": _order_series_other(oth_dinein),
    "To Go": _order_series_other(oth_togo),
    "Delivery": _order_series_other(oth_delivery),
    "Total": _order_series_other(oth_total),
}).fillna(0).astype(int)

cat_grand_total_other = int(reason_totals_other["Total"].sum())
st.metric("Category Grand Total — Other", cat_grand_total_other)

reason_totals_other.loc["— Grand Total —"] = reason_totals_other.sum(numeric_only=True)
st.dataframe(style_table(reason_totals_other), use_container_width=True)

# ───────────────── 6) Period change summary ─────────────────
st.header("6) Period change summary (vs previous period)")
if not (prior_label := (ordered_headers[ordered_headers.index(sel_col)-1] if sel_col in ordered_headers and ordered_headers.index(sel_col)>0 else None)):
    st.info("No earlier period available to compare against.")
else:
    def totals_by_reason_for(period_label: str, reasons: list[str], allowed_sections: set[str]) -> pd.Series:
        sums = {r: 0 for r in reasons}
        for ad, stores in raw_data.items():
            for store, sections in stores.items():
                for section, reason_map in sections.items():
                    per = reason_map.get("__all__", {})
                    for r in reasons:
                        allowed = SPECIAL_REASON_SECTIONS.get(r, allowed_sections)
                        if section not in allowed:
                            continue
                        sums[r] += int(per.get(r, {}).get(period_label, 0))
        return pd.Series(sums).astype(int)

    missing_sections = {"To Go", "Delivery"}
    attitude_sections = {"Dine-In","To Go","Delivery"}
    other_sections    = {"Dine-In","To Go","Delivery"}

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
    lines.append("To-go Missing Complaints (To-Go + Delivery; ‘Out of menu item’ also includes Dine-In)")
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

# ───────────────── 7) Historical context ─────────────────
st.header("7) Historical context — highs/lows vs all periods (lower = better)")

def build_reason_period_matrix(reasons: list[str], default_sections: set[str]) -> dict[str, dict[str, int]]:
    mat = {p: {r: 0 for r in reasons} for p in ordered_headers}
    for ad, stores in raw_data.items():
        for store, sections_map in stores.items():
            for sec, reason_map in sections_map.items():
                per = reason_map.get("__all__", {})
                for r in reasons:
                    allowed = SPECIAL_REASON_SECTIONS.get(r, default_sections)
                    if sec not in allowed:
                        continue
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
        best_period, best_val = min(series, key=lambda kv: kv[1])
        worst_period, worst_val = max(series, key=lambda kv: kv[1])
        best_vals[r] = best_val
        worst_vals[r] = worst_val
        sorted_asc = sorted(series, key=lambda kv: kv[1])
        rank = next(i + 1 for i, (p, v) in enumerate(sorted_asc) if p == sel_col)
        cur_val = mat[sel_col][r]
        rows.append({
            "Reason": r,
            "Current": cur_val,
            "Rank": f"{rank}/{num_periods}",
            "Best (Period)": f"{best_val} @ {best_period}",
            "Worst (Period)": f"{worst_val} @ {worst_period}",
        })

    df_reasons = pd.DataFrame(rows).set_index("Reason")

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

    totals_by_period = {p: sum(mat[p][r] for r in reasons) for p in ordered_headers}
    best_p, best_total   = min(totals_by_period.items(), key=lambda kv: kv[1])
    worst_p, worst_total = max(totals_by_period.items(), key=lambda kv: kv[1])
    current_total = totals_by_period.get(sel_col, 0)
    rank_list = sorted(totals_by_period.items(), key=lambda kv: kv[1])
    current_rank_idx = next(i + 1 for i, (p, v) in enumerate(rank_list) if p == sel_col)

    current_cls = " best" if current_total == best_total else (" worst" if current_total == worst_total else "")

    card_css = """
<style>
.kpi-wrap{display:flex;gap:18px;margin:6px 0 8px 0}
.kpi{flex:1;background:#1113;border:2px solid #2a2f36;border-radius:12px;padding:14px 16px;text-align:center}
.kpi.best{border-color:#4CAF50}
.kpi.worst{border-color:#E53935}
.kpi h4{margin:0 0 6px 0;font-weight:600;font-size:0.95rem;color:#cfd8e3}
.kpi .num{font-size:2.6rem;line-height:1.1;font-weight:700;color:#fff;margin:0}
.kpi .sub{margin-top:4px;color:#9fb3c8;font-size:.95rem}
@media (max-width:900px){.kpi-wrap{flex-direction:column}}
</style>
"""
    st.markdown(card_css, unsafe_allow_html=True)

    kpi_html = f"""
<div class="kpi-wrap">
  <div class="kpi{current_cls}">
    <h4>Current total (lower is better)</h4>
    <div class="num">{current_total}</div>
    <div class="sub">{sel_col}</div>
  </div>
  <div class="kpi best">
    <h4>Best total (lowest)</h4>
    <div class="num">{best_total}</div>
    <div class="sub">{best_p}</div>
  </div>
  <div class="kpi worst">
    <h4>Worst total (highest)</h4>
    <div class="num">{worst_total}</div>
    <div class="sub">{worst_p}</div>
  </div>
</div>
"""
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.caption(f"Current period rank: {current_rank_idx}/{num_periods} (1 = lowest/best)")

    sty = style_table(df_reasons, highlight_grand_total=False).apply(highlight_current, subset=["Current"])
    st.dataframe(sty, use_container_width=True)

build_highlow_tables(MISSING_REASONS, {"To Go","Delivery"}, "7a) To-go Missing Complaints (To-Go + Delivery) — highs/lows")
build_highlow_tables(ATTITUDE_REASONS, {"Dine-In","To Go","Delivery"}, "7b) Attitude (All segments) — highs/lows")
build_highlow_tables(OTHER_REASONS, {"Dine-In","To Go","Delivery"}, "7c) Other (All segments) — highs/lows")

# ───────────────── 8) EXPORT — Excel (All Sheets + QA) ─────────────────
st.header("8) Export results")

buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    df_detail.to_excel(writer, index=False, sheet_name="Detail (Selected Period)")
    ad_totals.to_excel(writer, index=False, sheet_name="AD Totals (Selected)")
    store_totals.to_excel(writer, index=False, sheet_name="Store Totals (Selected)")
    reason_totals_missing.to_excel(writer, sheet_name="Reason Totals (Missing)")
    reason_totals_attitude.to_excel(writer, sheet_name="Reason Totals (Attitude)")
    reason_totals_other.to_excel(writer, sheet_name="Reason Totals (Other)")

st.download_button(
    "📥 Download Excel (All Sheets)",
    data=buff.getvalue(),
    file_name=f"ad_store_{sel_col.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

qa = io.BytesIO()
with pd.ExcelWriter(qa, engine="openpyxl") as writer:
    if debug_mode and debug_log.get("raw_layout"):
        pd.DataFrame(debug_log["raw_layout"]).to_excel(writer, index=False, sheet_name="Raw Layout (Audit)")
    facts_df = pd.DataFrame(debug_log["facts"]) if debug_log.get("facts") else df_all[
        ["Area Director","Store","Section","Reason","Period","Value"]
    ]
    facts_df.to_excel(writer, index=False, sheet_name="Normalized Facts")

st.download_button(
    "📥 Download QA Workbook (Audit + Facts)",
    data=qa.getvalue(),
    file_name=f"qa_{sel_col.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ───────────────── DEBUG PANEL ─────────────────
if debug_mode:
    st.header("🧠 Troubleshooting / Debug Output")

    with st.expander("Unmatched Labels (not recognized)"):
        if debug_log["unmatched_labels"]:
            st.dataframe(pd.DataFrame(debug_log["unmatched_labels"]))
        else:
            st.success("✅ All labels matched known metrics.")

    with st.expander("Ignored Tokens (outside period columns or ≥ TOTAL)"):
        if debug_log["ignored_tokens"]:
            st.dataframe(pd.DataFrame(debug_log["ignored_tokens"]))
        else:
            st.success("✅ No tokens were ignored for being out of range.")

    with st.expander("Token Trace (raw matches)"):
        trace_df = pd.DataFrame(debug_log["token_trace"])
        if trace_df.empty:
            st.info("No tokens traced yet.")
        else:
            col_reason = st.selectbox("Filter by reason:", sorted(trace_df["reason"].unique()))
            col_period = st.selectbox("Filter by period:", sorted(trace_df["period"].unique()))
            filtered = trace_df[(trace_df["reason"] == col_reason) & (trace_df["period"] == col_period)]
            st.dataframe(filtered)

    with st.expander("Header/Bin Diagnostics"):
        hb = debug_log.get("header_bins", [])
        if not hb:
            st.info("No header/bin data captured yet.")
        else:
            rows = []
            for rec in hb:
                page = rec["page"]
                total_x = rec["total_x"]
                for b in rec["bins"]:
                    rows.append({
                        "page": page,
                        "period": b["period"],
                        "left": round(b["left"], 1),
                        "right": round(b["right"], 1),
                        "total_x": round(total_x, 1) if total_x is not None else None,
                    })
            st.dataframe(pd.DataFrame(rows).sort_values(["page","period"]))
