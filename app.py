# file: app.py
# Greeno Big Three v1.9.0 — Pure Count + Normal modes
# This version maps raw labels like "Missing Item (Food)" → category "Missing food"

import io, os, re, base64, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Pattern
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── STYLING ─────────────────
def style_table(df: pd.DataFrame, highlight_grand_total: bool = True) -> "pd.io.formats.style.Styler":
    def zebra(series):
        return [
            "background-color: #F5F7FA" if i % 2 == 0 else "background-color: #E9EDF2"
            for i, _ in enumerate(series)
        ]
    sty = (
        df.style
        .set_properties(**{
            "color": "#111", "border-color": "#CCD3DB",
            "border-width": "0.5px", "border-style": "solid",
        })
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
st.set_page_config(page_title="Greeno Big Three v1.9.0", layout="wide")

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
      <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.9.0</h1>
      <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ───────────────── SIDEBAR ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose the PDF report", type=["pdf"])
    st.caption("Missing = To-Go/Delivery (except ‘Out of menu item’ includes Dine-In). Attitude/Other = all segments.")
    st.divider()
    pure_mode = st.toggle(
        "✅ Pure Count Mode (ignore AD/Store/Segment)",
        value=False,
        help="Counts directly by reason × period only. Great for sanity checks."
    )
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
SPECIAL_REASON_SECTIONS = {"Out of menu item": {"To Go", "Delivery", "Dine-In"}}
COMP_CANON = "No/insufficient compensation offered"

# ───────────────── NORMALIZATION / MATCHING ─────────────────
def _lc(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

# Ordered regex → canonical reason (so “food” doesn’t get stolen by “bev”, etc.)
CANON_REGEX: List[Tuple[str, Tuple[Pattern, ...]]] = [
    # To-go Missing
    ("Missing food", (
        re.compile(r"\bmissing\s+item\s*\(\s*food\s*\)\b"),
        re.compile(r"\bmissing\s+food\b"),
    )),
    ("Order wrong", (
        re.compile(r"\border\s+wrong\b"),
        re.compile(r"\bwrong\s+order\b"),
    )),
    ("Missing condiments", (
        re.compile(r"\bmissing\s+condiments?\b"),
    )),
    ("Out of menu item", (
        re.compile(r"\bout\s+of\s+menu\s+item\b"),
    )),
    ("Missing bev", (
        re.compile(r"\bmissing\s+item\s*\(\s*bev[^\)]*\)\b"),
        re.compile(r"\bmissing\s+bev(?:erage)?\b"),
    )),
    ("Missing ingredients", (
        re.compile(r"\bmissing\s+ingredient"),
    )),
    ("Packaging to-go complaint", (
        re.compile(r"\bpackaging\s+to\s*[- ]?go\s+complaint\b"),
        re.compile(r"\bpackaging\b"),
    )),

    # Attitude
    ("Unprofessional/Unfriendly", (
        re.compile(r"\bunprofessional\b"),
        re.compile(r"\bunfriendly\b"),
        re.compile(r"\bunprofessional\s+behavior\b"),
    )),
    ("Manager directly involved", (
        re.compile(r"\bmanager\s+directly\s+involved\b"),
        re.compile(r"\bdirectly\s+involved\b"),
    )),
    ("Manager not available", (
        re.compile(r"\bmanager\s+not\s+available\b"),
        re.compile(r"\bmanagement\s+not\s+available\b"),
    )),
    ("Manager did not visit", (
        re.compile(r"\bno\s+visit\b"),
        re.compile(r"\bdid\s+not\s+visit\b"),
        re.compile(r"\bdidn[’']?t\s+visit\b"),
    )),
    ("Negative mgr-employee exchange", (
        re.compile(r"\bmanager[- ]employee\b"),
        re.compile(r"\bnegative\s+(?:manager|mgr)[- ]employee"),
    )),
    ("Manager did not follow up", (
        re.compile(r"\bfollow[- ]?up\b"),
    )),
    ("Argued with guest", (
        re.compile(r"\bargued\b"),
        re.compile(r"\bargued\s+with\s+guest\b"),
    )),

    # Other
    ("Long hold/no answer", (
        re.compile(r"\blong\s+hold\b"),
        re.compile(r"\bno\s+answer\b"),
        re.compile(r"\bhung\s+up\b"),
        re.compile(r"\bhang\s+up\b"),
    )),
    (COMP_CANON, (
        re.compile(r"\bcompensation\b"),
        re.compile(r"\bno/unsatisfactory\b"),
        re.compile(r"\boffered\s+by\s+restaurant\b"),
    )),
    ("Did not attempt to resolve", (
        re.compile(r"\bdid\s+not\s+attempt\s+to\s+resolve\b"),
        re.compile(r"\battempt\s+to\s+resolve\b"),
        re.compile(r"\bdid\s+not\s+resolve\b"),
        re.compile(r"\bresolve\s+issue\b"),
        re.compile(r"\bresolve\b"),
    )),
    ("Guest left without ordering", (
        re.compile(r"\bguest\s+left\b"),
        re.compile(r"\bwithout\s+ordering\b"),
    )),
    ("Unknowledgeable", (
        re.compile(r"\bunknowledgeable\b"),
    )),
    ("Did not open on time", (
        re.compile(r"\bopen/close\s+on\s+time\b"),
        re.compile(r"\bopen\s+on\s+time\b"),
        re.compile(r"\bdidn[’']?t\s+open\b"),
    )),
    ("No/poor apology", (
        re.compile(r"\bapology\b"),
        re.compile(r"\bno\s+apology\b"),
        re.compile(r"\bpoor\s+apology\b"),
    )),
]

def get_canonical_reason(label_text: str) -> Optional[str]:
    txt = _lc(label_text)
    if txt.endswith(" total:") or txt in {"dine-in total:", "to go total:", "delivery total:", "total:"}:
        return None
    for canon, patterns in CANON_REGEX:
        for pat in patterns:
            if pat.search(txt):
                return canon
    return None

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
    headers = [h for h, _ in items]; xs = [x for _, x in items]
    med_gap = statistics.median([xs[i+1]-xs[i] for i in range(len(xs)-1)]) if len(xs) >= 2 else 60.0
    bins = []
    for i, (h, x) in enumerate(zip(headers, xs)):
        left = (xs[i-1] + x)/2 if i > 0 else x - 0.5*med_gap
        right = (x + xs[i+1])/2 if i < len(xs) - 1 else ((x + total_x)/2 if total_x is not None else x + 0.6*med_gap)
        bins.append((h, left-2, right+2))  # slight pad
    return bins

def map_x_to_header(header_bins: List[Tuple[str, float, float]], xmid: float) -> Optional[str]:
    for h, left, right in header_bins:
        if left <= xmid < right:
            return h
    return None

# ───────────────── LINE GROUPING ─────────────────
def extract_words_grouped(page):
    words = page.extract_words(x_tolerance=1.4, y_tolerance=2.4, keep_blank_chars=False, use_text_flow=True)
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

# ───────────────── DEBUG ROWS NORMALIZER ─────────────────
def _dbg_bins_to_rows(rec: dict) -> List[dict]:
    rows = []
    page = rec.get("page"); total_x = rec.get("total_x")
    for b in rec.get("bins", []):
        rows.append({
            "page": page, "period": b["period"],
            "left": b["left"], "right": b["right"], "total_x": total_x,
        })
    return rows

# ───────────────── PURE COUNT PARSER ─────────────────
def parse_pdf_pure_counts(file_bytes: bytes, debug: bool = False):
    counts = defaultdict(lambda: defaultdict(int))
    ordered_headers: List[str] = []
    debug_log = {"token_trace": [], "ignored_tokens": [], "header_bins": []}

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        header_positions: Dict[str, float] = {}
        carry_headers = None; carry_total_x = None

        for page in pdf.pages:
            headers_once = find_period_headers(page)
            headers = headers_once or carry_headers
            if not headers: continue
            if headers_once:
                carry_headers = headers[:]; carry_total_x = None

            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)

            total_x = find_total_header_x(page, header_y) or carry_total_x
            if total_x is not None: carry_total_x = total_x

            header_bins = build_header_bins({h: header_positions[h] for h in ordered_headers}, total_x)

            if debug:
                debug_log["header_bins"].append({
                    "page": page.page_number,
                    "header_positions": {h: header_positions[h] for h in ordered_headers},
                    "total_x": total_x,
                    "bins": [{"period": h, "left": left, "right": right} for (h, left, right) in header_bins],
                })

            first_period_x = min(header_positions[h] for h in ordered_headers)
            label_right_edge = first_period_x - 10
            lines = extract_words_grouped(page)
            if not lines: continue

            def left_label_text(line):
                return " ".join(
                    w["text"].strip()
                    for w in line["words"]
                    if w["x1"] <= label_right_edge and w["text"].strip()
                ).strip()

            def consume(line_obj, canon_reason: str) -> int:
                y_band = line_obj["y"]; got = 0
                for w in line_obj["words"]:
                    token = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", token): continue
                    if w["x0"] <= label_right_edge: continue
                    w_y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
                    if abs(w_y_mid - y_band) > 0.01: continue
                    xmid = (w["x0"] + w["x1"]) / 2
                    if total_x is not None and xmid >= (total_x - 1.0):
                        if debug:
                            debug_log["ignored_tokens"].append({
                                "page": page.page_number, "token": token, "xmid": xmid,
                                "reason": f"{canon_reason} (>= TOTAL cutoff)",
                            })
                        continue
                    mapped = map_x_to_header(header_bins, xmid)
                    if mapped is None or mapped not in ordered_headers:
                        if debug:
                            debug_log["ignored_tokens"].append({
                                "page": page.page_number, "token": token, "xmid": xmid,
                                "reason": f"{canon_reason} (no header bin)",
                            })
                        continue
                    counts[canon_reason][mapped] += int(token)
                    got += 1
                    if debug:
                        debug_log["token_trace"].append({
                            "page": page.page_number, "reason": canon_reason,
                            "period": mapped, "value": int(token),
                        })
                return got

            i = 0
            while i < len(lines):
                L = lines[i]
                label_text = left_label_text(L)
                label_lc = _lc(label_text)

                # compensation multi-line capture
                if ("no/unsatisfactory" in label_lc or "compensation offered by" in label_lc or
                    label_lc.strip() == "restaurant" or "compensation" in label_lc):
                    canon = COMP_CANON
                    consume(L, canon)
                    if i > 0: consume(lines[i-1], canon)
                    if i + 1 < len(lines): consume(lines[i+1], canon)
                    i += 1
                    continue

                canon = get_canonical_reason(label_text)
                if not canon:
                    i += 1
                    continue

                got = consume(L, canon)
                if got == 0:
                    if i > 0: got += consume(lines[i-1], canon)
                    if got == 0 and i + 1 < len(lines): got += consume(lines[i+1], canon)

                i += 1

    return counts, ordered_headers, debug_log

# ───────────────── FULL PARSER ─────────────────
def parse_pdf_full(file_bytes: bytes, debug: bool = False):
    header_positions: Dict[str, float] = {}
    ordered_headers: List[str] = []
    pairs_debug: List[Tuple[str, str]] = []
    data: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    debug_log = {
        "unmatched_labels": [], "ignored_tokens": [], "token_trace": [],
        "events": [], "header_bins": [], "raw_layout": [], "facts": [],
    }

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        carry_headers = None; carry_total_x = None
        for page in pdf.pages:
            headers_once = find_period_headers(page)
            headers = headers_once or carry_headers
            if not headers: continue
            if headers_once:
                carry_headers = headers[:]; carry_total_x = None
            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)

            total_x = find_total_header_x(page, header_y) or carry_total_x
            if total_x is not None: carry_total_x = total_x
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
            if not lines: continue

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
                    if not re.fullmatch(r"-?\d+", token): continue
                    if w["x0"] <= label_right_edge: continue
                    w_y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
                    if abs(w_y_mid - y_band) > 0.01: continue
                    xmid = (w["x0"] + w["x1"]) / 2
                    if total_x is not None and xmid >= (total_x - 1.0):
                        if debug:
                            debug_log["ignored_tokens"].append({
                                "page": page.page_number, "token": token, "xmid": xmid,
                                "reason": f"{canon_reason} (>= TOTAL cutoff)",
                                "store": current_store, "section": current_section,
                            })
                        continue
                    mapped = map_x_to_header(header_bins, xmid)
                    if mapped is None or mapped not in ordered_headers:
                        if debug:
                            debug_log["ignored_tokens"].append({
                                "page": page.page_number, "token": token, "xmid": xmid,
                                "reason": f"{canon_reason} (no header bin)",
                                "store": current_store, "section": current_section,
                            })
                        continue
                    sect = data[current_ad].setdefault(current_store, {}).setdefault(current_section, {})
                    per_header = sect.setdefault("__all__", defaultdict(lambda: defaultdict(int)))
                    per_header[canon_reason][mapped] += int(token)

                    if debug:
                        debug_log["token_trace"].append({
                            "page": page.page_number, "ad": current_ad, "store": current_store,
                            "section": current_section, "reason": canon_reason,
                            "period": mapped, "value": int(token),
                        })
                        debug_log["facts"].append({
                            "Area Director": current_ad, "Store": current_store, "Section": current_section,
                            "Reason": canon_reason, "Period": mapped, "Value": int(token),
                        })

            def consume_and_count(line_obj, canon_reason: str) -> int:
                if debug:
                    before = len(debug_log["token_trace"]); consume_words(line_obj, canon_reason)
                    after = len(debug_log["token_trace"]); return max(0, after - before)
                else:
                    cnt = 0; y_band = line_obj["y"]
                    for w in line_obj["words"]:
                        token = w["text"].strip()
                        if not re.fullmatch(r"-?\d+", token): continue
                        if w["x0"] <= label_right_edge: continue
                        w_y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
                        if abs(w_y_mid - y_band) <= 0.01:
                            cnt += 1
                    consume_words(line_obj, canon_reason)
                    return cnt

            def find_ad_for_store(lines: List[dict], store_idx: int, left_margin: float, back_limit: int = 12) -> Optional[str]:
                def is_left_aligned(x): return (x - left_margin) <= 24
                for j in range(store_idx - 1, max(store_idx - back_limit, -1), -1):
                    cand = lines[j]; s = cand["text"].strip()
                    if is_left_aligned(cand["x_min"]) and looks_like_name(s):
                        return s
                for j in range(store_idx - back_limit - 1, -1, -1):
                    cand = lines[j]; s = cand["text"].strip()
                    if looks_like_name(s): return s
                return None

            idx = 0
            while idx < len(lines):
                L = lines[idx]; txt = L["text"].strip()

                if debug:
                    for w in L["words"]:
                        debug_log["raw_layout"].append({
                            "page": page.page_number, "y": L["y"], "x0": w["x0"], "x1": w["x1"],
                            "xmid": (w["x0"] + w["x1"]) / 2, "text": w["text"],
                            "ad": current_ad, "store": current_store, "section": current_section,
                        })

                if STORE_LINE_RX.match(txt):
                    ad_for_this_store = find_ad_for_store(lines, idx, min(L["x_min"] for L in lines))
                    if ad_for_this_store: current_ad = ad_for_this_store
                    current_store = txt; current_section = None
                    if current_ad: pairs_debug.append((current_ad, current_store))
                    idx += 1; continue

                if SECTION_TOGO.match(txt):   current_section = "To Go";   idx += 1; continue
                if SECTION_DELIV.match(txt):  current_section = "Delivery"; idx += 1; continue
                if SECTION_DINEIN.match(txt): current_section = "Dine-In";  idx += 1; continue

                if txt in HEADINGS: idx += 1; continue
                if not (current_ad and current_store and current_section in {"To Go", "Delivery", "Dine-In"}):
                    idx += 1; continue

                label_text = " ".join(
                    w["text"].strip() for w in L["words"]
                    if w["x1"] <= (min(header_positions[h] for h in ordered_headers) - 10) and w["text"].strip()
                )
                label_lc = _lc(label_text)

                if is_structural_total(label_lc):
                    idx += 1; continue

                # compensation special-case (still necessary)
                if (COMP_CANON == get_canonical_reason(label_text)) or (
                    "no/unsatisfactory" in label_lc or "compensation offered by" in label_lc or
                    label_lc.strip() == "restaurant" or "compensation" in label_lc
                ):
                    canon = COMP_CANON
                    consume_words(L, canon)
                    if idx > 0:
                        prev_lc = _lc(" ".join(
                            w["text"].strip() for w in lines[idx - 1]["words"]
                            if w["x1"] <= (min(header_positions[h] for h in ordered_headers) - 10) and w["text"].strip()
                        ))
                        if ("no/unsatisfactory" in prev_lc or "compensation offered by" in prev_lc or
                            prev_lc.strip() == "restaurant" or "compensation" in prev_lc):
                            consume_words(lines[idx - 1], canon)
                    if (idx + 1) < len(lines):
                        next_lc = _lc(" ".join(
                            w["text"].strip() for w in lines[idx + 1]["words"]
                            if w["x1"] <= (min(header_positions[h] for h in ordered_headers) - 10) and w["text"].strip()
                        ))
                        if ("no/unsatisfactory" in next_lc or "compensation offered by" in next_lc or
                            next_lc.strip() == "restaurant" or "compensation" in next_lc):
                            consume_words(lines[idx + 1], canon)
                    idx += 1
                    continue

                canon = get_canonical_reason(label_text)
                if not canon:
                    if debug:
                        debug_log["unmatched_labels"].append({
                            "page": page.page_number, "text": label_text,
                            "ad": current_ad, "store": current_store, "section": current_section,
                        })
                    idx += 1
                    continue

                got = consume_and_count(L, canon)
                if got == 0:
                    if idx > 0:
                        prev_L = lines[idx - 1]
                        prev_label_lc = _lc(" ".join(
                            w["text"].strip() for w in prev_L["words"]
                            if w["x1"] <= (min(header_positions[h] for h in ordered_headers) - 10) and w["text"].strip()
                        ))
                        if not is_structural_total(prev_label_lc):
                            got += consume_and_count(prev_L, canon)
                    if got == 0 and (idx + 1) < len(lines):
                        next_L = lines[idx + 1]
                        next_label_lc = _lc(" ".join(
                            w["text"].strip() for w in next_L["words"]
                            if w["x1"] <= (min(header_positions[h] for h in ordered_headers) - 10) and w["text"].strip()
                        ))
                        if not is_structural_total(next_label_lc):
                            got += consume_and_count(next_L, canon)

                    if debug and got > 0:
                        debug_log["events"].append({
                            "type": "adjacent_capture", "reason": canon, "page": page.page_number,
                            "ad": current_ad, "store": current_store, "section": current_section,
                            "note": "numbers on adjacent line due to wrap"
                        })
                idx += 1

    return header_positions, data, sort_headers(list(header_positions.keys())), pairs_debug, debug_log

# ───────────────── RUN ─────────────────
with st.spinner("Roll Tide…"):
    if pure_mode:
        counts_pure, ordered_headers, debug_log = parse_pdf_pure_counts(file_bytes, debug=debug_mode)
        raw_data = None
    else:
        header_x_map, raw_data, ordered_headers, pairs_debug, debug_log = parse_pdf_full(file_bytes, debug=debug_mode)

if not ordered_headers:
    st.error("No period headers (like ‘P9 25’) found.")
    st.stop()

# ───────────────── PERIOD PICKER ─────────────────
st.header("2) Pick the period")
sel_col = st.selectbox("Period", options=ordered_headers, index=len(ordered_headers)-1)

# ───────────────── PURE COUNT MODE UI ─────────────────
if pure_mode:
    st.header("3) Pure Count — reason × period (ignores AD/Store/Segment)")

    periods = ordered_headers
    df_pure = pd.DataFrame(index=ALL_CANONICAL, columns=periods).fillna(0).astype(int)
    for reason, per_map in counts_pure.items():
        for p, v in per_map.items():
            if reason in df_pure.index and p in df_pure.columns:
                df_pure.loc[reason, p] = int(v)
    df_pure["Total"] = df_pure.sum(axis=1)

    def cat_total(reasons): return int(df_pure.loc[reasons, sel_col].sum())

    tot_missing = cat_total(MISSING_REASONS)
    tot_att     = cat_total(ATTITUDE_REASONS)
    tot_other   = cat_total(OTHER_REASONS)
    overall     = tot_missing + tot_att + tot_other

    score_css = """
    <style>
    .score-wrap{display:flex;gap:16px;margin:10px 0 8px 0}
    .score{flex:1;background:#1113;border:2px solid #38414a;border-radius:14px;padding:18px 20px;text-align:center}
    .score h4{margin:0 0 8px 0;font-weight:700;font-size:1.15rem;color:#cfd8e3}
    .score .num{font-size:3rem;line-height:1.1;font-weight:800;color:#fff;margin:2px 0 2px}
    @media (max-width:900px){.score-wrap{flex-direction:column}}
    </style>
    """
    st.markdown(score_css, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="score-wrap">
          <div class="score"><h4>Overall (selected period)</h4><div class="num">{overall}</div></div>
          <div class="score"><h4>To-go Missing Complaints</h4><div class="num">{tot_missing}</div></div>
          <div class="score"><h4>Attitude</h4><div class="num">{tot_att}</div></div>
          <div class="score"><h4>Other</h4><div class="num">{tot_other}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Reason × Period counts")
    st.dataframe(style_table(df_pure), use_container_width=True)

    try:
        total_resolve = int(df_pure.loc["Did not attempt to resolve", periods].sum())
        sel_resolve   = int(df_pure.loc["Did not attempt to resolve", sel_col])
        st.caption(f"Did not attempt to resolve — total across PDF: {total_resolve} • Selected period ({sel_col}): {sel_resolve}")
    except Exception:
        pass

    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df_pure.to_excel(writer, sheet_name="Pure Count (All Periods)")
        roll = pd.DataFrame({
            "Category": ["To-go Missing Complaints","Attitude","Other","Overall"],
            **{p: [
                int(df_pure.loc[MISSING_REASONS, p].sum()),
                int(df_pure.loc[ATTITUDE_REASONS, p].sum()),
                int(df_pure.loc[OTHER_REASONS, p].sum()),
                int(df_pure[p].sum())
            ] for p in periods}
        })
        roll["Total"] = roll[periods].sum(axis=1)
        roll.to_excel(writer, sheet_name="Category Rollups", index=False)
        if debug_mode:
            pd.DataFrame(debug_log.get("token_trace", [])).to_excel(writer, sheet_name="Token Trace", index=False)
            pd.DataFrame(debug_log.get("ignored_tokens", [])).to_excel(writer, sheet_name="Ignored Tokens", index=False)
            hb = debug_log.get("header_bins", [])
            if hb:
                rows = []
                for rec in hb:
                    rows.extend(_dbg_bins_to_rows(rec))
                pd.DataFrame(rows).to_excel(writer, sheet_name="Header Bins", index=False)

    st.download_button(
        "📥 Download Excel (Pure Count)",
        data=buff.getvalue(),
        file_name=f"pure_count_{sel_col.replace(' ','_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.stop()

# ───────────────── NORMAL MODE (existing app flow) ─────────────────
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

# For export only
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

CATEGORY_TOGO_MISSING = "To-go Missing Complaints"
CATEGORY_ATTITUDE     = "Attitude"
CATEGORY_OTHER        = "Other"
CATEGORY_MAP = {r: CATEGORY_TOGO_MISSING for r in MISSING_REASONS}
CATEGORY_MAP.update({r: CATEGORY_ATTITUDE for r in ATTITUDE_REASONS})
CATEGORY_MAP.update({r: CATEGORY_OTHER for r in OTHER_REASONS})
df["Category"] = df["Reason"].map(CATEGORY_MAP).fillna("Unassigned")

# Quick glance vs previous
st.markdown("### Quick glance")
try:
    cur_idx = ordered_headers.index(sel_col)
    prior_label = ordered_headers[cur_idx - 1] if cur_idx > 0 else None
except ValueError:
    prior_label = None

def _allowed_sections_for_reason(reason: str, default_sections: set[str]) -> set[str]:
    return SPECIAL_REASON_SECTIONS.get(reason, default_sections)

def _total_for(period_label: Optional[str], reasons: list[str], default_sections: set[str]) -> int:
    if not period_label: return 0
    total = 0
    for ad, stores in raw_data.items():
        for store, sects in stores.items():
            for sec_name, reason_map in sects.items():
                per = reason_map.get("__all__", {})
                for r in reasons:
                    allowed = _allowed_sections_for_reason(r, default_sections)
                    if sec_name not in allowed: continue
                    total += int(per.get(r, {}).get(period_label, 0))
    return int(total)

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

def diff_val(cur, prv, has_prior):  return (cur - prv) if has_prior else None
def fmt_diff(d):                    return "n/a" if d is None else f"{d:+d}"
def cls_from_delta(d):              return "" if d is None else (" best" if d < 0 else (" worst" if d > 0 else ""))

overall_diff = diff_val(overall_cur, overall_prv, prior_label is not None)
miss_diff    = diff_val(tot_missing_cur, tot_missing_prv, prior_label is not None)
att_diff     = diff_val(tot_att_cur,     tot_att_prv,     prior_label is not None)
oth_diff     = diff_val(tot_other_cur,   tot_other_prv,   prior_label is not None)

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
  <div class="score{cls_from_delta(overall_diff)}">
    <h4>Overall (all categories)</h4>
    <div class="num">{overall_cur}</div>
    <div class="delta {diff_class(overall_diff)}">{fmt_diff(overall_diff)}<span class="vs">vs {prior_text}</span></div>
  </div>
  <div class="score{cls_from_delta(miss_diff)}">
    <h4>To-go Missing Complaints</h4>
    <div class="num">{tot_missing_cur}</div>
    <div class="delta {diff_class(miss_diff)}">{fmt_diff(miss_diff)}<span class="vs">vs {prior_text}</span></div>
  </div>
  <div class="score{cls_from_delta(att_diff)}">
    <h4>Attitude</h4>
    <div class="num">{tot_att_cur}</div>
    <div class="delta {diff_class(att_diff)}">{fmt_diff(att_diff)}<span class="vs">vs {prior_text}</span></div>
  </div>
  <div class="score{cls_from_delta(oth_diff)}">
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

# ───────────────── REASON TOTALS — Missing ─────────────────
st.header("4) Reason totals — To-go Missing Complaints (selected period)")
st.caption("To-Go and Delivery columns shown; Total for “Out of menu item” includes Dine-In as well.")

missing_df = df_all[(df_all["Reason"].isin(MISSING_REASONS)) & (df_all["Period"] == sel_col)]

def _order_series_missing(s: pd.Series) -> pd.Series:
    return s.reindex(MISSING_REASONS)

tot_togo = (missing_df[missing_df["Section"] == "To Go"].groupby("Reason", as_index=True)["Value"].sum())
tot_delivery = (missing_df[missing_df["Section"] == "Delivery"].groupby("Reason", as_index=True)["Value"].sum())
tot_dinein = (missing_df[missing_df["Section"] == "Dine-In"].groupby("Reason", as_index=True)["Value"].sum())

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

att_dinein  = att_df[att_df["Section"] == "Dine-In"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
att_togo    = att_df[att_df["Section"] == "To Go"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
att_delivery= att_df[att_df["Section"] == "Delivery"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
att_total   = att_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)

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

oth_dinein  = oth_df[oth_df["Section"] == "Dine-In"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
oth_togo    = oth_df[oth_df["Section"] == "To Go"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
oth_delivery= oth_df[oth_df["Section"] == "Delivery"].groupby("Reason", as_index=True)["Value"].sum().astype(int)
oth_total   = oth_df.groupby("Reason", as_index=True)["Value"].sum().astype(int)

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

# ───────────────── EXPORTS ─────────────────
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
    facts_df = pd.DataFrame(debug_log["facts"]) if debug_mode and debug_log.get("facts") else df_all[
        ["Area Director","Store","Section","Reason","Period","Value"]
    ]
    facts_df.to_excel(writer, index=False, sheet_name="Normalized Facts")
    if debug_mode:
        pd.DataFrame(debug_log.get("ignored_tokens", [])).to_excel(writer, index=False, sheet_name="Ignored Tokens")
        hb = debug_log.get("header_bins", [])
        if hb:
            rows = []
            for rec in hb:
                rows.extend(_dbg_bins_to_rows(rec))
            pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Header Bins")
st.download_button(
    "📥 Download QA Workbook (Audit + Facts)",
    data=qa.getvalue(),
    file_name=f"qa_{sel_col.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
