# Greeno Big Three v1.9.3 — Deep Stitch
# - Robust line stitching:
#   * Merge multi-line labels (2+ lines) before numbers
#   * Cross-page stitching: carry dangling label to next page
# - Wider label zone tolerance
# - Reason x Period rollups + category totals
# - Debug exports to validate stitching & header bins

import io, os, re, base64, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── STYLING ─────────────────
def style_table(df: pd.DataFrame, highlight_grand_total: bool = True):
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
    if highlight_grand_total and "— Grand Total —" in df.index.astype(str):
        def highlight_total(row):
            if str(row.name) == "— Grand Total —":
                return ["background-color: #FFE39B; color: #111; font-weight: 700;"] * len(row)
            return [""] * len(row)
        sty = sty.apply(highlight_total, axis=1)
    sty = sty.set_table_styles(
        [{"selector": "th.row_heading, th.blank", "props": [("color", "#111"), ("border-color", "#CCD3DB")]}]
    )
    return sty

# ───────────────── PAGE HEADER / THEME ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.9.3 — Deep Stitch", layout="wide")

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
      <h1 style="margin:0; font-size:2.2rem;">Greeno Big Three v1.9.3 — Deep Stitch</h1>
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
    st.caption("Deep Stitch merges multi-line labels and attaches numeric rows; also stitches across page breaks.")
    st.divider()
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

if pdfplumber is None:
    st.error("pdfplumber is not installed. Add `pdfplumber` to requirements.txt")
    st.stop()

file_bytes = up.read()

# ───────────────── CONSTANTS ─────────────────
HEADER_RX = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

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

# Strict anchors (preferred)
KEYWORD_REGEX = {
    "Missing food":               re.compile(r"\bmissing\s+item\s*\(food\)", re.I),
    "Missing bev":                re.compile(r"\bmissing\s+item\s*\(bev\)",  re.I),
    "Missing condiments":         re.compile(r"\bmissing\s+condiments?",     re.I),
    "Missing ingredients":        re.compile(r"\bmissing\s+ingredient",      re.I),
    "Out of menu item":           re.compile(r"\bout\s+of\s+menu\s+item",    re.I),
    "Packaging to-go complaint":  re.compile(r"\bpackaging\s+to-?\s*go",     re.I),
}
# Safe substrings (fallback)
KEYWORD_SUBSTR = {
    "Order wrong":                          ["order wrong"],
    "Unprofessional/Unfriendly":            ["unfriendly"],
    "Manager directly involved":            ["directly involved"],
    "Manager not available":                ["manager not available"],
    "Manager did not visit":                ["did not visit", "no visit"],
    "Negative mgr-employee exchange":       ["manager-employee"],
    "Manager did not follow up":            ["follow up"],
    "Argued with guest":                    ["argued"],
    "Long hold/no answer":                  ["hung up", "long hold", "no answer"],
    "No/insufficient compensation offered": ["compensation", "no/unsatisfactory"],
    "Did not attempt to resolve":           ["resolve"],
    "Guest left without ordering":          ["without ordering"],
    "Unknowledgeable":                      ["unknowledgeable"],
    "Did not open on time":                 ["open on time"],
    "No/poor apology":                      ["apology"],
}
COMP_CANON = "No/insufficient compensation offered"

# ───────────────── UTILS ─────────────────
def _lc(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

def sort_headers(headers: List[str]) -> List[str]:
    def key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999, 999)
    return sorted(headers, key=key)

def is_structural_total(label_text_lc: str) -> bool:
    return (
        label_text_lc.endswith(" total:") or
        label_text_lc == "dine-in total:" or
        label_text_lc == "to go total:" or
        label_text_lc == "delivery total:" or
        label_text_lc == "total:"
    )

# ───────────────── HEADER FINDERS ─────────────────
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

def find_total_header_x(page, header_y: float) -> Optional[float]:
    words = page.extract_words(x_tolerance=1.0, y_tolerance=2.0, keep_blank_chars=False, use_text_flow=True)
    for w in words:
        y_mid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        if abs(y_mid - header_y) <= 2.8 and w["text"].strip().lower() == "total":
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
        left = (xs[i-1] + x)/2 if i > 0 else x - 0.55*med_gap
        if i < len(xs) - 1:
            right = (x + xs[i+1])/2
        else:
            right = (x + (total_x if total_x is not None else (x + 0.6*med_gap)))/2
        bins.append((h, left-3, right+3))  # widen 3px
    return bins

def map_x_to_header(header_bins: List[Tuple[str, float, float]], xmid: float) -> Optional[str]:
    for h, left, right in header_bins:
        if left <= xmid < right:
            return h
    return None

# ───────────────── GROUP WORDS ─────────────────
def extract_words_grouped(page):
    words = page.extract_words(
        x_tolerance=1.5, y_tolerance=2.6,
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

# ───────────────── DEEP STITCH ─────────────────
def deep_stitch_labels_and_numbers(all_pages_lines: List[List[dict]], first_period_x_by_page: Dict[int, float]):
    """
    1) Within each page:
       - Collapse consecutive label-only lines (left of period, no digits) into one label block.
       - Attach the immediate following numeric line to that label block.
    2) Across pages:
       - If a page ends with a dangling label block (no numbers attached),
         carry it and prepend to the first numeric line on the next page.
    """
    stitched_pages = []
    carry_label_block = None  # (page_number, text, words, y, x_min)

    for page_idx, lines in enumerate(all_pages_lines, start=1):
        stitched = []
        label_right_edge = first_period_x_by_page.get(page_idx, None)
        if label_right_edge is None:
            stitched_pages.append([])
            continue
        label_right_edge -= 8  # widen label zone

        def is_label_only(L: dict) -> bool:
            left_of_period = all(w["x1"] <= label_right_edge for w in L["words"])
            has_digit = any(re.search(r"\d", w["text"]) for w in L["words"])
            return left_of_period and not has_digit

        def is_numeric_row(L: dict) -> bool:
            # numeric tokens to the right of label zone
            has_num = any(re.fullmatch(r"-?\d+", w["text"].strip()) for w in L["words"])
            right_tokens = [w for w in L["words"] if w["x0"] > label_right_edge]
            return has_num and len(right_tokens) > 0

        i = 0
        # If there's a carry_label_block from previous page, try to attach to the FIRST numeric row on this page
        if carry_label_block:
            # find first numeric row
            j = 0
            attached = False
            while j < len(lines):
                if is_numeric_row(lines[j]):
                    merged_words = carry_label_block["words"] + lines[j]["words"]
                    merged_text = " ".join(w["text"].strip() for w in merged_words if w["text"].strip())
                    stitched.append({
                        "y": carry_label_block["y"],
                        "x_min": min(carry_label_block["x_min"], lines[j]["x_min"]),
                        "words": merged_words,
                        "text": merged_text,
                    })
                    i = j + 1
                    attached = True
                    break
                j += 1
            if not attached:
                # no numeric row on this page; keep carrying
                stitched_pages.append(stitched)  # empty or whatever we have
                continue
            carry_label_block = None  # consumed

        while i < len(lines):
            L = lines[i]
            if is_label_only(L):
                # collapse consecutive label-only lines
                label_words = list(L["words"])
                label_y = L["y"]
                label_xmin = L["x_min"]
                i += 1
                while i < len(lines) and is_label_only(lines[i]):
                    label_words += lines[i]["words"]
                    label_y = min(label_y, lines[i]["y"])
                    label_xmin = min(label_xmin, lines[i]["x_min"])
                    i += 1
                # now expect a numeric row; if none on this page, carry to next page
                if i < len(lines) and is_numeric_row(lines[i]):
                    merged_words = label_words + lines[i]["words"]
                    merged_text = " ".join(w["text"].strip() for w in merged_words if w["text"].strip())
                    stitched.append({
                        "y": label_y,
                        "x_min": label_xmin,
                        "words": merged_words,
                        "text": merged_text,
                    })
                    i += 1
                else:
                    # carry this label block to next page
                    carry_label_block = {"y": label_y, "x_min": label_xmin, "words": label_words}
                    # and stop processing this page (remaining rows likely unrelated)
                    break
            else:
                # keep rows that aren't pure labels (might be unrelated headings; we ignore later)
                stitched.append(L)
                i += 1

        stitched_pages.append(stitched)

    return stitched_pages

# ───────────────── MATCH HELPERS ─────────────────
def match_reason(label_text: str) -> Optional[str]:
    s = _lc(label_text)
    for canon, rx in KEYWORD_REGEX.items():
        if rx.search(s):
            return canon
    for canon, keys in KEYWORD_SUBSTR.items():
        for k in keys:
            if k in s:
                return canon
    if ("compensation offered by" in s) or ("no/unsatisfactory" in s) or ("compensation" in s):
        return COMP_CANON
    return None

# ───────────────── PARSER ─────────────────
def parse_pdf_deep_stitch_counts(file_bytes: bytes, debug: bool = False):
    counts = defaultdict(lambda: defaultdict(int))  # reason -> period -> sum
    ordered_headers: List[str] = []
    debug_log = {"token_trace": [], "ignored_tokens": [], "header_bins": [], "stitched_preview": []}

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        header_positions: Dict[str, float] = {}
        carry_headers = None
        carry_total_x = None

        # first pass: collect header x per page and lines per page
        all_pages_lines = []
        first_period_x_by_page = {}

        for page in pdf.pages:
            headers = find_period_headers(page) or carry_headers
            if not headers:
                all_pages_lines.append([])
                continue
            # only reset carry if we found real headers on this page
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

            first_period_x = min(header_positions[h] for h in ordered_headers)
            first_period_x_by_page[page.page_number] = first_period_x

            lines = extract_words_grouped(page)
            all_pages_lines.append(lines)

        if not ordered_headers:
            return counts, ordered_headers, debug_log

        # build stitched pages with cross-page handling
        stitched_pages = deep_stitch_labels_and_numbers(all_pages_lines, first_period_x_by_page)

        # now do counting pass using bins for each page (recompute per page to get total_x)
        header_positions = {}
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
                    "header_positions": {h: header_positions[h] for h in ordered_headers},
                    "total_x": total_x,
                    "bins": header_bins,
                })

            # stitch result for this page
            lines = stitched_pages[page.page_number - 1]
            if debug:
    # keep *all* stitched lines (full debug visibility)
    for L in lines:
        debug_log["stitched_preview"].append({
            "page": page.page_number,
            "y": L["y"],
            "text": L["text"]
        })
            if not lines:
                continue

            first_period_x = min(header_positions[h] for h in ordered_headers)
            label_right_edge = first_period_x - 8  # match deep stitch tolerance

            def left_label_text(line):
                return " ".join(
                    w["text"].strip()
                    for w in line["words"]
                    if w["x1"] <= label_right_edge and w["text"].strip()
                ).strip()

            def consume(line_obj, canon_reason: str):
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
                    if debug:
                        debug_log["token_trace"].append({
                            "page": page.page_number, "reason": canon_reason,
                            "period": mapped, "value": int(token)
                        })

            i = 0
            while i < len(lines):
                L = lines[i]
                label_text = left_label_text(L)
                label_lc = _lc(label_text)

                if is_structural_total(label_lc):
                    i += 1
                    continue

                # compensation: grab neighbors too
                if ("no/unsatisfactory" in label_lc or
                    "compensation offered by" in label_lc or
                    label_lc.strip() == "restaurant" or
                    "compensation" in label_lc):
                    canon = COMP_CANON
                    consume(L, canon)
                    if i > 0: consume(lines[i-1], canon)
                    if i + 1 < len(lines): consume(lines[i+1], canon)
                    i += 1
                    continue

                canon = match_reason(label_text)
                if not canon:
                    i += 1
                    continue

                consume(L, canon)
                i += 1

    return counts, ordered_headers, debug_log

# ───────────────── RUN ─────────────────
with st.spinner("Roll Tide…"):
    counts_map, ordered_headers, debug_log = parse_pdf_deep_stitch_counts(file_bytes, debug=debug_mode)

if not ordered_headers:
    st.error("No period headers (like ‘P9 25’) found.")
    st.stop()

# ───────────────── PERIOD PICKER ─────────────────
st.header("2) Pick the period")
sel_col = st.selectbox("Period", options=ordered_headers, index=len(ordered_headers)-1)

# ───────────────── REASON x PERIOD TABLE ─────────────────
periods = ordered_headers
df_all = pd.DataFrame(index=ALL_CANONICAL, columns=periods).fillna(0).astype(int)
for reason, per_map in counts_map.items():
    for p, v in per_map.items():
        if reason in df_all.index and p in df_all.columns:
            df_all.loc[reason, p] = int(v)
df_all["Total"] = df_all[periods].sum(axis=1)

# Category totals (selected)
tot_missing = int(df_all.loc[MISSING_REASONS, sel_col].sum())
tot_att     = int(df_all.loc[ATTITUDE_REASONS, sel_col].sum())
tot_other   = int(df_all.loc[OTHER_REASONS, sel_col].sum())
overall     = tot_missing + tot_att + tot_other

# ───────────────── SCOREBOARD ─────────────────
score_css = """
<style>
.score-wrap{display:flex;gap:16px;margin:10px 0 8px 0}
.score{flex:1;background:#1113;border:2px solid #38414a;border-radius:14px;padding:18px 20px;text-align:center}
.score h4{margin:0 0 8px 0;font-weight:700;font-size:1.05rem;color:#cfd8e3}
.score .num{font-size:2.4rem;line-height:1.1;font-weight:800;color:#fff;margin:2px 0 2px}
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

# ───────────────── SECTION TABLES ─────────────────
st.header("3) Reason totals — To-go Missing Complaints")
missing_df = pd.DataFrame({"Total": df_all.loc[MISSING_REASONS, sel_col].astype(int)})
missing_df.loc["— Grand Total —"] = missing_df.sum(numeric_only=True)
st.dataframe(style_table(missing_df), use_container_width=True)

st.header("3b) Reason totals — Attitude")
att_df = pd.DataFrame({"Total": df_all.loc[ATTITUDE_REASONS, sel_col].astype(int)})
att_df.loc["— Grand Total —"] = att_df.sum(numeric_only=True)
st.dataframe(style_table(att_df), use_container_width=True)

st.header("3c) Reason totals — Other")
oth_df = pd.DataFrame({"Total": df_all.loc[OTHER_REASONS, sel_col].astype(int)})
oth_df.loc["— Grand Total —"] = oth_df.sum(numeric_only=True)
st.dataframe(style_table(oth_df), use_container_width=True)

# ───────────────── EXPORTS ─────────────────
st.header("4) Export results")
buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    df_all.to_excel(writer, sheet_name="Reason x Period (All)", index=True)
    missing_df.to_excel(writer, sheet_name="Missing (Selected)", index=True)
    att_df.to_excel(writer, sheet_name="Attitude (Selected)", index=True)
    oth_df.to_excel(writer, sheet_name="Other (Selected)", index=True)
    if debug_mode:
        pd.DataFrame(debug_log.get("token_trace", [])).to_excel(writer, sheet_name="Token Trace", index=False)
        pd.DataFrame(debug_log.get("ignored_tokens", [])).to_excel(writer, sheet_name="Ignored Tokens", index=False)
        pd.DataFrame(debug_log.get("stitched_preview", [])).to_excel(writer, sheet_name="Stitched Preview", index=False)
        # bins compact view
        rows = []
        for rec in debug_log.get("header_bins", []):
            page = rec["page"]; total_x = rec["total_x"]
            for (period,left,right) in rec["bins"]:
                rows.append({"page":page,"period":period,"left":round(left,1),"right":round(right,1),"total_x":round(total_x,1) if total_x else None})
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Header Bins", index=False)

st.download_button(
    "📥 Download Excel (All Sheets)",
    data=buff.getvalue(),
    file_name=f"greeno_big_three_{sel_col.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
