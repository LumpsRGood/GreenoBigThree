# Greeno Big Three v1.9.2 — wrap-aware + strict matching, Pure Count toggle kept
import io, re, math, itertools
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

try:
    import pdfplumber  # robust word positions
except Exception as e:
    st.error("pdfplumber is required. Please add pdfplumber to your environment.\n\n" + str(e))
    st.stop()

st.set_page_config(page_title="Greeno Big Three v1.9.2", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────────

PERIOD_PATTERN = re.compile(r"\bP(\d{1,2})\s*(\d{2})\b", re.I)
TOTAL_HEADER_PATTERN = re.compile(r"\bTotal\b", re.I)

CATEGORIES = OrderedDict({
    "To-go Missing Complaints": [
        "Missing food",
        "Order wrong",
        "Missing condiments",
        "Out of menu item",
        "Missing bev",
        "Missing ingredients",
        "Packaging to-go complaint",
    ],
    "Attitude": [
        "Unprofessional/Unfriendly",
        "Manager directly involved",
        "Manager not available",
        "Manager did not visit",
        "Negative mgr-employee exchange",
        "Manager did not follow up",
        "Argued with guest",
    ],
    "Other": [
        "Long hold/no answer",
        "No/insufficient compensation offered",
        "Did not attempt to resolve",
        "Guest left without ordering",
        "Unknowledgeable",
        "Did not open on time",
        "No/poor apology",
    ],
})

# Strict regex triggers (match on normalized label text)
REGEX_TRIGGERS: Dict[str, List[str]] = {
    # TO-GO MISSING
    "Missing food":                 [r"\bmissing\s+food\b"],
    "Order wrong":                  [r"\border\s+wrong\b"],
    "Missing condiments":           [r"\bmissing\s+condiments?\b"],
    "Out of menu item":             [r"\bout\s+of\s+menu\s+item\b"],
    "Missing bev":                  [r"\bmissing\s+bev(?:erage)?\b"],
    "Missing ingredients":          [r"\bmissing\s+ingredients?\b"],
    "Packaging to-go complaint":    [r"\bpackaging\s+to\s*[- ]?\s*go\s+complaint\b"],

    # ATTITUDE
    "Unprofessional/Unfriendly":    [r"\bunprofessional\b", r"\bunfriendly\b"],
    "Manager directly involved":    [r"\bmanager\s+directly\s+involved\b"],
    "Manager not available":        [r"\bmanagement?\s+not\s+available\b"],
    "Manager did not visit":        [r"\bmanager\s+did\s+not\s+visit\b"],
    "Negative mgr-employee exchange":[r"\bnegative\s+manager[- ]employee\s+(?:interaction|exchange)\b"],
    "Manager did not follow up":    [r"\bmanager\s+did\s+not\s+follow\s+up\b"],
    "Argued with guest":            [r"\bargued\s+with\s+guest\b"],

    # OTHER
    "Long hold/no answer":          [r"\blong\s+hold\b", r"\bno\s+answer\b", r"\bhung\s+up\b"],
    "No/insufficient compensation offered": [
        r"\bno\/?unsatisfactory\b.*\bcompensation\b.*\brestaurant\b",
        r"\bcompensation\s+offered\s+by\s+restaurant\b"
    ],
    "Did not attempt to resolve":   [r"\bdid\s+not\s+attempt\s+to\s+resolve(?:\s+issue)?\b"],
    "Guest left without ordering":  [r"\bguest\s+left\s+without\s+(?:dining\s+or\s+)?ordering\b"],
    "Unknowledgeable":              [r"\bunknowledgeable\b"],
    "Did not open on time":         [r"\bdid\s+not\s+open\s+on\s+time\b"],
    "No/poor apology":              [r"\bno\/?poor\s+apology\b"],
}

ALL_CANON = [m for group in CATEGORIES.values() for m in group]

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _norm_label(s: str) -> str:
    s = s.lower()
    # normalize common parentheticals to plain words
    s = s.replace("(food)", "food").replace("(bev)", "bev")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _match_canon_with_regex(norm_label: str) -> Optional[str]:
    for canon, pats in REGEX_TRIGGERS.items():
        for pat in pats:
            if re.search(pat, norm_label):
                return canon
    return None

def _looks_like_header(w):
    # very rough: bold headers often have larger height or are underlined in the report,
    # but we don't rely on style — we rely on the words themselves
    return False

def _cluster_lines(words, y_tol=2.2):
    """Group words into text lines by y (top)."""
    # words: list of dicts with x0, x1, top, bottom, text
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    current = []
    last_top = None
    for w in words:
        t = w["top"]
        if last_top is None or abs(t - last_top) <= y_tol:
            current.append(w)
            last_top = t if last_top is None else (last_top + t) / 2
        else:
            lines.append(sorted(current, key=lambda z: z["x0"]))
            current = [w]
            last_top = t
    if current:
        lines.append(sorted(current, key=lambda z: z["x0"]))
    return lines

def _line_text(line_words) -> str:
    return " ".join(w["text"] for w in line_words).strip()

def _left_text_and_numbers(line_words, header_x_map: Dict[str, float]):
    """Split a line into (left_text, [(period, value)]) by using number x-positions vs header map."""
    if not header_x_map:
        return _line_text(line_words), []

    left_parts = []
    nums = []  # (period, value)
    for w in line_words:
        txt = w["text"]
        if re.fullmatch(r"\d+", txt):
            # map to nearest header by x center
            cx = (w["x0"] + w["x1"]) / 2.0
            # find closest header x
            nearest = None
            best = 1e9
            for p, x in header_x_map.items():
                d = abs(cx - x)
                if d < best:
                    best = d
                    nearest = p
            if nearest:
                nums.append((nearest, int(txt)))
        else:
            left_parts.append(txt)
    return " ".join(left_parts).strip(), nums

def _detect_headers(page) -> Dict[str, float]:
    """
    Detect 'P# YY' headers along the top by looking for 'P\d+ \d\d' tokens and
    averaging their x centers. Return OrderedDict(period_label -> x_center)
    in left-to-right order. Ignore 'Total' header.
    """
    words = page.extract_words(x_tolerance=1.0, y_tolerance=1.0, keep_blank_chars=False)
    if not words:
        return {}
    # find header candidates by vertical position (top ~ first 15% of page)
    h = float(page.height)
    top_cut = h * 0.22
    header_words = [w for w in words if w["top"] <= top_cut]
    # collect period tokens and their centers
    found = []
    for w in header_words:
        t = w["text"]
        m = PERIOD_PATTERN.search(t)
        if m:
            label = f"P{int(m.group(1))} {m.group(2)}"
            cx = (w["x0"] + w["x1"]) / 2.0
            found.append((cx, label))
        elif TOTAL_HEADER_PATTERN.search(t):
            # we skip 'Total' as we don't use it
            pass
    found.sort(key=lambda z: z[0])
    # Make a clean ordered mapping with averaged positions (periods sometimes appear twice)
    pos_map = OrderedDict()
    for cx, lab in found:
        if lab not in pos_map:
            pos_map[lab] = []
        pos_map[lab].append(cx)
    pos_map = OrderedDict((k, float(np.mean(v))) for k, v in pos_map.items())
    return pos_map

def _merge_wraps(lines_text: List[str]) -> List[str]:
    """
    Merge vertical wraps: if a line has NO numbers and the next line has NO numbers,
    join them, because it's likely the reason label wrapped. Keep merging while true.
    """
    merged = []
    i = 0
    num_pat = re.compile(r"\b\d+\b")
    while i < len(lines_text):
        cur = lines_text[i]
        cur_has_num = bool(num_pat.search(cur))
        if not cur_has_num and (i + 1) < len(lines_text):
            nxt = lines_text[i + 1]
            nxt_has_num = bool(num_pat.search(nxt))
            if not nxt_has_num:
                # both are label fragments → merge
                cur = f"{cur} {nxt}".strip()
                i += 2
                # continue chaining if the following are also no-number lines
                while i < len(lines_text) and not num_pat.search(lines_text[i]):
                    cur = f"{cur} {lines_text[i]}".strip()
                    i += 1
                merged.append(cur)
                continue
        merged.append(cur)
        i += 1
    return merged

def _is_comp_window(win: List[str]) -> bool:
    """Window-level check for compensation multi-line phrase."""
    if not win:
        return False
    joined = _norm_label(" ".join(win))
    need = ["no unsatisfactory", "compensation offered by", "restaurant"]
    return all(tok in joined for tok in need)

# ───────────────────────────────────────────────────────────────────────────────
# Core parsing
# ───────────────────────────────────────────────────────────────────────────────

def parse_pdf_full(file_bytes: bytes) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Return (ordered_periods, counts) where counts[category][reason][period] = sum
    Uses wrap-aware labeling and header mapping. All segments included.
    """
    counts = {cat: {reason: defaultdict(int) for reason in CATEGORIES[cat]} for cat in CATEGORIES}
    periods_all: List[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            header_x = _detect_headers(page)
            if header_x:
                for p in header_x.keys():
                    if p not in periods_all:
                        periods_all.append(p)

            words = page.extract_words(x_tolerance=1.0, y_tolerance=1.2, keep_blank_chars=False)
            if not words:
                continue

            # form line objects and texts
            line_word_groups = _cluster_lines(words, y_tol=2.0)
            raw_lines = [_line_text(g) for g in line_word_groups]
            lines = _merge_wraps(raw_lines)  # wrap-aware

            # Now re-read again WITH positions for numeric mapping per line
            # We must walk the original groups and merge their texts the same way
            # Build a parallel structure: merged_groups with combined word lists
            def group_has_number(g):
                return any(re.fullmatch(r"\d+", w["text"]) for w in g)

            merged_groups = []
            i = 0
            while i < len(line_word_groups):
                g = line_word_groups[i]
                if not group_has_number(g) and (i + 1) < len(line_word_groups) and not group_has_number(line_word_groups[i + 1]):
                    # merge consecutive no-number lines
                    newg = list(g)
                    i += 1
                    while i < len(line_word_groups) and not group_has_number(line_word_groups[i]):
                        newg.extend(line_word_groups[i])
                        i += 1
                    merged_groups.append(sorted(newg, key=lambda z: z["x0"]))
                    continue
                merged_groups.append(g)
                i += 1

            # Walk merged groups; compute left text + numbers, match canon
            for g in merged_groups:
                left_txt, nums = _left_text_and_numbers(g, header_x)
                if not left_txt or not nums:
                    # also handle 3-line compensation windows; if nums empty we skip
                    continue
                norm = _norm_label(left_txt)
                canon = _match_canon_with_regex(norm)

                # If compensation appears split across lines: we already merged vertical wraps,
                # but sometimes compensation spans 3+ separated lines; be defensive and re-check window
                if not canon and ("compensation" in norm or "offered" in norm or "restaurant" in norm):
                    # build a small window of neighbor lines’ left texts (from raw_lines after merge)
                    # Heuristic: norm already contains joined label; treat as compensation if phrase present
                    if _is_comp_window([left_txt]):
                        canon = "No/insufficient compensation offered"

                if not canon:
                    continue

                # assign numbers to periods
                for per, val in nums:
                    for cat, reasons in CATEGORIES.items():
                        if canon in reasons:
                            counts[cat][canon][per] += val

    # order periods left to right by natural numeric sort (P#, then YY)
    def period_sort_key(p):
        m = re.match(r"P(\d+)\s+(\d{2})", p)
        if not m:
            return (999, 99)
        return (int(m.group(1)), int(m.group(2)))
    periods_all = sorted(list(set(periods_all)), key=period_sort_key)
    return periods_all, counts

def parse_pdf_pure_counts(file_bytes: bytes) -> Dict[str, int]:
    """
    Pure count mode: count occurrences of each reason across entire PDF
    (no periods / headers). Wrap-aware + strict regex.
    """
    totals = {m: 0 for m in ALL_CANON}

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.0, y_tolerance=1.2, keep_blank_chars=False)
            if not words:
                continue
            line_groups = _cluster_lines(words, y_tol=2.0)
            raw_lines = [_line_text(g) for g in line_groups]
            lines = _merge_wraps(raw_lines)

            # count per merged line if it has digits (otherwise it's a label-only line)
            for g in _cluster_lines(words, y_tol=2.0):  # we will use positions to see digits
                pass  # just to keep consistent structure; we only need texts here

            for s in lines:
                if not re.search(r"\d", s):
                    # label-only line, skip (numbers are on the number line, not the label line)
                    continue
                # split off numbers, leave label
                label = re.sub(r"\b\d+\b", " ", s)
                norm = _norm_label(label)
                canon = _match_canon_with_regex(norm)
                # compensation window heuristic: if the single line holds all parts, accept
                if not canon and ("compensation" in norm or "offered" in norm or "restaurant" in norm):
                    if _is_comp_window([label]):
                        canon = "No/insufficient compensation offered"
                if canon:
                    # Sum the digits on that line for a rough total in pure mode
                    vals = [int(x) for x in re.findall(r"\b\d+\b", s)]
                    if vals:
                        totals[canon] += sum(vals)

    return totals

# ───────────────────────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────────────────────

st.title("Greeno Big Three v1.9.2")

with st.sidebar:
    st.subheader("Upload")
    up = st.file_uploader("PDF report", type=["pdf"])
    pure_mode = st.toggle("Pure Count (debug/troubleshoot)", value=False)
    st.caption("Pure Count totals ignore period columns and simply sum reason rows. Use this to troubleshoot extraction vs hand counts.")

if not up:
    st.markdown(
        """
        <div style="text-align:center;padding:48px;border:1px dashed #444;border-radius:16px;">
            <div style="font-size:20px;opacity:.8">⬅️ Upload the PDF on the left to get started.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = up.read()

with st.spinner("Roll Tide"):
    if pure_mode:
        pure_totals = parse_pdf_pure_counts(file_bytes)
        # Display totals in the three category blocks
        for cat, reasons in CATEGORIES.items():
            st.subheader(cat)
            rows = []
            for r in reasons:
                rows.append({"Reason": r, "Total": pure_totals.get(r, 0)})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        periods, counts = parse_pdf_full(file_bytes)

        if not periods:
            st.error("No period headers detected across the PDF. Check the file formatting.")
            st.stop()

        # Compose category tables (period columns in order)
        for cat, reasons in CATEGORIES.items():
            st.subheader(cat)
            records = []
            for r in reasons:
                row = {"Reason": r}
                for p in periods:
                    row[p] = counts[cat][r].get(p, 0)
                row["Total"] = sum(row[p] for p in periods)
                records.append(row)
            df = pd.DataFrame(records)
            # Totals row
            total_row = {"Reason": "— Grand Total —"}
            for p in periods:
                total_row[p] = df[p].sum()
            total_row["Total"] = sum(total_row[p] for p in periods)
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

            # style for readability
            def style_table(dframe: pd.DataFrame):
                sty = (dframe.style
                       .set_properties(**{"background-color": "#111", "color": "#EDEDED", "border-color": "#333"})
                       .highlight_max(color="#1b4332", axis=0)
                       )
                # zebra rows
                zebra = np.where(np.arange(len(dframe)) % 2 == 0, "#161a1f", "#0f1216")
                sty = sty.set_properties(subset=pd.IndexSlice[:, :], **{"background-color": ""})
                sty = sty.apply(lambda s: ["background-color: %s" % zebra[s.index.get_loc(i)] for i in s.index], axis=0)
                # grand total row
                gt_idx = dframe.index[dframe["Reason"] == "— Grand Total —"]
                if len(gt_idx):
                    def _gt_highlight(row):
                        return ["background-color: #3a2f00; color: #fff; font-weight:600" if row.name in gt_idx else "" for _ in row]
                    sty = sty.apply(_gt_highlight, axis=1)
                return sty

            st.dataframe(df, use_container_width=True, hide_index=True)
            try:
                st.write(style_table(df).to_html(), unsafe_allow_html=True)
            except Exception:
                pass

# End
