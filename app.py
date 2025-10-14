# Greeno Big Three v1.9.5 — Robust header detection (global prepass) + wrap-aware label merge
# Keeps UI simple (title, sidebar upload, "Roll Tide", Pure Count toggle, 3 tables with grand totals)

from __future__ import annotations
import io
import re
from collections import defaultdict, OrderedDict, Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception as e:
    st.error("Missing dependency: pdfplumber\n\n" + str(e))
    st.stop()

st.set_page_config(page_title="Greeno Big Three v1.9.5", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# Config / Canonical reasons
# ───────────────────────────────────────────────────────────────────────────────

PERIOD_RE = re.compile(r"\bP(\d{1,2})\b", re.I)        # catch split headers like "P9" on one token
YEAR2_RE  = re.compile(r"\b(\d{2})\b")                 # "25" possibly next to it
PERIOD_FULL_RE = re.compile(r"\bP(\d{1,2})\s*(\d{2})\b", re.I)
TOTAL_HDR_RE = re.compile(r"\bTotal\b", re.I)
NUM_RE = re.compile(r"\b\d+\b")

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

ALL_CANON = [m for group in CATEGORIES.values() for m in group]

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

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def norm_label(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("(food)", "food").replace("(bev)", "bev")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def match_canon(norm_text: str) -> Optional[str]:
    for canon, pats in REGEX_TRIGGERS.items():
        for pat in pats:
            if re.search(pat, norm_text):
                return canon
    return None

def looks_like_comp_phrase(text: str) -> bool:
    n = norm_label(text)
    need = ["no unsatisfactory", "compensation offered by", "restaurant"]
    return all(tok in n for tok in need)

def cluster_lines(words: List[dict], y_tol: float = 2.0) -> List[List[dict]]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines, cur, last_top = [], [], None
    for w in words:
        t = w["top"]
        if last_top is None or abs(t - last_top) <= y_tol:
            cur.append(w)
            last_top = t if last_top is None else (last_top + t) / 2
        else:
            lines.append(sorted(cur, key=lambda z: z["x0"]))
            cur = [w]; last_top = t
    if cur:
        lines.append(sorted(cur, key=lambda z: z["x0"]))
    return lines

def line_text(words: List[dict]) -> str:
    return " ".join(w["text"] for w in words).strip()

def group_has_number(words: List[dict]) -> bool:
    return any(NUM_RE.fullmatch(w["text"]) for w in words)

def merge_wrapped_label_lines(line_word_groups: List[List[dict]]) -> List[List[dict]]:
    """Merge consecutive label-only lines (no numbers) to handle vertical wraps."""
    merged: List[List[dict]] = []
    i = 0
    while i < len(line_word_groups):
        g = line_word_groups[i]
        if not group_has_number(g):
            bucket = list(g)
            j = i + 1
            while j < len(line_word_groups) and not group_has_number(line_word_groups[j]):
                bucket.extend(line_word_groups[j]); j += 1
            bucket = sorted(bucket, key=lambda w: w["x0"])
            merged.append(bucket)
            i = j
        else:
            merged.append(g); i += 1
    return merged

def line_left_and_nums(line_words: List[dict], header_x_map: Dict[str, float]) -> Tuple[str, List[Tuple[str, int]]]:
    if not header_x_map:
        return line_text(line_words), []
    left, nums = [], []
    for w in line_words:
        t = w["text"]
        if NUM_RE.fullmatch(t):
            cx = (w["x0"] + w["x1"]) / 2.0
            nearest, best = None, 1e9
            for p, x in header_x_map.items():
                d = abs(cx - x)
                if d < best:
                    best, nearest = d, p
            if nearest is not None:
                nums.append((nearest, int(t)))
        else:
            left.append(t)
    return " ".join(left).strip(), nums

def period_sort_key(p: str) -> Tuple[int, int]:
    m = re.match(r"P(\d+)\s+(\d{2})", p)
    if not m:
        return (999, 99)
    return (int(m.group(1)), int(m.group(2)))

# ───────────────────────────────────────────────────────────────────────────────
# Robust header detection
# ───────────────────────────────────────────────────────────────────────────────

def detect_headers_page(page) -> OrderedDict[str, float]:
    """
    Conservative per-page header detection:
    - Look near the top for "P# YY"
    - Also support split tokens ("P9" on one word, "25" on another next to it)
    """
    words = page.extract_words(x_tolerance=1.2, y_tolerance=1.2, keep_blank_chars=False)
    if not words:
        return OrderedDict()
    h = float(page.height)
    top_cut = h * 0.35  # wider top band; some pages print headers lower
    header_words = [w for w in words if w["top"] <= top_cut]

    found: List[Tuple[float, str]] = []

    # 1) Direct "P# YY"
    for w in header_words:
        m = PERIOD_FULL_RE.search(w["text"])
        if m:
            lab = f"P{int(m.group(1))} {m.group(2)}"
            cx = (w["x0"] + w["x1"]) / 2.0
            found.append((cx, lab))

    # 2) Split "P#" next to separate "YY"
    #    within small vicinity (same row band, within ~120 px horizontally)
    for i, w in enumerate(header_words):
        t = w["text"]
        mP = PERIOD_RE.fullmatch(t)
        if not mP:
            continue
        # find a neighbor 2-digit year near same y band
        p_top = w["top"]; p_bot = w["bottom"]
        for v in header_words:
            if v is w:
                continue
            if abs(((v["top"] + v["bottom"]) / 2.0) - ((p_top + p_bot) / 2.0)) > 8.0:
                continue
            if not YEAR2_RE.fullmatch(v["text"]):
                continue
            # close horizontally
            if abs(v["x0"] - w["x1"]) > 120 and abs(w["x0"] - v["x1"]) > 120:
                continue
            lab = f"P{int(mP.group(1))} {v['text']}"
            cx = ((w["x0"] + w["x1"] + v["x0"] + v["x1"]) / 4.0)
            found.append((cx, lab))

    # 3) remove "Total"
    found = [(cx, lab) for (cx, lab) in found if not TOTAL_HDR_RE.search(lab)]

    if not found:
        return OrderedDict()

    # average duplicates for each label
    agg: Dict[str, List[float]] = {}
    for cx, lab in found:
        agg.setdefault(lab, []).append(cx)
    mapping = OrderedDict((lab, float(np.mean(xs))) for lab, xs in agg.items())
    # sort left->right
    mapping = OrderedDict(sorted(mapping.items(), key=lambda kv: kv[1]))
    return mapping

def build_global_headers(pdf) -> Tuple[List[str], OrderedDict[str, float]]:
    """
    Pass 1 over the whole PDF:
    - Try text-based header detection on each page, aggregate
    - If nothing found anywhere, fall back to most common number x-positions as columns
    """
    label_to_xs: Dict[str, List[float]] = {}
    any_found = False
    number_xs: List[float] = []

    for page in pdf.pages:
        # collect numbers’ x-centers for a potential fallback
        words_all = page.extract_words(x_tolerance=1.2, y_tolerance=1.2, keep_blank_chars=False)
        for w in (words_all or []):
            if NUM_RE.fullmatch(w["text"]):
                number_xs.append((w["x0"] + w["x1"]) / 2.0)

        local = detect_headers_page(page)
        if local:
            any_found = True
            for lab, cx in local.items():
                label_to_xs.setdefault(lab, []).append(cx)

    if any_found:
        # average per label; sort left->right
        avg = OrderedDict((lab, float(np.mean(xs))) for lab, xs in label_to_xs.items())
        avg = OrderedDict(sorted(avg.items(), key=lambda kv: kv[1]))
        periods = list(avg.keys())
        periods_sorted = sorted(periods, key=period_sort_key)
        # Reorder mapping to natural period order
        nat = OrderedDict((p, avg[p]) for p in periods_sorted if p in avg)
        return periods_sorted, nat

    # Fallback: derive columns by most common number x-positions
    if not number_xs:
        return [], OrderedDict()

    # Cluster number centers into columns using histogram peaks
    xs = np.array(number_xs)
    # histogram with reasonable bins; pick top N peaks (we don't know N, but typical ~9)
    counts, bins = np.histogram(xs, bins=40)
    # choose peaks above 40% of max count
    thr = 0.40 * counts.max()
    peaks = []
    for i, c in enumerate(counts):
        if c >= thr:
            x_center = (bins[i] + bins[i+1]) / 2.0
            peaks.append(x_center)
    # sort and unique-ish
    peaks = sorted(peaks)
    uniq = []
    for x in peaks:
        if not uniq or abs(x - uniq[-1]) > 18.0:
            uniq.append(x)
    # Name columns as P? ?? left->right if we can’t read period labels; still usable for alignment
    mapping = OrderedDict((f"C{i+1}", float(x)) for i, x in enumerate(uniq))
    return list(mapping.keys()), mapping

# ───────────────────────────────────────────────────────────────────────────────
# Core parsing
# ───────────────────────────────────────────────────────────────────────────────

def parse_pdf_full(file_bytes: bytes) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, int]]], OrderedDict[str, float]]:
    counts = {cat: {reason: defaultdict(int) for reason in reasons}
              for cat, reasons in CATEGORIES.items()}

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        # Global header map (robust)
        periods_all, global_header_map = build_global_headers(pdf)

        if not periods_all:
            return [], counts, OrderedDict()

        # Now parse each page; if page headers missing, use global map
        for page in pdf.pages:
            local_map = detect_headers_page(page)
            header_x = local_map if local_map else global_header_map

            words = page.extract_words(x_tolerance=1.2, y_tolerance=1.3, keep_blank_chars=False)
            if not words:
                continue

            line_groups = cluster_lines(words, y_tol=2.0)
            merged_groups = merge_wrapped_label_lines(line_groups)

            for g in merged_groups:
                label_txt, nums = line_left_and_nums(g, header_x)
                if not nums:
                    continue
                n = norm_label(label_txt)
                canon = match_canon(n)
                if not canon and ("compensation" in n or "restaurant" in n or "unsatisfactory" in n):
                    if looks_like_comp_phrase(label_txt):
                        canon = "No/insufficient compensation offered"
                if not canon:
                    continue

                for per, val in nums:
                    # If we fell back to generic column names (C1, C2,...), we can’t map to true periods.
                    # In that rare fallback case, just skip (or accumulate under per).
                    if per not in periods_all:
                        # attempt nearest by left->right index
                        pass
                    for cat, reason_list in CATEGORIES.items():
                        if canon in reason_list:
                            counts[cat][canon][per] += val

    # order periods by natural sort if they are real 'P# YY', else keep in left->right from global
    if all(PERIOD_FULL_RE.match(p or "") for p in periods_all):
        periods_all = sorted(set(periods_all), key=period_sort_key)
    else:
        periods_all = list(dict.fromkeys(periods_all))  # keep order

    return periods_all, counts, global_header_map

def parse_pdf_pure_counts(file_bytes: bytes) -> Dict[str, int]:
    totals = {r: 0 for r in ALL_CANON}
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.2, y_tolerance=1.3, keep_blank_chars=False)
            if not words:
                continue
            line_groups = cluster_lines(words, y_tol=2.0)
            merged_groups = merge_wrapped_label_lines(line_groups)

            for g in merged_groups:
                s = line_text(g)
                if not NUM_RE.search(s):
                    continue
                label_only = NUM_RE.sub(" ", s)
                n = norm_label(label_only)
                canon = match_canon(n)
                if not canon and ("compensation" in n or "restaurant" in n or "unsatisfactory" in n):
                    if looks_like_comp_phrase(label_only):
                        canon = "No/insufficient compensation offered"
                if not canon:
                    continue
                vals = [int(x) for x in NUM_RE.findall(s)]
                if vals:
                    totals[canon] += sum(vals)
    return totals

# ───────────────────────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────────────────────

st.title("Greeno Big Three v1.9.5")

with st.sidebar:
    st.subheader("Upload")
    up = st.file_uploader("PDF report", type=["pdf"])
    pure_mode = st.toggle("Pure Count (debug / troubleshoot)", value=False)
    show_diag = st.toggle("Show header diagnostics", value=False)
    st.caption("Pure Count ignores period columns and sums reasons across the whole PDF.")

if not up:
    st.markdown(
        """
        <div style="text-align:center;padding:48px;border:1px dashed #444;border-radius:16px;">
            <div style="font-size:20px;opacity:.85">⬅️ Upload the PDF on the left to get started.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = up.read()

with st.spinner("Roll Tide"):
    if pure_mode:
        totals = parse_pdf_pure_counts(file_bytes)
        for cat, reason_list in CATEGORIES.items():
            st.subheader(cat)
            df = pd.DataFrame([{"Reason": r, "Total": totals.get(r, 0)} for r in reason_list])
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        periods, counts, header_map = parse_pdf_full(file_bytes)

        if show_diag:
            st.markdown("##### Header diagnostics")
            if header_map:
                diag = pd.DataFrame([{"Period": k, "x_center": round(v, 1)} for k, v in header_map.items()])
                st.dataframe(diag, use_container_width=True, hide_index=True)
            else:
                st.info("No headers found in global pre-pass.")

        if not periods:
            st.error("No period headers detected anywhere in the PDF. Header parsing failed — check the file formatting.")
            st.stop()

        for cat, reason_list in CATEGORIES.items():
            st.subheader(cat)
            rows = []
            for r in reason_list:
                row = {"Reason": r}
                for p in periods:
                    row[p] = counts[cat][r].get(p, 0)
                row["Total"] = sum(row[p] for p in periods)
                rows.append(row)
            df = pd.DataFrame(rows)

            total_row = {"Reason": "— Grand Total —"}
            for p in periods:
                total_row[p] = df[p].sum()
            total_row["Total"] = sum(total_row[p] for p in periods)
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

            st.dataframe(df, use_container_width=True, hide_index=True)
