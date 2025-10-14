# Greeno Big Three v1.9.3 — Minimal/safe fix: wrap-aware labels + strict matching + Pure Count toggle

from __future__ import annotations
import io
import re
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# You need pdfplumber in your environment (Streamlit Cloud: add to requirements.txt)
try:
    import pdfplumber
except Exception as e:
    st.error("Missing dependency: pdfplumber\n\n" + str(e))
    st.stop()

st.set_page_config(page_title="Greeno Big Three v1.9.3", layout="wide")


# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

# Period headers like "P9 25"
PERIOD_RE = re.compile(r"\bP(\d{1,2})\s*(\d{2})\b", re.I)
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

# Strict regex triggers on *normalized* label text
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
# Helpers (safe, minimal additions)
# ───────────────────────────────────────────────────────────────────────────────

def norm_label(s: str) -> str:
    """Normalize label for matching: lowercase, remove punctuation, normalize parentheses."""
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
    """Group words into lines by Y proximity."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines, current, last_top = [], [], None
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

def line_text(words: List[dict]) -> str:
    return " ".join(w["text"] for w in words).strip()

def group_has_number(words: List[dict]) -> bool:
    return any(NUM_RE.fullmatch(w["text"]) for w in words)

def merge_wrapped_label_lines(line_word_groups: List[List[dict]]) -> List[List[dict]]:
    """
    Merge vertically wrapped label fragments (consecutive lines that both have no numbers).
    Keeps your original order and word positions.
    """
    merged: List[List[dict]] = []
    i = 0
    while i < len(line_word_groups):
        g = line_word_groups[i]
        if not group_has_number(g):
            bucket = list(g)
            j = i + 1
            while j < len(line_word_groups) and not group_has_number(line_word_groups[j]):
                bucket.extend(line_word_groups[j])
                j += 1
            bucket = sorted(bucket, key=lambda w: w["x0"])
            merged.append(bucket)
            i = j
        else:
            merged.append(g)
            i += 1
    return merged

def detect_headers(page) -> "OrderedDict[str, float]":
    """
    Map each detected period header (e.g., 'P9 25') to its x-center.
    Ignores the right-most 'Total' column since we don't use it.
    """
    words = page.extract_words(x_tolerance=1.0, y_tolerance=1.0, keep_blank_chars=False)
    if not words:
        return OrderedDict()
    h = float(page.height)
    top_cut = h * 0.22  # header region near the top
    header_words = [w for w in words if w["top"] <= top_cut]

    found: List[Tuple[float, str]] = []
    for w in header_words:
        t = w["text"]
        m = PERIOD_RE.search(t)
        if m:
            label = f"P{int(m.group(1))} {m.group(2)}"
            cx = (w["x0"] + w["x1"]) / 2.0
            found.append((cx, label))
        elif TOTAL_HDR_RE.search(t):
            # ignore "Total" header
            pass

    found.sort(key=lambda z: z[0])
    pos_map: Dict[str, List[float]] = OrderedDict()
    for cx, lab in found:
        pos_map.setdefault(lab, []).append(cx)
    # average duplicates
    return OrderedDict((lab, float(np.mean(xs))) for lab, xs in pos_map.items())

def left_text_and_numbers(line_words: List[dict], header_x_map: Dict[str, float]) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Split a merged line into label text and (period,value) numeric pairs
    by assigning digits to nearest period header x-center.
    """
    if not header_x_map:
        return line_text(line_words), []

    left_parts: List[str] = []
    nums: List[Tuple[str, int]] = []

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
            left_parts.append(t)

    return " ".join(left_parts).strip(), nums

def period_sort_key(p: str) -> Tuple[int, int]:
    m = re.match(r"P(\d+)\s+(\d{2})", p)
    if not m:
        return (999, 99)
    return (int(m.group(1)), int(m.group(2)))


# ───────────────────────────────────────────────────────────────────────────────
# Core parsing (minimal changes applied here)
# ───────────────────────────────────────────────────────────────────────────────

def parse_pdf_full(file_bytes: bytes) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Build per-period counts for each canonical reason.
    Returns (ordered_periods, counts[category][reason][period] -> int)
    """
    counts = {cat: {reason: defaultdict(int) for reason in reasons}
              for cat, reasons in CATEGORIES.items()}
    periods_all: List[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            header_x = detect_headers(page)
            if header_x:
                for p in header_x.keys():
                    if p not in periods_all:
                        periods_all.append(p)

            words = page.extract_words(x_tolerance=1.0, y_tolerance=1.2, keep_blank_chars=False)
            if not words:
                continue

            # Build lines, then merge wrapped label fragments (new)
            line_groups = cluster_lines(words, y_tol=2.0)
            merged_groups = merge_wrapped_label_lines(line_groups)

            for g in merged_groups:
                label_txt, nums = left_text_and_numbers(g, header_x)
                if not nums:
                    # numberless line => label-only; numbers will appear on its data line
                    # (we skip to avoid double counting)
                    continue

                n = norm_label(label_txt)
                canon = match_canon(n)

                # Special-case: compensation 3-part phrase when merged
                if not canon and ("compensation" in n or "restaurant" in n or "unsatisfactory" in n):
                    if looks_like_comp_phrase(label_txt):
                        canon = "No/insufficient compensation offered"

                if not canon:
                    continue

                for per, val in nums:
                    # Route to the correct category bucket
                    for cat, reason_list in CATEGORIES.items():
                        if canon in reason_list:
                            counts[cat][canon][per] += val

    periods_all = sorted(set(periods_all), key=period_sort_key)
    return periods_all, counts


def parse_pdf_pure_counts(file_bytes: bytes) -> Dict[str, int]:
    """
    Pure Count mode: sum values by reason across the whole PDF (no periods).
    Uses the same wrap-aware + strict matching so you can troubleshoot parity.
    """
    totals = {r: 0 for r in ALL_CANON}

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.0, y_tolerance=1.2, keep_blank_chars=False)
            if not words:
                continue
            line_groups = cluster_lines(words, y_tol=2.0)
            merged_groups = merge_wrapped_label_lines(line_groups)

            for g in merged_groups:
                s = line_text(g)
                if not NUM_RE.search(s):
                    continue
                # remove digits to leave the label-ish portion for matching
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

st.title("Greeno Big Three v1.9.3")

with st.sidebar:
    st.subheader("Upload")
    up = st.file_uploader("PDF report", type=["pdf"])
    pure_mode = st.toggle("Pure Count (debug / troubleshoot)", value=False)
    st.caption("Pure Count ignores period columns and sums reason rows across the whole PDF.")

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
            df = pd.DataFrame(
                [{"Reason": r, "Total": totals.get(r, 0)} for r in reason_list]
            )
            # Style for readability (zebra + subtle borders)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        periods, counts = parse_pdf_full(file_bytes)
        if not periods:
            st.error("No period headers detected. Check the file formatting.")
            st.stop()

        # For each category, render period table with grand total
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

            # Grand total row
            total_row = {"Reason": "— Grand Total —"}
            for p in periods:
                total_row[p] = df[p].sum()
            total_row["Total"] = sum(total_row[p] for p in periods)
            df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

            # Display
            st.dataframe(df, use_container_width=True, hide_index=True)
