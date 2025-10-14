# Greeno Big Three v1.9.4 — Resilient header fallback + tight counting
import io, os, re, base64
from collections import defaultdict
from typing import Optional, List, Tuple
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ---------- UI ----------
st.set_page_config(page_title="Greeno Big Three v1.9.4", layout="wide")
with st.sidebar:
    up = st.file_uploader("Upload the PDF report", type=["pdf"])
    pure_mode = st.toggle("✅ Pure Count Mode (ignore AD/Store/Segment)", value=False)
    debug_mode = st.checkbox("🔍 Debug", value=False)
if not up:
    st.stop()
if pdfplumber is None:
    st.error("pdfplumber not installed. `pip install pdfplumber`")
    st.stop()
file_bytes = up.read()

# ---------- constants ----------
HEADER_RX = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b", re.I)
STORE_LINE_RX = re.compile(r"^\s*\d{3,6}\s*-\s+.*")
SECTION_TOGO   = re.compile(r"^\s*(To[\s-]?Go|To-go)\s*:?\s*$", re.I)
SECTION_DELIV  = re.compile(r"^\s*Delivery\s*:?\s*$", re.I)
SECTION_DINEIN = re.compile(r"^\s*Dine[\s-]?In\s*:?\s*$", re.I)

# canonical reasons
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
COMP_CANON = "No/insufficient compensation offered"

# short/anchored triggers
KEYWORD_TRIGGERS = {
    # anchored / specific
    "Missing food":                 re.compile(r"\bmissing\s+item\s*\(food\)", re.I),
    "Missing bev":                  re.compile(r"\bmissing\s+item\s*\(bev\)",  re.I),
    "Missing condiments":           re.compile(r"\bmissing\s+condiments?",     re.I),
    "Missing ingredients":          re.compile(r"\bmissing\s+ingredient",      re.I),
    "Out of menu item":             re.compile(r"\bout\s+of\s+menu\s+item",    re.I),
    "Packaging to-go complaint":    re.compile(r"\bpackaging\s+to-?go",        re.I),

    # attitude / other by substrings
    "Unprofessional/Unfriendly":    "unfriendly",
    "Manager directly involved":    "directly involved",
    "Manager not available":        "manager not available",
    "Manager did not visit":        "did not visit",
    "Negative mgr-employee exchange":"manager-employee",
    "Manager did not follow up":    "follow up",
    "Argued with guest":            "argued",
    "Long hold/no answer":          "hung up",
    "No/insufficient compensation offered": "compensation",
    "Did not attempt to resolve":   "resolve",
    "Guest left without ordering":  "without ordering",
    "Unknowledgeable":              "unknowledgeable",
    "Did not open on time":         "open on time",
    "No/poor apology":              "apology",
}

def _lc(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def extract_lines(page):
    words = page.extract_words(x_tolerance=1.4, y_tolerance=2.4, use_text_flow=True)
    # group by y
    buckets = defaultdict(list)
    for w in words:
        ymid = round((w["top"] + w["bottom"]) / 2, 1)
        buckets[ymid].append(w)
    lines = []
    for y, ws in sorted(buckets.items(), key=lambda kv: kv[0]):
        ws = sorted(ws, key=lambda w: w["x0"])
        text = " ".join(w["text"].strip() for w in ws if w["text"].strip())
        if text:
            lines.append({"y": y, "x_min": ws[0]["x0"], "words": ws, "text": text})
    return lines

def left_label(line, right_edge: float) -> str:
    return " ".join(w["text"].strip()
                    for w in line["words"]
                    if w["x1"] <= right_edge and w["text"].strip()).strip()

def match_reason(label_text: str) -> Optional[str]:
    s = _lc(label_text)
    # regex first (anchored)
    for canon, rx in KEYWORD_TRIGGERS.items():
        if hasattr(rx, "search"):
            if rx.search(s): return canon
    # then substrings
    for canon, trig in KEYWORD_TRIGGERS.items():
        if isinstance(trig, str) and trig in s:
            return canon
    return None

def find_label_right_edge(page, lines) -> float:
    """
    1) Try header-based: leftmost 'P# YY' cell.
    2) Fallback: leftmost numeric token on page (the number columns).
    """
    words = page.extract_words(use_text_flow=True)
    header_xs = [float(w["x0"]) for w in words if HEADER_RX.search(w["text"])]
    if header_xs:
        return min(header_xs) - 12.0  # tighter than before
    # fallback: any numeric token
    numeric_xs = []
    for L in lines:
        for w in L["words"]:
            if re.fullmatch(r"-?\d+", w["text"].strip()):
                numeric_xs.append(float(w["x0"]))
    if numeric_xs:
        return min(numeric_xs) - 10.0
    # last resort: split page around median x
    return page.width * 0.45

def parse_pure(file_bytes: bytes):
    counts = defaultdict(lambda: defaultdict(int))  # reason -> period('TOTAL') -> value
    dbg = {"claimed": []}
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            lines = extract_lines(p)
            if not lines: continue
            right_edge = find_label_right_edge(p, lines)
            claimed = set()  # (xmid, ymid)
            i = 0
            while i < len(lines):
                L = lines[i]
                label = left_label(L, right_edge)
                if not label:
                    i += 1; continue
                canon = match_reason(label)
                if not canon:
                    # compensation 3-line join
                    s = _lc(label)
                    if ("compensation offered by" in s or "no/unsatisfactory" in s or "compensation" in s):
                        canon = COMP_CANON
                    else:
                        i += 1; continue
                got = 0
                yb = L["y"]
                # current line numbers
                for w in L["words"]:
                    t = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", t): continue
                    ymid = round((w["top"] + w["bottom"]) / 2, 1)
                    if abs(ymid - yb) > 0.6: continue
                    xmid = round((w["x0"] + w["x1"]) / 2, 1)
                    k = (xmid, ymid)
                    if k in claimed: continue
                    claimed.add(k)
                    counts[canon]["TOTAL"] += int(t)
                    got += 1
                # single-side fallback (next only)
                if got == 0 and i + 1 < len(lines):
                    N = lines[i+1]
                    for w in N["words"]:
                        t = w["text"].strip()
                        if not re.fullmatch(r"-?\d+", t): continue
                        ymid = round((w["top"] + w["bottom"]) / 2, 1)
                        if abs(ymid - N["y"]) > 0.6: continue
                        xmid = round((w["x0"] + w["x1"]) / 2, 1)
                        k = (xmid, ymid)
                        if k in claimed: continue
                        claimed.add(k)
                        counts[canon]["TOTAL"] += int(t)
                i += 1
    return counts, ["TOTAL"], dbg

def parse_full(file_bytes: bytes):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            lines = extract_lines(p)
            if not lines: continue
            right_edge = find_label_right_edge(p, lines)
            claimed = set()
            current_ad = None
            current_store = None
            current_section = None
            i = 0
            while i < len(lines):
                txt = lines[i]["text"].strip()
                # store
                if STORE_LINE_RX.match(txt):
                    current_store = txt
                    # AD fallback (previous left-aligned name)
                    # If needed later; safe to skip for now.
                    i += 1; continue
                # sections
                if SECTION_TOGO.match(txt):   current_section = "To Go";   i += 1; continue
                if SECTION_DELIV.match(txt):  current_section = "Delivery"; i += 1; continue
                if SECTION_DINEIN.match(txt): current_section = "Dine-In";  i += 1; continue

                label = left_label(lines[i], right_edge)
                if not label:
                    i += 1; continue

                canon = match_reason(label)
                if not canon:
                    s = _lc(label)
                    if ("compensation offered by" in s or "no/unsatisfactory" in s or "compensation" in s):
                        canon = COMP_CANON
                    else:
                        i += 1; continue

                # ensure section tag
                sec = current_section or "Unknown"

                got = 0
                yb = lines[i]["y"]
                for w in lines[i]["words"]:
                    t = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", t): continue
                    ymid = round((w["top"] + w["bottom"]) / 2, 1)
                    if abs(ymid - yb) > 0.6: continue
                    xmid = round((w["x0"] + w["x1"]) / 2, 1)
                    k = (xmid, ymid)
                    if k in claimed: continue
                    claimed.add(k)
                    data[current_ad][current_store][sec][canon]["TOTAL"] += int(t)
                    got += 1
                if got == 0 and i + 1 < len(lines):
                    N = lines[i+1]
                    for w in N["words"]:
                        t = w["text"].strip()
                        if not re.fullmatch(r"-?\d+", t): continue
                        ymid = round((w["top"] + w["bottom"]) / 2, 1)
                        if abs(ymid - N["y"]) > 0.6: continue
                        xmid = round((w["x0"] + w["x1"]) / 2, 1)
                        k = (xmid, ymid)
                        if k in claimed: continue
                        claimed.add(k)
                        data[current_ad][current_store][sec][canon]["TOTAL"] += int(t)
                i += 1
    return data, ["TOTAL"], {}

# ---------- run ----------
if pure_mode:
    counts, periods, _ = parse_pure(file_bytes)
    if not counts:
        st.error("No counts detected.")
        st.stop()
    df = pd.DataFrame(counts).T.fillna(0).astype(int)
    df["Grand Total"] = df.sum(axis=1)
    st.subheader("Pure Count — reason × TOTAL")
    st.dataframe(df.style.format(na_rep="0"), use_container_width=True)
else:
    data, periods, _ = parse_full(file_bytes)
    rows = []
    for ad, stores in data.items():
        for store, sects in stores.items():
            for sec, reasons in sects.items():
                for r, per in reasons.items():
                    rows.append({
                        "Area Director": ad or "",
                        "Store": store or "",
                        "Section": sec,
                        "Reason": r,
                        "Value": int(per.get("TOTAL", 0)),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No data parsed (after fallback).")
        st.stop()
    st.subheader("Totals by Section × Reason")
    pivot = df.pivot_table(index=["Section","Reason"], values="Value", aggfunc="sum").reset_index()
    st.dataframe(pivot, use_container_width=True)
