# Greeno Big Three v1.9.5-safe — Stable pure-count baseline
# - Ignores AD/Store/Section and periods; counts by robust text triggers
# - Anchored regex for Missing Item (Food/Bev) & Condiments
# - Tight y-band + de-dupe to avoid bleed
# - Category rollups + Excel export

import io, re, base64, os
from collections import defaultdict
from typing import Optional, List, Dict
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────── UI ─────────────
st.set_page_config(page_title="Greeno Big Three v1.9.5-safe", layout="wide")

logo_path = "greenosu.webp"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f'<img src="data:image/webp;base64,{logo_data}" width="240" style="border-radius:12px;">'
else:
    logo_html = '<div style="width:240px;height:240px;background:#fff;border-radius:12px;"></div>'

st.markdown(
    f"""
<div style="background:#0078C8;color:#fff;padding:20px 24px;border-radius:12px;display:flex;gap:16px;align-items:center">
  {logo_html}
  <div>
    <div style="font-size:26px;font-weight:800;margin:0">Greeno Big Three v1.9.5-safe</div>
    <div style="opacity:.9">Stable baseline — pure count only, robust triggers, Excel export</div>
  </div>
</div>
""", unsafe_allow_html=True
)

with st.sidebar:
    up = st.file_uploader("Upload PDF report", type=["pdf"])
    st.caption("This safe build totals by reason only (no AD/Store/Period).")

if not up:
    st.info("⬅️ Upload a PDF to begin.")
    st.stop()

if pdfplumber is None:
    st.error("pdfplumber is not installed. Run: pip install pdfplumber")
    st.stop()

file_bytes = up.read()

# ───────────── Canonical reasons ─────────────
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
ALL_REASONS = MISSING_REASONS + ATTITUDE_REASONS + OTHER_REASONS

# Anchored regex first (very specific)
KEYWORD_REGEX = {
    "Missing food":               re.compile(r"\bmissing\s+item\s*\(food\)", re.I),
    "Missing bev":                re.compile(r"\bmissing\s+item\s*\(bev\)",  re.I),
    "Missing condiments":         re.compile(r"\bmissing\s+condiments?",     re.I),
    "Missing ingredients":        re.compile(r"\bmissing\s+ingredient",      re.I),
    "Out of menu item":           re.compile(r"\bout\s+of\s+menu\s+item",    re.I),
    "Packaging to-go complaint":  re.compile(r"\bpackaging\s+to-?\s*go",     re.I),
}

# Unique substrings (short but distinctive)
KEYWORD_SUBSTR = {
    "Order wrong":                          ["order wrong"],
    "Unprofessional/Unfriendly":            ["unfriendly"],
    "Manager directly involved":            ["directly involved"],
    "Manager not available":                ["manager not available"],
    "Manager did not visit":                ["did not visit", "no visit"],
    "Negative mgr-employee exchange":       ["manager-employee", "negative"],
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

# ───────────── Text extraction helpers ─────────────
def extract_lines(page) -> List[Dict]:
    words = page.extract_words(x_tolerance=1.4, y_tolerance=2.4, use_text_flow=True)
    buckets = defaultdict(list)
    for w in words:
        ymid = round((w["top"] + w["bottom"]) / 2, 1)
        buckets[ymid].append(w)
    lines = []
    for y, ws in sorted(buckets.items(), key=lambda kv: kv[0]):
        ws = sorted(ws, key=lambda w: w["x0"])
        text = " ".join(w["text"].strip() for w in ws if w["text"].strip())
        if text:
            lines.append({"y": y, "words": ws, "text": text})
    return lines

def label_and_numbers(line, split_x: float):
    """Return (label_text, number_words_on_same_line) using split_x."""
    label = " ".join(w["text"].strip() for w in line["words"] if w["x1"] <= split_x and w["text"].strip()).strip()
    nums  = [w for w in line["words"] if w["x0"] > split_x and re.fullmatch(r"-?\d+", w["text"].strip())]
    return label, nums

def infer_split_x(page, lines) -> float:
    """Find column split:
       1) try leftmost number on page; else 2) 45% page width."""
    xs = []
    for L in lines:
        for w in L["words"]:
            if re.fullmatch(r"-?\d+", w["text"].strip()):
                xs.append(float(w["x0"]))
    if xs: return min(xs) - 10.0
    return page.width * 0.45

def match_reason(label_text: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", label_text.strip().lower())
    # regex first
    for canon, rx in KEYWORD_REGEX.items():
        if rx.search(s):
            return canon
    # substrings second
    for canon, keys in KEYWORD_SUBSTR.items():
        for k in keys:
            if k in s:
                return canon
    return None

# ───────────── Counting (pure, resilient) ─────────────
counts = defaultdict(int)  # reason -> TOTAL

with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
    for page in pdf.pages:
        lines = extract_lines(page)
        if not lines:
            continue
        split_x = infer_split_x(page, lines)
        claimed = set()  # (xmid, ymid) to avoid double-counting
        i = 0
        while i < len(lines):
            L = lines[i]
            label, nums = label_and_numbers(L, split_x)
            if not label:
                i += 1; continue

            canon = match_reason(label)
            # compensation 3-line join safety
            if not canon:
                s = label.lower()
                if ("compensation offered by" in s) or ("no/unsatisfactory" in s) or ("compensation" in s):
                    canon = "No/insufficient compensation offered"

            if not canon:
                i += 1
                continue

            got = 0
            # same-line numbers (tight y-band)
            yb = L["y"]
            for w in nums:
                t = w["text"].strip()
                ymid = round((w["top"] + w["bottom"]) / 2, 1)
                if abs(ymid - yb) > 0.6:
                    continue
                xmid = round((w["x0"] + w["x1"]) / 2, 1)
                key = (xmid, ymid)
                if key in claimed:
                    continue
                claimed.add(key)
                counts[canon] += int(t)
                got += 1

            # single-neighbor fallback (next line only) if none on same row
            if got == 0 and i + 1 < len(lines):
                N = lines[i + 1]
                _label_next, nums_next = label_and_numbers(N, split_x)
                for w in nums_next:
                    t = w["text"].strip()
                    ymid = round((w["top"] + w["bottom"]) / 2, 1)
                    if abs(ymid - N["y"]) > 0.6:
                        continue
                    xmid = round((w["x0"] + w["x1"]) / 2, 1)
                    key = (xmid, ymid)
                    if key in claimed:
                        continue
                    claimed.add(key)
                    counts[canon] += int(t)

            i += 1

# ───────────── Display & export ─────────────
if not counts:
    st.error("No data parsed.")
    st.stop()

# reason table
idx = MISSING_REASONS + ATTITUDE_REASONS + OTHER_REASONS
df = pd.DataFrame({"Reason": idx})
df["TOTAL"] = df["Reason"].map(lambda r: counts.get(r, 0)).astype(int)

# category rollups
cat_rows = []
cat_rows.append({"Category":"To-go Missing Complaints","TOTAL":int(df[df["Reason"].isin(MISSING_REASONS)]["TOTAL"].sum())})
cat_rows.append({"Category":"Attitude","TOTAL":int(df[df["Reason"].isin(ATTITUDE_REASONS)]["TOTAL"].sum())})
cat_rows.append({"Category":"Other","TOTAL":int(df[df["Reason"].isin(OTHER_REASONS)]["TOTAL"].sum())})
cat_df = pd.DataFrame(cat_rows)
overall = int(cat_df["TOTAL"].sum())

st.markdown("### Category totals (overall)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall", overall)
col2.metric("To-go Missing Complaints", int(cat_df.loc[cat_df["Category"]=="To-go Missing Complaints","TOTAL"].iloc[0]))
col3.metric("Attitude", int(cat_df.loc[cat_df["Category"]=="Attitude","TOTAL"].iloc[0]))
col4.metric("Other", int(cat_df.loc[cat_df["Category"]=="Other","TOTAL"].iloc[0]))

st.markdown("### Reason totals (overall)")
st.dataframe(df, use_container_width=True)

# Excel export
buff = io.BytesIO()
with pd.ExcelWriter(buff, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Reason Totals (Overall)", index=False)
    cat_df.to_excel(writer, sheet_name="Category Totals (Overall)", index=False)
st.download_button(
    "📥 Download Excel (Overall Totals)",
    data=buff.getvalue(),
    file_name="greeno_big_three_overall_totals.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
