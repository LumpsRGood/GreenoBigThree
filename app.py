import io
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception as e:
    pdfplumber = None

st.set_page_config(page_title="PDF Period Column Extractor", layout="wide")
st.title("📄 PDF Period Column Extractor")
st.caption("Reads text-style PDFs with headers like ‘P10 24 P11 24 … P1 25’ and extracts one column’s values for easy copy/export.")

# ----------------------- Utilities -----------------------
HEADER_RX = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")  # e.g., P9 24, P10 24, P1 25

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

def find_headers_on_page(page) -> List[Tuple[str, float, float]]:
    """
    Return list of (header_text, x_center, y_mid) for headers like 'P9 24'.
    Uses page.extract_words to get word boxes, then merges adjacent tokens (e.g., 'P9' + '24').
    """
    words = page.extract_words(x_tolerance=1, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
    # Build simple line-buckets by ymid to merge tokens that appear next to each other
    lines = defaultdict(list)
    for w in words:
        ymid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[ymid].append(w)
    headers = []
    for ymid, ws in lines.items():
        ws = sorted(ws, key=lambda w: w["x0"])
        # Create pairs of adjacent tokens to detect patterns like ['P9','24'] or a single token 'P9 24'
        merged = []
        i = 0
        while i < len(ws):
            t = ws[i]["text"]
            x0, x1 = ws[i]["x0"], ws[i]["x1"]
            candidate = t
            x1c = x1
            if i + 1 < len(ws):
                t2 = ws[i + 1]["text"]
                candidate2 = f"{t} {t2}"
                if HEADER_RX.fullmatch(candidate2):
                    x1c = ws[i + 1]["x1"]
                    candidate = candidate2
                    i += 2
                    merged.append((candidate, (x0 + x1c) / 2, ymid))
                    continue
            if HEADER_RX.fullmatch(candidate):
                merged.append((candidate, (x0 + x1) / 2, ymid))
            i += 1
        # If we found a line with many headers, that's our header row
        if len(merged) >= 3:
            headers.extend(merged)
    # De-duplicate by text (keep first occurrence vertically highest)
    seen = {}
    for txt, xc, ym in sorted(headers, key=lambda h: (h[2], h[1])):
        if txt not in seen:
            seen[txt] = (txt, xc, ym)
    return list(seen.values())

def parse_page_rows(page, headers: List[Tuple[str, float, float]]) -> List[Tuple[str, Dict[str, Optional[int]]]]:
    """
    Given detected headers (with x-centers), read rows below and map numbers under each header.
    Returns list of (row_label, {header: value_int_or_none}).
    """
    if not headers:
        return []

    words = page.extract_words(x_tolerance=1.5, y_tolerance=2.5, keep_blank_chars=False, use_text_flow=True)
    # Group into lines by y-mid
    lines = defaultdict(list)
    for w in words:
        ymid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[ymid].append(w)

    # Determine left boundary of label area as min header x minus some padding
    header_positions = sorted([(h[0], h[1]) for h in headers], key=lambda t: t[1])
    min_header_x = min(x for _, x in header_positions)
    label_right_edge = min_header_x - 12  # a little padding

    data_rows = []
    # We'll only consider lines below the header band (use header y to filter)
    header_y = min(h[2] for h in headers)
    for ymid, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        if ymid <= header_y:  # above header line: likely titles/metadata
            continue
        ws = sorted(ws, key=lambda w: w["x0"])
        # Split into label tokens (left of first header) and value tokens (under/near headers)
        label_tokens = [w["text"] for w in ws if w["x1"] <= label_right_edge]
        if not label_tokens:
            # Sometimes the row label wraps to previous line; skip lines with only numbers
            continue
        label = " ".join(label_tokens).strip()

        # Build values by closest header x
        values_for_row: Dict[str, Optional[int]] = {h[0]: None for h in headers}
        for w in ws:
            if w["x0"] > label_right_edge:
                txt = w["text"].strip()
                if not re.fullmatch(r"-?\d+", txt):
                    continue
                val = int(txt)
                # Assign to nearest header by x distance
                xmid = (w["x0"] + w["x1"]) / 2
                nearest = min(header_positions, key=lambda hp: abs(hp[1] - xmid))[0]
                # Prefer first fill; if multiple numbers land under same header on same line, keep the first
                if values_for_row[nearest] is None:
                    values_for_row[nearest] = val
        # Heuristic: consider it a real data row if at least one value landed
        if any(v is not None for v in values_for_row.values()):
            data_rows.append((label, values_for_row))
    return data_rows

def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse entire PDF into a wide dataframe:
      columns: ['Row'] + sorted headers (e.g., 'P9 24', 'P10 24', ..., 'Total' usually excluded)
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not available. Please install dependencies from requirements.txt.")

    all_rows: Dict[str, Dict[str, Optional[int]]] = defaultdict(dict)
    headers_seen: set = set()

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            headers = find_headers_on_page(page)
            if not headers:
                # Some pages might not have the header row (continuation); skip them
                continue
            # Capture headers this page knows about
            for htxt, _, _ in headers:
                headers_seen.add(htxt)

            rows = parse_page_rows(page, headers)
            for label, values in rows:
                # Merge values (some pages repeat the same label)
                for h, v in values.items():
                    if v is not None:
                        all_rows[label][h] = v

    if not headers_seen or not all_rows:
        return pd.DataFrame()

    # Build consistent column order: sort by (year, then period number)
    def header_sort_key(h: str):
        # h like "P9 24" -> (year=24, period=9)
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        if m:
            p = int(m.group(1))
            y = int(m.group(2))
            return (y, p)
        return (999, 999)

    ordered_headers = sorted(headers_seen, key=header_sort_key)
    # Create DataFrame
    rows_list = []
    for label, vals in all_rows.items():
        row = {"Row": label}
        for h in ordered_headers:
            row[h] = vals.get(h, None)
        rows_list.append(row)
    df = pd.DataFrame(rows_list)
    # Drop rows that are clearly totals or separators
    df = df[~df["Row"].str.match(r"^\s*(Total|.*Total:)\s*$", na=False)]
    return df

def to_excel_download(df: pd.DataFrame, filename="extracted.xlsx"):
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button("📥 Download Excel", data=buff.getvalue(), file_name=filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def to_csv_download(df: pd.DataFrame, filename="extracted.csv"):
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, file_name=filename, mime="text/csv")

# ----------------------- UI -----------------------
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose a PDF report", type=["pdf"])
    st.caption("The app reads text and character positions — no table extraction required.")

if not up:
    st.info("Upload a PDF to get started.")
    st.stop()

file_bytes = up.read()

with st.spinner("Parsing PDF into period columns…"):
    df_wide = parse_pdf(file_bytes)

if df_wide.empty or df_wide.shape[1] <= 1:
    st.error("I couldn't detect any period headers like ‘P# YY’ with numeric columns. "
             "If your report uses a different header format, tell me what it looks like and I’ll adjust the pattern.")
    st.stop()

st.success("Parsed!")
st.subheader("Preview (wide format)")
st.dataframe(df_wide.head(25), use_container_width=True)

# ------------- Column selection & extraction -------------
st.header("2) Pick the period column to extract")
period_cols = [c for c in df_wide.columns if c != "Row"]
sel_col = st.selectbox("Period", options=period_cols, index=len(period_cols)-1)

df_col = df_wide[["Row", sel_col]].rename(columns={sel_col: "Value"})

st.subheader(f"Selected column: {sel_col}")
st.dataframe(df_col.head(30), use_container_width=True)

# ------------- Filtering -------------
st.header("3) Filter to specific data")
c1, c2, c3 = st.columns([1,1,1])

with c1:
    nonzero_only = st.checkbox("Show non-zero only", value=True)
    min_val = st.number_input("Minimum value (inclusive)", value=0, step=1)
with c2:
    label_mode = st.radio("Row label filter", ["Any", "Contains", "Regex"], horizontal=True)
    label_query = st.text_input("Keyword or Regex (for label)", value="")
with c3:
    topn = st.number_input("Top N by value (after filters)", value=0, step=1, help="0 = no limit")

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if nonzero_only:
        out = out[out["Value"].fillna(0) != 0]
    if min_val:
        out = out[out["Value"].fillna(0) >= min_val]
    if label_mode == "Contains" and label_query.strip():
        out = out[out["Row"].str.contains(label_query.strip(), case=False, na=False)]
    elif label_mode == "Regex" and label_query.strip():
        try:
            out = out[out["Row"].str.contains(label_query, regex=True, na=False)]
        except re.error as e:
            st.warning(f"Invalid regex: {e}")
    # Sort by value desc, then label
    out = out.sort_values(["Value", "Row"], ascending=[False, True])
    if topn and topn > 0:
        out = out.head(int(topn))
    return out.reset_index(drop=True)

filtered = apply_filters(df_col)

st.subheader("Results")
if filtered.empty:
    st.warning("No rows match with the current filters.")
else:
    # Copy-friendly list
    st.markdown("**Quick copy list (Row → Value):**")
    st.text_area("Copy these values",
                 value="\n".join(f"{r} → {v}" for r, v in filtered.itertuples(index=False)),
                 height=220)

    st.dataframe(filtered, use_container_width=True, height=360)

    dl1, dl2 = st.columns(2)
    with dl1:
        to_excel_download(filtered, filename=f"extracted_{sel_col.replace(' ', '_')}.xlsx")
    with dl2:
        to_csv_download(filtered, filename=f"extracted_{sel_col.replace(' ', '_')}.csv")

# ----------------------- Tips -----------------------
with st.expander("Troubleshooting & tips"):
    st.markdown("""
- This parser looks for headers that match **`P# YY`** (e.g., `P9 24`, `P10 24`, `P1 25`).  
  If your format is slightly different (like `P09 2024`), tell me and I’ll tweak the pattern.
- It maps numbers beneath each header by **x-position proximity**. If alignment is messy,
  re-printing the report to PDF or using a consistent font can help.
- Use **‘Contains’** or **‘Regex’** to isolate certain issues (e.g., `Cold Food`, `Order wrong`).
- Export the filtered results to **Excel** or **CSV** for sharing or further analysis.
""")
