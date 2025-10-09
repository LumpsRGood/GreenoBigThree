# app.py — Greeno Big Three v1.3

import io
import re
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ───────────────── CONFIG & HEADER ─────────────────
st.set_page_config(page_title="Greeno Big Three v1.4.2", layout="wide")

# Your local logo file
logo_path = "greenosu.webp"
if os.path.exists(logo_path):
    import base64
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f'<img src="data:image/webp;base64,{logo_data}" width="240" style="border-radius:12px;">'
else:
    logo_html = '<div style="width:240px; height:240px; background:#fff; border-radius:12px;"></div>'

st.markdown(f"""
<div style="
    background-color:#0078C8;
    color:white;
    padding:2rem 2.5rem;
    border-radius:10px;
    display:flex;
    align-items:center;
    gap:2rem;
    box-shadow:0 4px 12px rgba(0,0,0,.2);
    position:sticky; top:0; z-index:50;
">
    {logo_html}
    <div style="display:flex; flex-direction:column; justify-content:center;">
        <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.4.2</h1>
        <div style="height:5px; background-color:#F44336; width:220px; margin-top:10px; border-radius:3px;"></div>
        <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
            Upload your PDF, pick a period column, and extract exactly what you need.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ───────────────── SIDEBAR: Upload ─────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose a PDF report", type=["pdf"])
    st.caption("Tip: Works best with text-based PDFs showing period headers like ‘P9 24’, ‘P1 25’, etc.")

if not up:
    # Pulse the sidebar + left-arrow helper
    st.markdown("""
    <style>
      [data-testid="stSidebar"] {
        outline: 3px solid #2e7df6;
        box-shadow: 0 0 0 4px rgba(46,125,246,.25);
        animation: pulse 1.2s ease-in-out infinite;
        border-radius: 6px;
      }
      @keyframes pulse {
        0% { outline-color: #2e7df6; }
        50% { outline-color: #90caf9; }
        100% { outline-color: #2e7df6; }
      }
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

# ───────────────── PDF PARSER ─────────────────
HEADER_RX = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")  # P9 24, P10 24, P1 25, etc.

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

def find_headers_on_page(page) -> List[Tuple[str, float, float]]:
    """
    Returns [(header_text, x_center, y_mid)] for headers like 'P9 24'.
    """
    words = page.extract_words(x_tolerance=1, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
    lines = defaultdict(list)
    for w in words:
        ymid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[ymid].append(w)

    headers = []
    for ymid, ws in lines.items():
        ws = sorted(ws, key=lambda w: w["x0"])
        merged = []
        i = 0
        while i < len(ws):
            t = ws[i]["text"]
            x0, x1 = ws[i]["x0"], ws[i]["x1"]
            cand = t
            x1c = x1
            if i + 1 < len(ws):
                t2 = ws[i + 1]["text"]
                cand2 = f"{t} {t2}"
                if HEADER_RX.fullmatch(cand2):
                    x1c = ws[i + 1]["x1"]
                    cand = cand2
                    i += 2
                    merged.append((cand, (x0 + x1c) / 2, ymid))
                    continue
            if HEADER_RX.fullmatch(cand):
                merged.append((cand, (x0 + x1) / 2, ymid))
            i += 1
        # A true header line usually has many matches
        if len(merged) >= 3:
            headers.extend(merged)

    # Dedup by text, keep first occurrence
    seen = {}
    for txt, xc, ym in sorted(headers, key=lambda h: (h[2], h[1])):
        if txt not in seen:
            seen[txt] = (txt, xc, ym)
    return list(seen.values())

def parse_page_rows(page, headers: List[Tuple[str, float, float]]) -> List[Tuple[str, Dict[str, Optional[int]]]]:
    """
    For a page, returns a list of (row_label, {header: value or None}).
    """
    if not headers:
        return []

    words = page.extract_words(x_tolerance=1.5, y_tolerance=2.5, keep_blank_chars=False, use_text_flow=True)
    lines = defaultdict(list)
    for w in words:
        ymid = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[ymid].append(w)

    header_positions = sorted([(h[0], h[1]) for h in headers], key=lambda t: t[1])
    min_header_x = min(x for _, x in header_positions)
    label_right_edge = min_header_x - 12
    header_y = min(h[2] for h in headers)

    rows = []
    for ymid, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        if ymid <= header_y:
            continue
        ws = sorted(ws, key=lambda w: w["x0"])
        label_tokens = [w["text"] for w in ws if w["x1"] <= label_right_edge]
        if not label_tokens:
            continue
        label = " ".join(label_tokens).strip()

        values = {h[0]: None for h in headers}
        for w in ws:
            if w["x0"] > label_right_edge:
                txt = w["text"].strip()
                if not re.fullmatch(r"-?\d+", txt):
                    continue
                val = int(txt)
                xmid = (w["x0"] + w["x1"]) / 2
                nearest = min(header_positions, key=lambda hp: abs(hp[1] - xmid))[0]
                if values[nearest] is None:
                    values[nearest] = val

        if any(v is not None for v in values.values()):
            rows.append((label, values))
    return rows

def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    """
    Build a wide DataFrame: columns ['Row', 'P9 24', ... 'P9 25'].
    - Sums duplicate row labels across pages.
    - Ignores 'Total' since it doesn't match HEADER_RX.
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not available. Install dependencies and retry.")

    all_rows: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    headers_seen: set = set()

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        carry_headers = None
        for page in pdf.pages:
            headers = find_headers_on_page(page)
            if not headers and carry_headers:
                headers = carry_headers  # carry-over if a page lacks the header line
            if not headers:
                continue
            carry_headers = headers[:]  # remember for next pages
            for htxt, _, _ in headers:
                headers_seen.add(htxt)

            rows = parse_page_rows(page, headers)
            for label, values in rows:
                for h, v in values.items():
                    if v is not None:
                        all_rows[label][h] += int(v)  # sum across pages

    if not headers_seen or not all_rows:
        return pd.DataFrame()

    def header_sort_key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        if m:
            p = int(m.group(1))
            y = int(m.group(2))
            return (y, p)
        return (999, 999)

    ordered_headers = sorted(headers_seen, key=header_sort_key)

    records = []
    for label, vals in all_rows.items():
        row = {"Row": label}
        for h in ordered_headers:
            row[h] = vals.get(h, 0)
        records.append(row)

    df = pd.DataFrame(records)
    # Drop obvious total lines
    df = df[~df["Row"].str.match(r"^\s*(Total|.*Total:)\s*$", na=False)]
    return df.reset_index(drop=True)

# ───────────────── Parse PDF ─────────────────
with st.spinner("Reading PDF and mapping period columns…"):
    df_wide = parse_pdf(file_bytes)

if df_wide.empty or df_wide.shape[1] <= 1:
    st.error("No readable period headers found (format like ‘P9 24’). If your report differs, tell me and I’ll tweak it.")
    st.stop()

st.success("✅ PDF processed successfully!")
st.subheader("Preview (wide format)")
st.dataframe(df_wide.head(25), use_container_width=True)

# ───────────────── Column pick ─────────────────
st.header("2) Pick the period column to extract")
period_cols = [c for c in df_wide.columns if c != "Row"]
sel_col = st.selectbox("Period", options=period_cols, index=len(period_cols) - 1)

df_col = df_wide[["Row", sel_col]].rename(columns={sel_col: "Value"})
st.subheader(f"Selected column: {sel_col}")
st.dataframe(df_col.head(30), use_container_width=True)

# ───────────────── Filters ─────────────────
st.header("3) Filter to specific data")

c1, c2, c3 = st.columns(3)
with c1:
    nonzero_only = st.checkbox("Hide zeros", value=True)
    op = st.selectbox("Threshold operator", [">=", ">", "=", "<=", "<"], index=0)
    threshold = st.number_input("Threshold value", value=0, step=1)
with c2:
    label_mode = st.radio("Row label filter", ["Any", "Contains", "Regex"], horizontal=True)
    label_query = st.text_input("Keyword / Regex", value="")
with c3:
    exact_list_raw = st.text_area("Exact row list (optional)\nOne label per line", height=130)
    topn = st.number_input("Top N by value (after filters)", value=0, step=1, help="0 = no limit")

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Exact list wins if provided
    exact_list = [ln.strip() for ln in exact_list_raw.splitlines() if ln.strip()]
    if exact_list:
        out = out[out["Row"].isin(exact_list)]

    # Label filter
    if not exact_list:
        if label_mode == "Contains" and label_query.strip():
            out = out[out["Row"].str.contains(label_query.strip(), case=False, na=False)]
        elif label_mode == "Regex" and label_query.strip():
            try:
                out = out[out["Row"].str.contains(label_query, regex=True, na=False)]
            except re.error as e:
                st.warning(f"Invalid regex: {e}")

    # Zero/threshold filters
    val = out["Value"].fillna(0)
    if nonzero_only:
        out = out[val != 0]

    if op == ">=":
        out = out[val >= threshold]
    elif op == ">":
        out = out[val > threshold]
    elif op == "=":
        out = out[val == threshold]
    elif op == "<=":
        out = out[val <= threshold]
    elif op == "<":
        out = out[val < threshold]

    out = out.sort_values(["Value", "Row"], ascending=[False, True]).reset_index(drop=True)
    if topn and topn > 0:
        out = out.head(int(topn))
    return out

filtered = apply_filters(df_col)

st.subheader("Results")
if filtered.empty:
    st.warning("No rows match your filters.")
else:
    # Copy format selector
    copy_fmt = st.radio("Quick copy format", ["Row → Value", "Row only", "Value only"], horizontal=True, index=0)
    if copy_fmt == "Row → Value":
        txt = "\n".join(f"{r} → {v}" for r, v in filtered.itertuples(index=False))
    elif copy_fmt == "Row only":
        txt = "\n".join(f"{r}" for r in filtered["Row"])
    else:
        txt = "\n".join(str(v) for v in filtered["Value"])

    st.text_area("Copy this", value=txt, height=220)
    st.dataframe(filtered, use_container_width=True, height=380)

    def to_excel_download(df: pd.DataFrame, filename="extracted.xlsx"):
        buff = io.BytesIO()
        with pd.ExcelWriter(buff, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        st.download_button("📥 Download Excel", data=buff.getvalue(), file_name=filename,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    def to_csv_download(df: pd.DataFrame, filename="extracted.csv"):
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name=filename, mime="text/csv")

    d1, d2 = st.columns(2)
    with d1:
        to_excel_download(filtered, filename=f"extracted_{sel_col.replace(' ', '_')}.xlsx")
    with d2:
        to_csv_download(filtered, filename=f"extracted_{sel_col.replace(' ', '_')}.csv")

with st.expander("ℹ️ Tips & Troubleshooting"):
    st.markdown("""
- Looks for headers in the format **P# YY** (e.g., `P9 24`, `P1 25`). `Total` is ignored.
- If a page is missing the header line, the parser **carries over** headers from the prior page.
- Numbers below headers are mapped by **x-position proximity**. If alignment looks off, re-print to PDF.
- Use **Exact row list** to target a specific set of labels (one per line).
- Export directly to **Excel** or **CSV**.
""")
