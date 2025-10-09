import io
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st
from base64 import b64encode
import os

try:
    import pdfplumber
except Exception as e:
    pdfplumber = None

# ----------------------- APP CONFIG -----------------------
st.set_page_config(page_title="Greeno Big Three v1.2", layout="wide")

# ----------------------- HEADER -----------------------
logo_path = "greenosu.webp"

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = b64encode(f.read()).decode("utf-8")
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
">
    {logo_html}
    <div style="display:flex; flex-direction:column; justify-content:center;">
        <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.2</h1>
        <div style="height:5px; background-color:#F44336; width:220px; margin-top:10px; border-radius:3px;"></div>
        <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
            Upload your PDF, pick a period column, and extract exactly what you need.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)



# ----------------------- UPLOAD AREA -----------------------
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose a PDF report", type=["pdf"])
    st.caption("Tip: Works best with text-based PDFs showing period headers like ‘P9 24’, ‘P1 25’, etc.")

# 🔹 Friendly “point-left” prompt when no upload yet
if not up:
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

# ----------------------- UTILITIES -----------------------
HEADER_RX = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

def _round_to(x: float, base: int = 2) -> float:
    return round(x / base) * base

def find_headers_on_page(page) -> List[Tuple[str, float, float]]:
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
        if len(merged) >= 3:
            headers.extend(merged)
    seen = {}
    for txt, xc, ym in sorted(headers, key=lambda h: (h[2], h[1])):
        if txt not in seen:
            seen[txt] = (txt, xc, ym)
    return list(seen.values())

def parse_page_rows(page, headers: List[Tuple[str, float, float]]) -> List[Tuple[str, Dict[str, Optional[int]]]]:
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
    data_rows = []
    header_y = min(h[2] for h in headers)
    for ymid, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        if ymid <= header_y:
            continue
        ws = sorted(ws, key=lambda w: w["x0"])
        label_tokens = [w["text"] for w in ws if w["x1"] <= label_right_edge]
        if not label_tokens:
            continue
        label = " ".join(label_tokens).strip()
        values_for_row = {h[0]: None for h in headers}
        for w in ws:
            if w["x0"] > label_right_edge:
                txt = w["text"].strip()
                if not re.fullmatch(r"-?\d+", txt):
                    continue
                val = int(txt)
                xmid = (w["x0"] + w["x1"]) / 2
                nearest = min(header_positions, key=lambda hp: abs(hp[1] - xmid))[0]
                if values_for_row[nearest] is None:
                    values_for_row[nearest] = val
        if any(v is not None for v in values_for_row.values()):
            data_rows.append((label, values_for_row))
    return data_rows

def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not available.")
    all_rows: Dict[str, Dict[str, Optional[int]]] = defaultdict(dict)
    headers_seen: set = set()
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            headers = find_headers_on_page(page)
            if not headers:
                continue
            for htxt, _, _ in headers:
                headers_seen.add(htxt)
            rows = parse_page_rows(page, headers)
            for label, values in rows:
                for h, v in values.items():
                    if v is not None:
                        all_rows[label][h] = v
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
    rows_list = []
    for label, vals in all_rows.items():
        row = {"Row": label}
        for h in ordered_headers:
            row[h] = vals.get(h, None)
        rows_list.append(row)
    df = pd.DataFrame(rows_list)
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

# ----------------------- PARSE & UI -----------------------
with st.spinner("Reading PDF and mapping period columns..."):
    df_wide = parse_pdf(file_bytes)

if df_wide.empty or df_wide.shape[1] <= 1:
    st.error("No readable period headers found. If your report format differs, let’s adjust the pattern.")
    st.stop()

st.success("✅ PDF processed successfully!")

st.subheader("Preview (wide format)")
st.dataframe(df_wide.head(25), use_container_width=True)

st.header("2) Pick the period column to extract")
period_cols = [c for c in df_wide.columns if c != "Row"]
sel_col = st.selectbox("Period", options=period_cols, index=len(period_cols)-1)

df_col = df_wide[["Row", sel_col]].rename(columns={sel_col: "Value"})
st.subheader(f"Selected column: {sel_col}")
st.dataframe(df_col.head(30), use_container_width=True)

st.header("3) Filter to specific data")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    nonzero_only = st.checkbox("Show non-zero only", value=True)
    min_val = st.number_input("Minimum value (inclusive)", value=0, step=1)
with c2:
    label_mode = st.radio("Row label filter", ["Any", "Contains", "Regex"], horizontal=True)
    label_query = st.text_input("Keyword or Regex (for label)", value="")
with c3:
    topn = st.number_input("Top N by value", value=0, step=1, help="0 = no limit")

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
    out = out.sort_values(["Value", "Row"], ascending=[False, True])
    if topn and topn > 0:
        out = out.head(int(topn))
    return out.reset_index(drop=True)

filtered = apply_filters(df_col)

st.subheader("Results")
if filtered.empty:
    st.warning("No rows match your filters.")
else:
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

with st.expander("ℹ️ Tips & Troubleshooting"):
    st.markdown("""
- Looks for headers in the format **P# YY** (e.g., `P9 24`, `P1 25`).  
- Maps numbers beneath each header by **x-position proximity**.  
- Re-print to PDF if alignment looks off.  
- Use **‘Contains’** or **‘Regex’** to target specific row labels.  
- Export results directly to **Excel** or **CSV**.
""")
