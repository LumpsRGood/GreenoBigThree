# app.py — Greeno Big Three v1.3.2 (To-go Missing Complaints focus)

import io, os, re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ── Config & Header ───────────────────────────────────────────────
st.set_page_config(page_title="Greeno Big Three v1.3.2", layout="wide")

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
    background-color:#0078C8; color:white; padding:2rem 2.5rem; border-radius:10px;
    display:flex; align-items:center; gap:2rem; box-shadow:0 4px 12px rgba(0,0,0,.2);
">
    {logo_html}
    <div style="display:flex; flex-direction:column; justify-content:center;">
        <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.3.2</h1>
        <div style="height:5px; background-color:#F44336; width:220px; margin-top:10px; border-radius:3px;"></div>
        <p style="margin:10px 0 0; opacity:.9; font-size:1.05rem;">
            To-go Missing Complaints extractor: pick a period, get the seven rows + Total.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Upload ────────────────────────────────────────────────
with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose a PDF report", type=["pdf"])
    st.caption("Looks for P# YY headers (e.g., P9 24).")

if not up:
    st.markdown("""
    <style>
      [data-testid="stSidebar"]{outline:3px solid #2e7df6;box-shadow:0 0 0 4px rgba(46,125,246,.25);
      animation:pulse 1.2s ease-in-out infinite;border-radius:6px;}
      @keyframes pulse{0%{outline-color:#2e7df6}50%{outline-color:#90caf9}100%{outline-color:#2e7df6}}
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;margin-top:8vh;"><div style="font-size:3rem;">⬅️</div>'
        '<div style="font-size:1.25rem;font-weight:600;margin-top:.5rem;">Upload your PDF in the <em>left sidebar</em></div>'
        '<div style="opacity:.85;margin-top:.25rem;">Click <strong>“Choose a PDF report”</strong> to begin.</div></div>',
        unsafe_allow_html=True)
    st.stop()

file_bytes = up.read()

# ── PDF parser (same core as v1.3) ────────────────────────────────
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
            t = ws[i]["text"]; x0, x1 = ws[i]["x0"], ws[i]["x1"]
            cand, x1c = t, x1
            if i + 1 < len(ws):
                t2 = ws[i + 1]["text"]; cand2 = f"{t} {t2}"
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

def parse_page_rows(page, headers: List[Tuple[str, float, float]]):
    if not headers: return []
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
        if ymid <= header_y: continue
        ws = sorted(ws, key=lambda w: w["x0"])
        label_tokens = [w["text"] for w in ws if w["x1"] <= label_right_edge]
        if not label_tokens: continue
        label = " ".join(label_tokens).strip()
        values = {h[0]: None for h in headers}
        for w in ws:
            if w["x0"] > label_right_edge:
                txt = w["text"].strip()
                if not re.fullmatch(r"-?\d+", txt): continue
                val = int(txt); xmid = (w["x0"] + w["x1"]) / 2
                nearest = min(header_positions, key=lambda hp: abs(hp[1] - xmid))[0]
                if values[nearest] is None:
                    values[nearest] = val
        if any(v is not None for v in values.values()):
            rows.append((label, values))
    return rows

def parse_pdf(file_bytes: bytes) -> pd.DataFrame:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not available.")
    all_rows: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    headers_seen: set = set()
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        carry_headers = None
        for page in pdf.pages:
            headers = find_headers_on_page(page) or carry_headers
            if not headers: continue
            carry_headers = headers[:]
            for htxt, _, _ in headers:
                headers_seen.add(htxt)
            for label, values in parse_page_rows(page, headers):
                for h, v in values.items():
                    if v is not None:
                        all_rows[label][h] += int(v)
    if not headers_seen or not all_rows:
        return pd.DataFrame()

    def header_sort_key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999, 999)

    ordered_headers = sorted(headers_seen, key=header_sort_key)
    records = []
    for label, vals in all_rows.items():
        row = {"Row": label}
        for h in ordered_headers:
            row[h] = vals.get(h, 0)
        records.append(row)
    df = pd.DataFrame(records)
    df = df[~df["Row"].str.match(r"^\s*(Total|.*Total:)\s*$", na=False)]
    return df.reset_index(drop=True)

# ── Parse PDF ─────────────────────────────────────────────────────
with st.spinner("Reading PDF and mapping period columns…"):
    df_wide = parse_pdf(file_bytes)

if df_wide.empty or df_wide.shape[1] <= 1:
    st.error("No readable period headers found (format like ‘P9 24’).")
    st.stop()

st.success("✅ PDF processed successfully!")
st.subheader("Preview (wide format)")
st.dataframe(df_wide.head(25), use_container_width=True)

# ── Focus set: To-go Missing Complaints ───────────────────────────
CATEGORY = "To-go Missing Complaints"
FOCUS_ROWS = [
    "Missing food",
    "Order wrong",
    "Missing condiments",
    "Out of menu item",
    "Missing bev",
    "Missing ingredients",
    "Packaging to-go complaint",
]

st.header("2) Pick the period column")
period_cols = [c for c in df_wide.columns if c != "Row"]
sel_col = st.selectbox("Period", options=period_cols, index=len(period_cols) - 1)

# Filter to focus rows (case-insensitive exact match)
df_focus = (
    df_wide.assign(Row_norm=df_wide["Row"].str.strip().str.casefold())
           .merge(pd.DataFrame({"Row_norm":[r.casefold() for r in FOCUS_ROWS],
                                "Keep":True}), on="Row_norm", how="left")
)
df_focus = df_focus[df_focus["Keep"].fillna(False)].drop(columns=["Row_norm","Keep"])

# Build narrow view + add Total row
df_out = df_focus[["Row", sel_col]].rename(columns={sel_col: "Value"}).copy()
total_val = int(df_out["Value"].fillna(0).sum())
df_out = pd.concat([df_out, pd.DataFrame([{"Row":"Total", "Value": total_val}])], ignore_index=True)

st.subheader(f"{CATEGORY} — {sel_col}")
st.dataframe(df_out, use_container_width=True)

# Controls
st.header("3) Filters & Export")
c1, c2 = st.columns(2)
with c1:
    hide_zeros = st.checkbox("Hide zero rows (excl. Total)", value=False)
with c2:
    sort_desc = st.checkbox("Sort by Value desc (excl. Total)", value=True)

df_show = df_out.copy()
if hide_zeros:
    df_show = pd.concat([
        df_out[df_out["Row"]!="Total"].loc[df_out["Value"].fillna(0)!=0],
        df_out[df_out["Row"]=="Total"]
    ])
if sort_desc:
    non_total = df_show[df_show["Row"]!="Total"].sort_values("Value", ascending=False)
    df_show = pd.concat([non_total, df_show[df_show["Row"]=="Total"]], ignore_index=True)

# Quick copy
copy_mode = st.radio("Quick copy format", ["Row → Value", "Row only", "Value only"], horizontal=True)
if copy_mode == "Row → Value":
    copy_text = "\n".join(f"{r} → {v}" for r, v in df_show.itertuples(index=False))
elif copy_mode == "Row only":
    copy_text = "\n".join(df_show["Row"].astype(str).tolist())
else:
    copy_text = "\n".join(df_show["Value"].astype(str).tolist())

st.text_area("Copy this", value=copy_text, height=200)

# Downloads
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
    to_excel_download(df_show, filename=f"togo_missing_{sel_col.replace(' ','_')}.xlsx")
with d2:
    to_csv_download(df_show, filename=f"togo_missing_{sel_col.replace(' ','_')}.csv")

with st.expander("ℹ️ Notes"):
    st.markdown(f"""
- Focused on **{CATEGORY}** only.
- The **Total** row is the sum of the seven items shown.
- Matching is **case-insensitive** exact match on these labels:
  {", ".join(FOCUS_ROWS)}.
""")
