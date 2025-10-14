# Greeno Big Three v1.10.0 — Nuclear Mode
# Adds a robust "table extraction" path using Tabula (Java) or Camelot (Ghostscript/Poppler).
# If neither extractor is available, the app will explain what to install, and no crash.

import io, os, re, base64, tempfile, pathlib
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import pandas as pd
import streamlit as st

# Optional libraries (best effort)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ─────────────────────────── UI SHELL ───────────────────────────
st.set_page_config(page_title="Greeno Big Three v1.10.0 — Nuclear Mode", layout="wide")

def _load_logo():
    logo_path = "greenosu.webp"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

logo_b64 = _load_logo()
logo_html = (
    f'<img src="data:image/webp;base64,{logo_b64}" width="240" style="border-radius:12px;">'
    if logo_b64 else '<div style="width:240px;height:240px;background:#fff;border-radius:12px;"></div>'
)
st.markdown(
    f"""
<div style="background:#0078C8;color:#fff;padding:20px 24px;border-radius:12px;display:flex;gap:16px;align-items:center">
  {logo_html}
  <div>
    <div style="font-size:26px;font-weight:800;margin:0">Greeno Big Three v1.10.0 — Nuclear Mode</div>
    <div style="opacity:.9">Uses Tabula/Camelot to rip period columns into a matrix for exact matching</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    up = st.file_uploader("Upload PDF report", type=["pdf"])
    nuclear = st.toggle("☢ Nuclear Mode (Tabula/Camelot)", value=True,
                        help="Uses table extraction to reconstruct period columns. Requires tabula-py (Java) or camelot-py[cv] + system deps.")
    st.caption("This build totals by reason/period from table-extracted matrices (no AD/Store breakdown).")
    debug_view = st.checkbox("Show raw extracted table preview", value=False)

if not up:
    st.info("⬅️ Upload a PDF to begin.")
    st.stop()

file_bytes = up.read()

# ─────────────────────────── CANONICAL REASONS ───────────────────────────
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

# Distinct short substrings
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

def match_reason_strict(label_text: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", (label_text or "").strip().lower())
    for canon, rx in KEYWORD_REGEX.items():
        if rx.search(s):
            return canon
    for canon, keys in KEYWORD_SUBSTR.items():
        for k in keys:
            if k in s:
                return canon
    # 3-line wrap safety for compensation
    if ("compensation offered by" in s) or ("no/unsatisfactory" in s) or ("compensation" in s):
        return "No/insufficient compensation offered"
    return None

PERIOD_COL_RX = re.compile(r"^P(?:[1-9]|1[0-2])\s+\d{2}$", re.I)

def _coerce_int(x) -> int:
    if pd.isna(x):
        return 0
    t = str(x).strip()
    m = re.search(r"-?\d+", t)
    return int(m.group(0)) if m else 0

# ─────────────────────────── EXTRACTORS ───────────────────────────
def extract_tables_tabula(pdf_path: str) -> List[pd.DataFrame]:
    import tabula  # requires Java runtime
    dfs = tabula.read_pdf(
        pdf_path, pages="all", multiple_tables=True, stream=True,
        pandas_options={"dtype": str, "header": None}
    )
    return dfs or []

def extract_tables_camelot(pdf_path: str) -> List[pd.DataFrame]:
    import camelot  # requires Ghostscript/Poppler
    out = []
    for flavor in ("stream", "lattice"):
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor, strip_text="\n", line_scale=40)
            for t in tables:
                df = t.df
                df = df.applymap(lambda v: v if isinstance(v, str) else ("" if pd.isna(v) else str(v)))
                out.append(df)
            if out:
                break
        except Exception:
            continue
    return out

def nuclear_read_as_matrix(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (matrix, raw_all)
      matrix columns: ['label', 'P1 25', 'P2 25', ..., 'Total?'] as available
      raw_all: concatenated raw tables for debugging
    Raises RuntimeError if no extractor is available.
    """
    with tempfile.TemporaryDirectory() as td:
        pdf_path = str(pathlib.Path(td) / "in.pdf")
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        dfs = []
        errs = []

        # Try Tabula first (fast if Java is installed)
        try:
            dfs = extract_tables_tabula(pdf_path)
        except Exception as e:
            errs.append(f"Tabula failed: {e}")

        # Camelot fallback
        if not dfs:
            try:
                dfs = extract_tables_camelot(pdf_path)
            except Exception as e:
                errs.append(f"Camelot failed: {e}")

        if not dfs:
            raise RuntimeError(
                "No table extractor available.\n"
                "Install either:\n"
                "  • tabula-py  (requires Java runtime), or\n"
                "  • camelot-py[cv]  (requires Ghostscript/Poppler)\n\n"
                + ("; ".join(errs) if errs else "")
            )

        # Clean & concat
        cleaned = []
        for df in dfs:
            df = df.replace({None: "", pd.NA: "", "nan": ""})
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                continue
            df = df.applymap(lambda s: re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else s)
            cleaned.append(df)

        if not cleaned:
            raise RuntimeError("Extractor returned no usable tables.")

        raw_all = pd.concat(cleaned, ignore_index=True)

        # Build a generic wide matrix: for each row, choose a label cell (left-most texty cell),
        # and map any columns that look like period headers to numeric values.
        nrows, ncols = raw_all.shape

        # Detect per-column period names by scanning header band rows
        col_period_name: Dict[int, Optional[str]] = {j: None for j in range(ncols)}
        header_scan_rows = min(8, nrows)
        for j in range(ncols):
            for i in range(header_scan_rows):
                v = str(raw_all.iat[i, j]).strip()
                if PERIOD_COL_RX.match(v):
                    col_period_name[j] = v
                    break

        # If still none, attempt a row that holds multiple period tokens (true header row)
        if not any(col_period_name.values()):
            for i in range(header_scan_rows):
                tokens = [str(x).strip() for x in raw_all.iloc[i, :].tolist()]
                periods_in_row = [(idx, tok) for idx, tok in enumerate(tokens) if PERIOD_COL_RX.match(tok)]
                if len(periods_in_row) >= 3:
                    for idx, tok in periods_in_row:
                        col_period_name[idx] = tok
                    break

        # If we *still* have none, we cannot nuclear-map periods
        if not any(col_period_name.values()):
            raise RuntimeError("Could not locate any period headers like 'P9 25' in extracted tables.")

        period_cols = [j for j, name in col_period_name.items() if name]
        periods = [col_period_name[j] for j in period_cols]

        # Choose a label column per row: first non-empty cell from left that isn't itself a period token
        records = []
        for i in range(nrows):
            row_vals = [str(x).strip() for x in raw_all.iloc[i, :].tolist()]
            # label candidate
            label = None
            for j in range(ncols):
                cell = row_vals[j]
                if cell and not PERIOD_COL_RX.match(cell):
                    label = cell
                    break
            if not label:
                continue

            # Build period→value map (numbers only)
            values_map = {}
            for j in period_cols:
                cell = row_vals[j]
                if not cell:
                    continue
                val = _coerce_int(cell)
                # Only keep if row looks like a "reason" row (we'll filter via match_reason_strict later)
                if val != 0 or cell.isdigit():
                    values_map[col_period_name[j]] = val

            if values_map:
                records.append({"label": label, **values_map})

        if not records:
            raise RuntimeError("No rows with period-aligned numeric values found after extraction.")

        matrix = pd.DataFrame(records).fillna(0)

        # Deduplicate by keeping the most numeric-dense version of a label (common with multi-table splits)
        if "label" in matrix.columns:
            matrix = (
                matrix
                .assign(_nz=matrix.drop(columns=["label"]).astype(int).ne(0).sum(axis=1))
                .sort_values(["label", "_nz"], ascending=[True, False])
                .drop_duplicates(subset=["label"], keep="first")
                .drop(columns=["_nz"])
                .reset_index(drop=True)
            )

        return matrix, raw_all

# ─────────────────────────── RUN: NUCLEAR OR MESSAGE ───────────────────────────
if not nuclear:
    st.warning("Nuclear Mode is off. Turn it on in the sidebar to run table extraction.")
    st.stop()

try:
    with st.spinner("Extracting tables (Nuclear Mode)…"):
        matrix, raw_all = nuclear_read_as_matrix(file_bytes)
except RuntimeError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unexpected extraction error: {e}")
    st.stop()

if matrix.empty:
    st.error("Extraction succeeded, but produced no usable rows.")
    st.stop()

# Optional preview for debugging
if debug_view:
    st.subheader("Raw extracted (preview)")
    st.dataframe(raw_all.head(50), use_container_width=True)
    st.subheader("Matrix (label + periods)")
    st.dataframe(matrix.head(50), use_container_width=True)

# ─────────────────────────── NORMALIZATION & ROLLUPS ───────────────────────────
# Identify available periods (columns that look like 'P9 25')
period_cols = [c for c in matrix.columns if PERIOD_COL_RX.match(str(c))]
if not period_cols:
    st.error("No period columns detected in the matrix.")
    st.stop()

# Period picker
st.header("1) Pick the period")
period_cols_sorted = sorted(
    period_cols,
    key=lambda h: tuple(map(int, re.findall(r"\d+", h)))[-2:] if re.findall(r"\d+", h) else (999, 999)
)
sel_period = st.selectbox("Period", options=period_cols_sorted, index=len(period_cols_sorted) - 1)

# Classify each row into a canonical reason (or drop if not tracked)
def to_reason(label: str) -> Optional[str]:
    return match_reason_strict(label)

work = matrix.copy()
work["Reason"] = work["label"].map(to_reason)
work = work[work["Reason"].notna()].copy()

# Totals by reason for selected period
work[sel_period] = work[sel_period].apply(_coerce_int)
reason_totals = (
    work.groupby("Reason", as_index=True)[sel_period]
        .sum()
        .reindex(ALL_REASONS)
        .fillna(0)
        .astype(int)
        .to_frame(name="Total")
)

# Category rollups
cat_rows = [
    {"Category": "To-go Missing Complaints", "Total": int(reason_totals.loc[MISSING_REASONS]["Total"].sum())},
    {"Category": "Attitude",                  "Total": int(reason_totals.loc[ATTITUDE_REASONS]["Total"].sum())},
    {"Category": "Other",                     "Total": int(reason_totals.loc[OTHER_REASONS]["Total"].sum())},
]
cat_df = pd.DataFrame(cat_rows)
overall = int(cat_df["Total"].sum())

# ─────────────────────────── DISPLAY ───────────────────────────
st.markdown("### Quick glance (selected period)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall", overall)
c2.metric("To-go Missing", int(cat_df.loc[cat_df["Category"]=="To-go Missing Complaints","Total"].iloc[0]))
c3.metric("Attitude", int(cat_df.loc[cat_df["Category"]=="Attitude","Total"].iloc[0]))
c4.metric("Other", int(cat_df.loc[cat_df["Category"]=="Other","Total"].iloc[0]))

st.markdown("### Reason totals (selected period)")
rt = reason_totals.reset_index().rename(columns={"index":"Reason"})
st.dataframe(rt, use_container_width=True)

# ─────────────────────────── EXPORTS ───────────────────────────
st.header("2) Export")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    # Full matrix (all periods)
    matrix.to_excel(writer, sheet_name="Matrix (All Periods)", index=False)
    # Reason totals (selected)
    rt.to_excel(writer, sheet_name=f"Reason Totals ({sel_period})", index=False)
    # Category totals (selected)
    pd.DataFrame({"Category": cat_df["Category"], "Total": cat_df["Total"]}) \
      .to_excel(writer, sheet_name=f"Category Totals ({sel_period})", index=False)
st.download_button(
    "📥 Download Excel (Matrix + Totals)",
    data=buf.getvalue(),
    file_name=f"greeno_big_three_{sel_period.replace(' ','_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Nuclear Mode: counts come from table extraction; AD/Store/Section are not used here.")
