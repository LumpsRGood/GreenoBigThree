# path: app.py
# Streamlit: PDF -> CSV with metric mapping (labels + sections), ready for Streamlit Cloud.

from __future__ import annotations

import io
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ----- Page config -----
st.set_page_config(page_title="PDF → CSV (Metric Mapping)", page_icon="📄", layout="wide")

# ----- Optional OCR (why: image-only PDFs) -----
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# ----- Required PDF text extraction -----
try:
    import pdfplumber
except Exception:
    st.error("Missing dependency: pdfplumber. Add `pdfplumber` to requirements.txt.")
    st.stop()

# ----- Default Mapping JSON (paste your rules here) -----
DEFAULT_MAPPING_JSON = r"""
{
  "case_insensitive": true,
  "metrics": [
    { "label": "Missing food", "patterns": ["Missing Item (Food)"], "regex": false, "sections": ["To-Go", "Delivery"], "section_aggregate": "sum" },
    { "label": "Order wrong", "patterns": ["Order Wrong"], "regex": false, "sections": ["To-Go", "Delivery"], "section_aggregate": "sum" },
    { "label": "Missing condiments", "patterns": ["Missing Condiments"], "regex": false, "sections": ["To-Go", "Delivery"], "section_aggregate": "sum" },
    { "label": "Out of menu item", "patterns": ["Out Of Menu Item"], "regex": false, "sections": ["To-Go", "Delivery"], "section_aggregate": "sum" },
    { "label": "Missing bev", "patterns": ["Missing Item (Bev)"], "regex": false, "sections": ["To-Go", "Delivery"], "section_aggregate": "sum" },
    { "label": "Missing ingredients", "patterns": ["Missing Ingredient (Food)"], "regex": false, "sections": ["To-Go", "Delivery"], "section_aggregate": "sum" },
    { "label": "Packaging to-go complaint", "patterns": ["Packaging To Go Complaint"], "regex": false, "sections": ["To-Go", "Delivery"], "section_aggregate": "sum" },

    { "label": "Unprofessional/Unfriendly", "patterns": ["Unprofessional Behavior", "Unfriendly Attitude"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager directly involved", "patterns": ["Manager Directly Involved In Complaint"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager not available", "patterns": ["Management Not Available"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager did not visit", "patterns": ["Manager Did Not Visit"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Negative mgr-employee exchange", "patterns": ["Negative Manager-Employee Interaction"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager did not follow up", "patterns": ["Manager Did Not Follow Up"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Argued with guest", "patterns": ["Argued With Guest"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },

    { "label": "Long hold/no answer", "patterns": ["Long Hold/No Answer/Hung Up"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "No/insufficient compensation offered", "patterns": ["No/Unsatisfactory Compensation Offered By Restaurant"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Did not attempt to resolve", "patterns": ["Did Not Attempt To Resolve Issue"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Guest left without ordering", "patterns": ["Guest Left Without Dining or Ordering"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Unknowledgeable", "patterns": ["Unknowledgeable"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Did not open on time", "patterns": ["Didn[’']t Open/close On Time"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "No/poor apology", "patterns": ["No/Poor Apology"], "regex": false, "sections": ["*"], "section_aggregate": "sum" }
  ]
}
"""

# ----- Config -----
@dataclass
class ExtractConfig:
    use_ocr: bool = False
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    remove_empty_lines: bool = True
    hyphenation_fix: bool = False
    drop_header_lines: int = 0
    drop_footer_lines: int = 0
    remove_page_numbers: bool = True

# ----- Extraction / Cleaning -----
def extract_pdf_text(file: io.BytesIO, use_ocr: bool) -> Tuple[str, List[str]]:
    if use_ocr and not _HAS_TESSERACT:
        st.warning("OCR selected but pytesseract is unavailable; using native text.")
        use_ocr = False
    pages: List[str] = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if use_ocr:
                try:
                    img = page.to_image(resolution=300).original
                    txt = pytesseract.image_to_string(img)
                except Exception as e:
                    st.warning(f"OCR failed; using native text. ({e})")
                    txt = page.extract_text(layout=True) or ""
            else:
                txt = page.extract_text(layout=True) or ""
            pages.append(txt or "")
    return "\n<<<PAGE_BREAK>>>\n".join(pages), pages

def normalize_text(s: str, cfg: ExtractConfig) -> str:
    if cfg.normalize_unicode:
        s = unicodedata.normalize("NFKC", s)
    if cfg.hyphenation_fix:
        s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # join wrapped words
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.collapse_whitespace:
        s = "\n".join(re.sub(r"[ \t]+", " ", ln) for ln in s.split("\n"))
    if cfg.remove_empty_lines:
        s = "\n".join([ln for ln in s.split("\n") if ln.strip()])
    return s

def strip_headers_footers(text: str, cfg: ExtractConfig) -> str:
    chunks = text.split("\n<<<PAGE_BREAK>>>\n")
    out = []
    for ch in chunks:
        lines = ch.split("\n")
        if cfg.drop_header_lines > 0:
            lines = lines[cfg.drop_header_lines:]
        if cfg.drop_footer_lines > 0 and cfg.drop_footer_lines < len(lines):
            lines = lines[:-cfg.drop_footer_lines]
        out.append("\n".join(lines))
    return "\n<<<PAGE_BREAK>>>\n".join(out)

def remove_page_numbers(text: str) -> str:
    keep = []
    for ln in text.split("\n"):
        t = ln.strip()
        if re.fullmatch(r"Page\s+\d+(?:\s*/\s*\d+)?", t, flags=re.I):
            continue
        if re.fullmatch(r"\d{1,4}", t):
            continue
        keep.append(ln)
    return "\n".join(keep)

# ----- Matrix parser (Reason for Contact … P9 24 … Total) -----
SECTION_ALIASES = {
    "delivery": "Delivery",
    "dine in": "Dine-In",
    "dine-in": "Dine-In",
    "carryout": "Carryout",
    "carry out": "Carryout",
    "takeout": "Takeout",
    "to go": "To-Go",
    "to-go": "To-Go",
}
def norm_section(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    k = s.strip().lower()
    return SECTION_ALIASES.get(k, s.strip())

def extract_period_labels(text: str, expected: int = 14) -> List[str]:
    for ln in text.splitlines():
        if ("Reason for Contact" in ln) and ("Total" in ln):
            parts = re.findall(r"P\d{1,2}\s+\d{2}", ln)
            if len(parts) >= expected - 1:
                labels = [p.replace(" ", "_") for p in parts[: expected - 1]]
                labels.append("Total")
                return labels
    return [f"col{i:02d}" for i in range(1, expected + 1)]

def metric_line_pattern(ncols: int = 14) -> re.Pattern:
    nums = r"\s+".join([fr"(?P<c{i:02d}>\d+)" for i in range(1, ncols + 1)])
    return re.compile(fr"^(?P<metric>[A-Za-z][A-Za-z0-9/'&()\- ]+?)\s*:?[\s]+{nums}\s*$")

def parse_matrix_blocks(text: str, ncols: int = 14) -> Tuple[pd.DataFrame, List[str]]:
    labels = extract_period_labels(text, expected=ncols)
    pat = metric_line_pattern(ncols)
    rows: List[Dict[str, Any]] = []
    section: Optional[str] = None
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.lower() in SECTION_ALIASES:
            section = norm_section(ln)
            continue
        m = pat.match(ln)
        if not m:
            continue
        gd = m.groupdict()
        metric_name = gd.pop("metric").rstrip(":").strip()
        vals = [gd[f"c{i:02d}"] for i in range(1, ncols + 1)]
        row = {"section": norm_section(section), "metric": metric_name}
        for j, lab in enumerate(labels):
            v = vals[j]
            row[lab] = int(v) if v is not None and v.isdigit() else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["section", "metric"] + labels]
    return df, labels

# ----- Mapping engine -----
def load_mapping(text: str) -> Dict[str, Any]:
    try:
        cfg = json.loads(text) if text.strip() else {"metrics": []}
        if "metrics" not in cfg or not isinstance(cfg["metrics"], list):
            raise ValueError("Missing 'metrics' array.")
        return cfg
    except Exception as e:
        st.error(f"Mapping JSON error: {e}")
        return {"metrics": []}

def match_metric(name: str, rule: Dict[str, Any], default_ci: bool) -> bool:
    pats: List[str] = rule.get("patterns", [])
    if not pats: return False
    use_regex = bool(rule.get("regex", False))
    ci = bool(rule.get("case_insensitive", default_ci))
    flags = re.IGNORECASE if ci else 0
    if use_regex:
        return any(re.search(p, name, flags=flags) for p in pats)
    return any((p.lower() == name.lower()) if ci else (p == name) for p in pats)

def section_allowed(section: Optional[str], rule: Dict[str, Any]) -> bool:
    allowed = rule.get("sections", ["*"])
    if not allowed or "*" in allowed:
        return True
    sec = norm_section(section) if section else section
    canon = set(norm_section(s) for s in allowed)
    return sec in canon

def apply_mapping(df: pd.DataFrame, labels: List[str], mapping: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    diags: List[Dict[str, Any]] = []
    if df.empty: return df, diags
    ci_default = bool(mapping.get("case_insensitive", False))
    # ensure numeric
    for lab in labels:
        if lab in df.columns:
            df[lab] = pd.to_numeric(df[lab], errors="coerce")
    outputs: List[pd.DataFrame] = []
    for idx, rule in enumerate(mapping.get("metrics", []), start=1):
        label = rule.get("label", f"Rule {idx}")
        by = (rule.get("section_aggregate") or "sum").lower()
        mask = df.apply(lambda r: match_metric(r["metric"], rule, ci_default) and section_allowed(r["section"], rule), axis=1)
        matched = df[mask].copy()
        diags.append({"label": label, "matched_rows": int(matched.shape[0])})
        if matched.empty:
            continue
        group_keys: List[str] = [] if by == "sum" else ["section"]
        agg = matched.groupby(group_keys)[labels].sum(min_count=1).reset_index()
        agg.insert(0, "label", label)
        period_cols = [c for c in labels if c.lower() != "total"]
        agg["Recalc_Total"] = agg[period_cols].sum(axis=1, min_count=1)
        if "Total" in labels:
            agg["PDF_Total"] = agg["Total"]
            agg["Diff"] = agg["Recalc_Total"] - agg["PDF_Total"]
        outputs.append(agg)
    if not outputs:
        cols = ["label"] + (["section"] if any(r.get("section_aggregate") == "by_section" for r in mapping.get("metrics", [])) else [])
        return pd.DataFrame(columns=cols + labels + ["Recalc_Total"]), diags
    out = pd.concat(outputs, ignore_index=True)
    id_cols = ["label"] + (["section"] if "section" in out.columns else [])
    value_cols = [c for c in out.columns if c not in id_cols]
    out = out.groupby(id_cols, dropna=False)[value_cols].sum(min_count=1).reset_index()
    ordered = id_cols + [c for c in labels if c in out.columns] + [c for c in ["Recalc_Total", "PDF_Total", "Diff"] if c in out.columns]
    return out[ordered], diags

def to_long(df: pd.DataFrame, id_cols: List[str], value_cols: List[str], include_recalc=True) -> pd.DataFrame:
    if df.empty: return df
    long = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="period", value_name="count")
    if include_recalc and "Recalc_Total" in df.columns:
        add = df[id_cols + ["Recalc_Total"]].rename(columns={"Recalc_Total": "count"})
        add["period"] = "Recalc_Total"
        long = pd.concat([long, add], ignore_index=True)
    return long.dropna(subset=["count"]).reset_index(drop=True)

# ----- UI -----
st.title("📄→📊 PDF to CSV — Metric Mapping")

with st.sidebar:
    st.header("Extraction / Cleaning")
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = ExtractConfig()
    cfg: ExtractConfig = st.session_state["cfg"]

    cfg.use_ocr = st.toggle("Use OCR (pytesseract)", value=cfg.use_ocr)
    cfg.normalize_unicode = st.checkbox("Normalize Unicode", value=cfg.normalize_unicode)
    cfg.collapse_whitespace = st.checkbox("Collapse extra spaces", value=cfg.collapse_whitespace)
    cfg.remove_empty_lines = st.checkbox("Remove empty lines", value=cfg.remove_empty_lines)
    cfg.hyphenation_fix = st.checkbox("Fix hyphenation joins", value=cfg.hyphenation_fix)
    cfg.drop_header_lines = st.number_input("Drop header lines per page", 0, 50, value=cfg.drop_header_lines)
    cfg.drop_footer_lines = st.number_input("Drop footer lines per page", 0, 50, value=cfg.drop_footer_lines)
    cfg.remove_page_numbers = st.checkbox("Remove page number lines", value=cfg.remove_page_numbers)

    st.subheader("OCR health check")
    try:
        t_path = shutil.which("tesseract")
        st.caption(f"Tesseract: **{bool(t_path)}** | path: `{t_path}`")
        if _HAS_TESSERACT:
            st.caption(f"Version: **{pytesseract.get_tesseract_version()}**")
    except Exception as e:
        st.warning(f"OCR check issue: {e}")

st.markdown("**Step 1 — Upload PDF**")
pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
if not pdf_file:
    st.info("Upload a PDF to begin.")
    st.stop()

st.markdown("**Step 2 — Mapping JSON (you can edit rules here)**")
mapping_text = st.text_area("Mapping JSON", value=DEFAULT_MAPPING_JSON, height=260)

# Extract + clean
pdf_bytes = io.BytesIO(pdf_file.read())
with st.spinner("Extracting text..."):
    raw_text, _ = extract_pdf_text(pdf_bytes, use_ocr=cfg.use_ocr)

txt = strip_headers_footers(raw_text, cfg)
if cfg.remove_page_numbers:
    txt = remove_page_numbers(txt)
txt = normalize_text(txt, cfg)

with st.expander("Cleaned text (preview)", expanded=False):
    st.text_area("Cleaned", value=txt[:20000], height=240)
    st.caption("Preview capped at ~20k chars.")

# Parse matrix
with st.spinner("Parsing matrix..."):
    df_wide, labels = parse_matrix_blocks(txt, ncols=14)

if df_wide.empty:
    st.error("No matrix rows matched. Ensure header contains 'Reason for Contact' and metric lines end with 14 numbers.")
    st.stop()

st.success(f"Detected period columns: {', '.join(labels)}")

# Apply mapping
mapping_cfg = load_mapping(mapping_text)
with st.spinner("Applying mapping & aggregating..."):
    result_df, diags = apply_mapping(df_wide, labels, mapping_cfg)

with st.expander("Diagnostics", expanded=False):
    st.write(pd.DataFrame(diags))

if result_df.empty:
    st.warning("No rows after mapping. Verify rule patterns and segments.")
    st.stop()

st.markdown("**Step 3 — Review & Export**")
long_out = st.toggle("Output long (tidy) format", value=False)
if long_out:
    ids = ["label"] + (["section"] if "section" in result_df.columns else [])
    vals = [c for c in labels if c in result_df.columns]
    long_df = to_long(result_df, id_cols=ids, value_cols=vals, include_recalc=True)
    st.dataframe(long_df.head(1000), use_container_width=True)
    st.download_button("⬇️ Download CSV (long)", data=long_df.to_csv(index=False).encode("utf-8"),
                       file_name="mapped_metrics_long.csv", mime="text/csv", use_container_width=True)
else:
    st.dataframe(result_df.head(1000), use_container_width=True)
    st.download_button("⬇️ Download CSV (wide)", data=result_df.to_csv(index=False).encode("utf-8"),
                       file_name="mapped_metrics_wide.csv", mime="text/csv", use_container_width=True)
