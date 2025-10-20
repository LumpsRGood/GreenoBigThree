# path: app.py
# Streamlit — Metric totals (latest period) + per-page debug (carry-forward sections)
# Updated per request:
#  - "Order wrong" now ONLY matches "Order Wrong" (excludes "Not Made To Order").
#  - Add all-segment metrics:
#      * Unprofessional/Unfriendly → "Unfriendly Attitude", "Unprofessional Behavior"
#      * Manager directly involved → "Manager Directly Involved", "Manager Directly Involved In Complaint"
#      * Manager not available → "Management Not Available"

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

st.set_page_config(page_title="PDF → CSV — Metric totals + per-page debug", page_icon="📄", layout="wide")

# Optional OCR
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# PDF text extraction
try:
    import pdfplumber
except Exception:
    st.error("Missing dependency: pdfplumber. Add `pdfplumber` to requirements.txt.")
    st.stop()

# -------- Mapping rules (JSON; no comments) --------
DEFAULT_MAPPING_JSON = r"""
{
  "case_insensitive": true,
  "metrics": [
    {
      "label": "Missing food",
      "patterns": ["Item \\(Food\\)"],
      "regex": true,
      "sections": ["To-Go", "Dine-In", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Order wrong",
      "patterns": ["Order Wrong"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Missing condiments",
      "patterns": ["Missing Condiments"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Out of menu item",
      "patterns": ["Out Of Menu Item"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Missing bev",
      "patterns": ["Item \\(Bev\\)"],
      "regex": true,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Missing ingredients",
      "patterns": ["Ingredient \\(Food\\)"],
      "regex": true,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },
    {
      "label": "Packaging to-go complaint",
      "patterns": ["Packaging To Go Complaint"],
      "regex": false,
      "sections": ["To-Go", "Delivery"],
      "section_aggregate": "sum"
    },

    // ---- All segments below (use in-app label names you specified) ----
    {
      "label": "Unprofessional/Unfriendly",
      "patterns": ["Unfriendly Attitude", "Unprofessional Behavior"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager directly involved",
      "patterns": ["Manager Directly Involved", "Manager Directly Involved In Complaint"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager not available",
      "patterns": ["Management Not Available"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    }
  ]
}
"""

# ---------- extraction / cleaning ----------
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
                    txt = pytesseract.image_to_string(img)  # only for image-only pages
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
        s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # why: join hyphen-breaks
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
        if re.fullmatch(r"Page\s+\d+(?:\s*/\s*\d+)?", t, flags=re.I): continue
        if re.fullmatch(r"\d{1,4}", t): continue
        keep.append(ln)
    return "\n".join(keep)

# ---------- parser ----------
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
        if not ln: continue
        if ln.lower() in SECTION_ALIASES:
            section = norm_section(ln)
            continue
        m = pat.match(ln)
        if not m: continue
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

# ---------- mapping / aggregation ----------
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
    for lab in labels:
        if lab in df.columns:
            df[lab] = pd.to_numeric(df[lab], errors="coerce")
    ci_default = bool(mapping.get("case_insensitive", False))
    outputs: List[pd.DataFrame] = []
    for idx, rule in enumerate(mapping.get("metrics", []), start=1):
        label = rule.get("label", f"Rule {idx}")
        agg_mode = (rule.get("section_aggregate") or "sum").lower()
        mask = df.apply(lambda r: match_metric(r["metric"], rule, ci_default) and section_allowed(r["section"], rule), axis=1)
        matched = df[mask].copy()
        diags.append({"label": label, "matched_rows": int(matched.shape[0])})
        if matched.empty:
            continue
        use_cols = [c for c in labels if c in matched.columns]
        if agg_mode == "by_section":
            agg = matched.groupby(["section"], dropna=False)[use_cols].sum(min_count=1).reset_index()
            agg.insert(0, "label", label)
        else:
            totals = matched[use_cols].sum(min_count=1)
            agg = pd.DataFrame([totals]); agg.insert(0, "label", label)
        period_cols = [c for c in use_cols if c.lower() != "total"]
        agg["Recalc_Total"] = agg[period_cols].sum(axis=1, min_count=1)
        if "Total" in use_cols:
            agg["PDF_Total"] = agg["Total"]; agg["Diff"] = agg["Recalc_Total"] - agg["PDF_Total"]
        outputs.append(agg)
    if not outputs:
        return pd.DataFrame(columns=["label"] + labels + ["Recalc_Total"]), diags
    out = pd.concat(outputs, ignore_index=True)
    id_cols = ["label"] + (["section"] if "section" in out.columns else [])
    value_cols = [c for c in out.columns if c not in id_cols]
    out = out.groupby(id_cols, dropna=False)[value_cols].sum(min_count=1).reset_index()
    ordered = ["label"] + [c for c in labels if c in out.columns] + [c for c in ["Recalc_Total","PDF_Total","Diff"] if c in out.columns]
    return out[ordered], diags

def pick_latest_period_label(labels: List[str]) -> Optional[str]:
    cand = [lab for lab in labels if lab.lower() != "total"]
    if not cand: return None
    def key(lbl: str) -> int:
        m = re.match(r"^P(\d{1,2})_(\d{2})$", lbl)
        if not m: return -1
        p = int(m.group(1)); y = int(m.group(2))
        return y * 100 + p
    scored = [(lbl, key(lbl)) for lbl in cand]
    if all(k >= 0 for _, k in scored):
        return max(scored, key=lambda x: x[1])[0]
    return cand[-1]

# ---------- per-page debug (carry-forward sections) ----------
def per_page_totals_for_metric(
    pdf_bytes: io.BytesIO,
    labels: List[str],
    target_label: str,
    pattern_regex: str,
    allowed_sections: List[str],
    carry_forward_sections: bool = True,
) -> pd.DataFrame:
    """Carry-forward avoids misses when section headers don't repeat across pages."""
    allowed = set(norm_section(s) for s in allowed_sections) if allowed_sections else None
    pages_totals: Dict[int, int] = {}
    pat_line = metric_line_pattern(ncols=14)

    with pdfplumber.open(pdf_bytes) as pdf:
        section = None  # carry across pages
        for page_idx, page in enumerate(pdf.pages, start=1):
            if not carry_forward_sections:
                section = None
            txt = page.extract_text(layout=True) or ""
            lines = [unicodedata.normalize("NFKC", ln) for ln in txt.splitlines()]
            for raw in lines:
                ln = re.sub(r"[ \t]+", " ", raw.strip())
                if not ln:
                    continue
                if ln.lower() in SECTION_ALIASES or ln in {"Delivery","Dine-In","Dine In","To-Go","To Go","Carryout","Carry Out","Takeout"}:
                    section = norm_section(ln)
                    continue
                m = pat_line.match(ln)
                if not m:
                    continue
                if allowed and section not in allowed:
                    continue
                metric_name = m.group("metric").rstrip(":").strip()
                if not re.search(pattern_regex, metric_name, flags=re.I):
                    continue
                vals = [m.group(f"c{i:02d}") for i in range(1, 14 + 1)]
                if target_label in labels:
                    idx = labels.index(target_label)
                    v = vals[idx] if idx < len(vals) else None
                    cnt = int(v) if v and v.isdigit() else 0
                    if cnt:
                        pages_totals[page_idx] = pages_totals.get(page_idx, 0) + cnt

    if not pages_totals:
        return pd.DataFrame(columns=["page","total"])
    return pd.DataFrame(sorted(pages_totals.items()), columns=["page","total"])

# ---------- UI ----------
st.title("📄→📊 Metric totals (latest period) + per-page debug")

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
    st.info("Upload a PDF to begin."); st.stop()

st.markdown("**Step 2 — Mapping JSON**")
mapping_text = st.text_area("Mapping JSON", value=DEFAULT_MAPPING_JSON, height=520)

# Extract + clean
pdf_bytes = io.BytesIO(pdf_file.read())
with st.spinner("Extracting text..."):
    raw_text, _pages = extract_pdf_text(pdf_bytes, use_ocr=cfg.use_ocr)

txt = strip_headers_footers(raw_text, cfg)
if cfg.remove_page_numbers: txt = remove_page_numbers(txt)
txt = normalize_text(txt, cfg)

# Parse → labels
with st.spinner("Parsing matrix..."):
    df_wide, labels = parse_matrix_blocks(txt, ncols=14)
if df_wide.empty:
    st.error("No matrix rows matched. Ensure header includes 'Reason for Contact' and rows end with 14 numbers.")
    st.stop()

latest_label = pick_latest_period_label(labels)
if not latest_label:
    st.error("Could not detect period labels."); st.stop()
st.success(f"Detected period columns: {', '.join(labels)} • Latest: **{latest_label}**")

# Apply mapping
mapping_cfg = load_mapping(mapping_text)
with st.spinner("Applying mapping & aggregating..."):
    result_df, diags = apply_mapping(df_wide, labels, mapping_cfg)
if result_df.empty:
    st.warning("No rows after mapping. Verify your patterns and sections.")
    st.stop()

# Render per metric (cards + per-page debug)
rules: List[Dict[str, Any]] = mapping_cfg.get("metrics", [])
label_to_rule = {r["label"]: r for r in rules}
available_labels = [r["label"] for r in rules]

for label in available_labels:
    st.markdown("---")
    st.subheader(f"{label} — total ({latest_label})")

    if label not in result_df["label"].values:
        st.info("No matches for this metric in the PDF."); continue

    total_value = int(result_df.loc[result_df["label"] == label, latest_label].sum())
    st.metric(label=f"{label} — {latest_label}", value=total_value)

    out_df = pd.DataFrame({"label": [label], latest_label: [total_value]})
    st.download_button(
        f"⬇️ Download CSV — {label} total ({latest_label})",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{label.lower().replace(' ','_')}_total_{latest_label}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    rule = label_to_rule[label]
    patterns = rule.get("patterns", [])
    regex = bool(rule.get("regex", False))
    if not patterns: continue
    pat_regex = "|".join(f"(?:{p})" for p in patterns) if regex else "|".join(re.escape(p) for p in patterns)

    allowed_sections = rule.get("sections", ["*"])
    if "*" in allowed_sections:
        allowed_sections = ["Delivery","Dine-In","To-Go","Carryout","Takeout","Carry Out","Dine In","To Go"]

    with st.expander(f"🔎 Debug — per-page totals for {label} ({latest_label})"):
        with st.spinner("Computing per-page totals..."):
            page_totals = per_page_totals_for_metric(
                pdf_bytes=io.BytesIO(pdf_bytes.getvalue()),
                labels=labels,
                target_label=latest_label,
                pattern_regex=pat_regex,
                allowed_sections=allowed_sections,
                carry_forward_sections=True,  # keep this ON
            )
        if page_totals.empty:
            st.info("No pages with non-zero totals for the latest period.")
        else:
            st.dataframe(page_totals, use_container_width=True)
            st.download_button(
                f"⬇️ Download CSV — per-page {label} ({latest_label})",
                data=page_totals.to_csv(index=False).encode("utf-8"),
                file_name=f"{label.lower().replace(' ','_')}_per_page_{latest_label}.csv",
                mime="text/csv",
                use_container_width=True,
            )

with st.expander("Diagnostics"):
    st.write(pd.DataFrame(diags))
