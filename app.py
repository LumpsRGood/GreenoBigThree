# path: app.py
# Streamlit — PDF → Scoreboard (selected period totals)
# Clean UI: no sidebar, no debug/diagnostics. Top scoreboard with per-metric spinners.

from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PDF → Scoreboard", page_icon="📊", layout="wide")

# PDF extractor
try:
    import pdfplumber
except Exception:
    st.error("Missing dependency: pdfplumber. Add `pdfplumber` to requirements.txt.")
    st.stop()

# ---------------------- Locked Mapping (JSON, no comments) ----------------------
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

    {
      "label": "Unprofessional/Unfriendly",
      "patterns": ["Unfriendly Attitude", "Unprofessional Behavior"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager directly involved",
      "patterns": ["^Manager\\s*Directly(?:\\s|-)?"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager not available",
      "patterns": ["Management Not Available"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager did not visit",
      "patterns": ["Not Visit", "Manager Did Not Visit"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Negative mgr-employee exchange",
      "patterns": ["Negative Manager", "Negative Manager-Employee Interaction"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Manager did not follow up",
      "patterns": ["Manager Did Not Follow", "Manager Did Not Follow Up"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Argued with guest",
      "patterns": ["Argued", "Argued With Guest"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Long hold/no answer",
      "patterns": ["Long Hold", "No Answer", "Hung Up", "Long Hold/No Answer/Hung Up"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "No/insufficient compensation offered",
      "patterns": ["No/Unsatisfactory", "No/Unsatisfactory Compensation Offered By Restaurant"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Did not attempt to resolve",
      "patterns": ["Resolve", "Did Not Attempt To Resolve"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Guest left without ordering",
      "patterns": ["Guest Left", "Guest Left Without Dining or Ordering"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Unknowledgeable",
      "patterns": ["Unknowledgeable"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "Did not open on time",
      "patterns": ["Open/close", "Didn’t Open/close On Time", "Didn't Open/close On Time"],
      "regex": true,
      "sections": ["*"],
      "section_aggregate": "sum"
    },
    {
      "label": "No/poor apology",
      "patterns": ["No/Poor Apology"],
      "regex": false,
      "sections": ["*"],
      "section_aggregate": "sum"
    }
  ]
}
"""

# ---------------------- Styles (score bug + spinner) ----------------------
SCOREBOARD_CSS = """
<style>
.score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.score-card { border: 1px solid rgba(0,0,0,0.08); border-radius: 14px; padding: 14px 16px; background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
.score-title { font-size: 0.93rem; color: #444; margin: 0 0 6px 0; line-height: 1.2; }
.score-value { font-size: 1.8rem; font-weight: 700; margin: 2px 0 0 0; }
.spinner {
  width: 20px; height: 20px; border: 3px solid #eee; border-top-color: #2e7df6; border-radius: 50%;
  animation: spin 0.9s linear infinite; display:inline-block; vertical-align: middle; margin-right:8px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-row { display:flex; align-items:center; color:#666; }
</style>
"""

# ---------------------- Extraction / cleaning ----------------------
@dataclass
class ExtractConfig:
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    remove_empty_lines: bool = True
    hyphenation_fix: bool = True       # ON by default to resist line wraps
    drop_header_lines: int = 0
    drop_footer_lines: int = 0
    remove_page_numbers: bool = True

def extract_pdf_text(file: io.BytesIO) -> Tuple[str, List[str]]:
    pages: List[str] = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(layout=True) or ""
            pages.append(txt or "")
    return "\n<<<PAGE_BREAK>>>\n".join(pages), pages

def normalize_text(s: str, cfg: ExtractConfig) -> str:
    if cfg.normalize_unicode:
        s = unicodedata.normalize("NFKC", s)
    if cfg.hyphenation_fix:
        s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # join hyphen-split words
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
        if re.fullmatch(r"Page\s+\d+(?:\s*/\s*\d+)?", t, flags=re.I):  # FIXED
            continue
        if re.fullmatch(r"\d{1,4}", t):  # FIXED
            continue
        keep.append(ln)
    return "\n".join(keep)

# ---------------------- Parser ----------------------
SECTION_ALIASES = {
    "delivery": "Delivery",
    "dine in": "Dine-In",
    "dine-in": "Dine-In",
    "carryout": "Carryout",
    "carry out": "Carryout",
    "takeout": "Takeout",
    "to go": "To-Go",
    "to-go": "To-Go"
}
SECTION_SYMBOLS = {"Delivery","Dine-In","Dine In","To-Go","To Go","Carryout","Carry Out","Takeout"}

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
    # FIXED: proper raw escapes for \d and \s
    nums = r"\s+".join([fr"(?P<c{i:02d}>\d+)" for i in range(1, ncols + 1)])
    return re.compile(
        fr"^(?P<metric>[A-Za-z][A-Za-z0-9/'&()\-\s]+?)\s*:?\s+{nums}\s*$"
    )

def parse_matrix_blocks(text: str, ncols: int = 14) -> Tuple[pd.DataFrame, List[str]]:
    labels = extract_period_labels(text, expected=ncols)
    pat = metric_line_pattern(ncols)
    rows: List[Dict[str, Any]] = []
    section: Optional[str] = None
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln: continue
        if ln.lower() in SECTION_ALIASES or ln in SECTION_SYMBOLS:
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

# ---------------------- Mapping engine ----------------------
def load_mapping_constant() -> Dict[str, Any]:
    return json.loads(DEFAULT_MAPPING_JSON)

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

def apply_mapping(df: pd.DataFrame, labels: List[str], mapping: Dict[str, Any]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
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
        if matched.empty:
            totals = pd.Series({c: 0 for c in labels})
            agg = pd.DataFrame([totals]); agg.insert(0, "label", label)
            outputs.append(agg); continue
        use_cols = [c for c in labels if c in matched.columns]
        if agg_mode == "by_section":
            agg = matched.groupby(["section"], dropna=False)[use_cols].sum(min_count=1).reset_index()
            agg.insert(0, "label", label)
        else:
            totals = matched[use_cols].sum(min_count=1)
            agg = pd.DataFrame([totals]); agg.insert(0, "label", label)
        outputs.append(agg)
    if not outputs:
        return pd.DataFrame(columns=["label"] + labels)
    out = pd.concat(outputs, ignore_index=True)
    id_cols = ["label"] + (["section"] if "section" in out.columns else [])
    value_cols = [c for c in out.columns if c not in id_cols]
    out = out.groupby(id_cols, dropna=False)[value_cols].sum(min_count=1).reset_index()
    ordered = ["label"] + [c for c in labels if c in out.columns]
    return out[ordered]

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

# ---------------------- UI (no sidebar) ----------------------
st.markdown("# 📊 Scoreboard")
st.caption("Upload the PDF, choose a period, and see totals at a glance.")

pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
if not pdf_file:
    st.stop()

cfg = ExtractConfig()  # locked defaults

with st.spinner("Extracting & parsing…"):
    pdf_bytes = io.BytesIO(pdf_file.read())
    raw_text, _pages = extract_pdf_text(pdf_bytes)
    txt = strip_headers_footers(raw_text, cfg)
    if cfg.remove_page_numbers: txt = remove_page_numbers(txt)
    txt = normalize_text(txt, cfg)
    df_wide, labels = parse_matrix_blocks(txt, ncols=14)

if df_wide.empty:
    st.error("No matrix rows matched. Ensure header includes 'Reason for Contact' and rows end with 14 numbers.")
    st.stop()

latest_label = pick_latest_period_label(labels) or labels[0]
period_choices = [c for c in labels if c.lower() != "total"]
cols = st.columns([1,2,2,2])
with cols[0]:
    st.write("**Period**")
with cols[1]:
    period_label = st.selectbox("", options=period_choices, index=period_choices.index(latest_label) if latest_label in period_choices else 0, label_visibility="collapsed")

# Scoreboard CSS and placeholders
st.markdown(SCOREBOARD_CSS, unsafe_allow_html=True)
mapping_cfg = load_mapping_constant()
metric_rules: List[Dict[str, Any]] = mapping_cfg.get("metrics", [])
metric_labels = [r["label"] for r in metric_rules]

grid = st.container()
with grid:
    st.markdown('<div class="score-grid">', unsafe_allow_html=True)
    placeholders: Dict[str, st.delta_generator.DeltaGenerator] = {}
    for lab in metric_labels:
        ph = st.empty()
        placeholders[lab] = ph
        ph.markdown(f'''
            <div class="score-card">
              <div class="score-title">{lab}</div>
              <div class="loading-row"><span class="spinner"></span> loading…</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Compute totals once
with st.spinner("Computing totals…"):
    result_df = apply_mapping(df_wide, labels, mapping_cfg)

# Fill in cards
for lab in metric_labels:
    val = 0
    if (not result_df.empty) and (lab in result_df["label"].values) and (period_label in result_df.columns):
        val = int(result_df.loc[result_df["label"] == lab, period_label].sum())
    placeholders[lab].markdown(f'''
        <div class="score-card">
          <div class="score-title">{lab}</div>
          <div class="score-value">{val}</div>
        </div>
    ''', unsafe_allow_html=True)
