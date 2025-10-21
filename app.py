# path: app.py
# Streamlit — "Greeno Bad Three" • Pretty square scoreboard, no sidebar/debug

from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# --- Page chrome ---
ALABAMA_A_ICON_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Alabama_A_logo.png/240px-Alabama_A_logo.png"
st.set_page_config(page_title="Greeno Bad Three", page_icon=ALABAMA_A_ICON_URL, layout="wide")

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
    { "label": "Missing food", "patterns": ["Item \\(Food\\)"], "regex": true, "sections": ["To-Go","Dine-In","Delivery"], "section_aggregate": "sum" },
    { "label": "Order wrong", "patterns": ["Order Wrong"], "regex": false, "sections": ["To-Go","Delivery"], "section_aggregate": "sum" },
    { "label": "Missing condiments", "patterns": ["Missing Condiments"], "regex": false, "sections": ["To-Go","Delivery"], "section_aggregate": "sum" },
    { "label": "Out of menu item", "patterns": ["Out Of Menu Item"], "regex": false, "sections": ["To-Go","Delivery"], "section_aggregate": "sum" },
    { "label": "Missing bev", "patterns": ["Item \\(Bev\\)"], "regex": true, "sections": ["To-Go","Delivery"], "section_aggregate": "sum" },
    { "label": "Missing ingredients", "patterns": ["Ingredient \\(Food\\)"], "regex": true, "sections": ["To-Go","Delivery"], "section_aggregate": "sum" },
    { "label": "Packaging to-go complaint", "patterns": ["Packaging To Go Complaint"], "regex": false, "sections": ["To-Go","Delivery"], "section_aggregate": "sum" },

    { "label": "Unprofessional/Unfriendly", "patterns": ["Unfriendly Attitude","Unprofessional Behavior"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager directly involved", "patterns": ["^Manager\\s*Directly(?:\\s|-)?"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager not available", "patterns": ["Management Not Available"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager did not visit", "patterns": ["Not Visit","Manager Did Not Visit"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Negative mgr-employee exchange", "patterns": ["Negative Manager","Negative Manager-Employee Interaction"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Manager did not follow up", "patterns": ["Manager Did Not Follow","Manager Did Not Follow Up"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Argued with guest", "patterns": ["Argued","Argued With Guest"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Long hold/no answer", "patterns": ["Long Hold","No Answer","Hung Up","Long Hold/No Answer/Hung Up"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "No/insufficient compensation offered", "patterns": ["No/Unsatisfactory","No/Unsatisfactory Compensation Offered By Restaurant"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Did not attempt to resolve", "patterns": ["Resolve","Did Not Attempt To Resolve"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Guest left without ordering", "patterns": ["Guest Left","Guest Left Without Dining or Ordering"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Unknowledgeable", "patterns": ["Unknowledgeable"], "regex": false, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "Did not open on time", "patterns": ["Open/close","Didn’t Open/close On Time","Didn't Open/close On Time"], "regex": true, "sections": ["*"], "section_aggregate": "sum" },
    { "label": "No/poor apology", "patterns": ["No/Poor Apology"], "regex": false, "sections": ["*"], "section_aggregate": "sum" }
  ]
}
"""

# ---------------------- Styles: square colorful scorebugs ----------------------
SCOREBOARD_CSS = """
<style>
:root {
  --card-radius: 18px;
}
.score-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
}
.score-card {
  position: relative;
  border-radius: var(--card-radius);
  aspect-ratio: 1 / 1;
  color: white;
  padding: 14px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: 0 4px 12px rgba(0,0,0,.12);
}
.score-title {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: .2px;
  line-height: 1.15;
  margin: 0;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-shadow: 0 1px 1px rgba(0,0,0,.25);
}
.score-value {
  font-size: 2.1rem;
  font-weight: 800;
  line-height: 1;
  text-align: right;
  text-shadow: 0 2px 4px rgba(0,0,0,.35);
}
.spinner {
  width: 22px; height: 22px;
  border: 3px solid rgba(255,255,255,.35);
  border-top-color: rgba(255,255,255,.95);
  border-radius: 50%;
  animation: spin .9s linear infinite;
  margin-right: 8px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading {
  display: flex; align-items: center; justify-content: flex-end; gap: 8px;
  color: rgba(255,255,255,.9);
  font-weight: 700;
}
.bg-1 { background: linear-gradient(135deg, #6EE7B7, #3B82F6); }
.bg-2 { background: linear-gradient(135deg, #F59E0B, #EF4444); }
.bg-3 { background: linear-gradient(135deg, #A78BFA, #EC4899); }
.bg-4 { background: linear-gradient(135deg, #34D399, #10B981); }
.bg-5 { background: linear-gradient(135deg, #F472B6, #8B5CF6); }
.bg-6 { background: linear-gradient(135deg, #22D3EE, #3B82F6); }
.bg-7 { background: linear-gradient(135deg, #F43F5E, #F97316); }
.bg-8 { background: linear-gradient(135deg, #60A5FA, #2563EB); }
.bg-9 { background: linear-gradient(135deg, #84CC16, #22C55E); }
.bg-10{ background: linear-gradient(135deg, #FDE047, #F59E0B); }
.card-inner { height: 100%; display:flex; flex-direction:column; justify-content:space-between; }
.header-row { display:flex; align-items:flex-start; justify-content:space-between; gap:8px; }
.logo-badge {
  width: 26px; height: 26px; border-radius: 8px;
  background: rgba(255,255,255,.22); display:flex; align-items:center; justify-content:center;
  font-weight: 800; color: rgba(255,255,255,.95);
}
</style>
"""

# ---------------------- Extraction / cleaning ----------------------
@dataclass
class ExtractConfig:
    normalize_unicode: bool = True
    collapse_whitespace: bool = True
    remove_empty_lines: bool = True
    hyphenation_fix: bool = True
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
        s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
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
    return SECTION_ALIASES.get(s.strip().lower(), s.strip())

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
    return re.compile(fr"^(?P<metric>[A-Za-z][A-Za-z0-9/'&()\-\s]+?)\s*:?\s+{nums}\s*$")

def parse_matrix_blocks(text: str, ncols: int = 14) -> Tuple[pd.DataFrame, List[str]]:
    labels = extract_period_labels(text, expected=ncols)
    pat = metric_line_pattern(ncols)
    rows: List[Dict[str, Any]] = []
    section: Optional[str] = None
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln: continue
        if ln.lower() in SECTION_ALIASES or ln in SECTION_SYMBOLS:
            section = norm_section(ln); continue
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
    if not allowed or "*" in allowed: return True
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
st.markdown("# Greeno's Bad Three")
st.caption("Upload the PDF, choose a period, and see totals at a glance.")

pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
if not pdf_file:
    st.stop()

cfg = ExtractConfig()

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
period_cols = st.columns([1,3,8])
with period_cols[0]:
    st.write("**Period**")
with period_cols[1]:
    period_label = st.selectbox("", options=period_choices, index=period_choices.index(latest_label) if latest_label in period_choices else 0, label_visibility="collapsed")

# CSS + placeholders
st.markdown(SCOREBOARD_CSS, unsafe_allow_html=True)
mapping_cfg = load_mapping_constant()
rules: List[Dict[str, Any]] = mapping_cfg.get("metrics", [])
labels_list = [r["label"] for r in rules]

palette_classes = ["bg-1","bg-2","bg-3","bg-4","bg-5","bg-6","bg-7","bg-8","bg-9","bg-10"]

grid = st.container()
with grid:
    st.markdown('<div class="score-grid">', unsafe_allow_html=True)
    placeholders: Dict[str, st.delta_generator.DeltaGenerator] = {}
    for i, lab in enumerate(labels_list):
        cls = palette_classes[i % len(palette_classes)]
        ph = st.empty()
        placeholders[lab] = ph
        ph.markdown(f'''
            <div class="score-card {cls}">
              <div class="card-inner">
                <div class="header-row">
                  <div class="logo-badge">A</div>
                  <h4 class="score-title">{lab}</h4>
                </div>
                <div class="loading"><span class="spinner"></span>loading…</div>
              </div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Compute totals once
with st.spinner("Computing totals…"):
    result_df = apply_mapping(df_wide, labels, mapping_cfg)

# Fill cards
for i, lab in enumerate(labels_list):
    cls = palette_classes[i % len(palette_classes)]
    val = 0
    if (not result_df.empty) and (lab in result_df["label"].values) and (period_label in result_df.columns):
        val = int(result_df.loc[result_df["label"] == lab, period_label].sum())
    placeholders[lab].markdown(f'''
        <div class="score-card {cls}">
          <div class="card-inner">
            <div class="header-row">
              <div class="logo-badge">A</div>
              <h4 class="score-title">{lab}</h4>
            </div>
            <div class="score-value">{val}</div>
          </div>
        </div>
    ''', unsafe_allow_html=True)
