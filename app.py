# path: app.py
# Greeno Bad Three — PDF → Scoreboard (3×7 squares, centered large totals, muted corporate palette)

from __future__ import annotations

import io, re, os, json, unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# --- Page config must be first ---
st.set_page_config(page_title="Greeno Bad Three", page_icon="🐘", layout="wide")

# PDF extractor
try:
    import pdfplumber
except Exception:
    st.error("Missing dependency: pdfplumber. Add `pdfplumber` to requirements.txt.")
    st.stop()

# ---------------------- Locked Mapping ----------------------
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

# ---------------------- Styles (square tiles; centered large totals; muted palette) ----------------------
SCOREBOARD_CSS = """
<style>
/* Fixed 3×7 grid; squares via aspect-ratio. */
.tiles-row { display:grid; grid-template-columns:repeat(7,1fr); gap:10px; margin-bottom:10px; }
.tile { position:relative; width:100%; aspect-ratio:1/1; border-radius:12px; overflow:hidden;
        box-shadow:0 6px 18px rgba(0,0,0,.10), 0 2px 6px rgba(0,0,0,.06); }
.tile-inner { position:absolute; inset:0; display:grid; grid-template-rows:auto 1fr; padding:10px 12px; }
.title { font-size:.78rem; font-weight:800; margin:0; opacity:.98; text-shadow:0 1px 2px rgba(0,0,0,.18); }
.value-wrap { display:flex; align-items:center; justify-content:center; }
.value { font-weight:900; line-height:1; margin:0; letter-spacing:-0.5px;
         font-size:clamp(2.4rem, 5.8vw, 3.6rem); text-shadow:0 2px 4px rgba(0,0,0,.22); }
.loading { display:flex; align-items:center; justify-content:center; gap:8px; opacity:.92; }
.spinner { width:18px; height:18px; border:3px solid rgba(255,255,255,.45); border-top-color:#fff;
           border-radius:50%; animation:spin .9s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
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

def extract_pdf_text(file: io.BytesIO) -> tuple[str, List[str]]:
    pages: List[str] = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text(layout=True) or "")
    return "\n<<<PAGE_BREAK>>>\n".join(pages), pages

def normalize_text(s: str, cfg: ExtractConfig) -> str:
    if cfg.normalize_unicode: s = unicodedata.normalize("NFKC", s)
    if cfg.hyphenation_fix: s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)  # join hyphen-split words (why: avoid wrap bugs)
    s = s.replace("\r\n","\n").replace("\r","\n")
    if cfg.collapse_whitespace: s = "\n".join(re.sub(r"[ \t]+"," ", ln) for ln in s.split("\n"))
    if cfg.remove_empty_lines: s = "\n".join(ln for ln in s.split("\n") if ln.strip())
    return s

def strip_headers_footers(text: str, cfg: ExtractConfig) -> str:
    chunks = text.split("\n<<<PAGE_BREAK>>>\n"); out=[]
    for ch in chunks:
        lines = ch.split("\n")
        if cfg.drop_header_lines>0: lines = lines[cfg.drop_header_lines:]
        if cfg.drop_footer_lines>0 and cfg.drop_footer_lines<len(lines): lines = lines[:-cfg.drop_footer_lines]
        out.append("\n".join(lines))
    return "\n<<<PAGE_BREAK>>>\n".join(out)

def remove_page_numbers(text: str) -> str:
    keep=[]
    for ln in text.split("\n"):
        t=ln.strip()
        if re.fullmatch(r"Page\s+\d+(?:\s*/\s*\d+)?", t, flags=re.I): continue
        if re.fullmatch(r"\d{1,4}", t): continue
        keep.append(ln)
    return "\n".join(keep)

# ---------------------- Parser ----------------------
SECTION_ALIASES = {
    "delivery":"Delivery","dine in":"Dine-In","dine-in":"Dine-In",
    "carryout":"Carryout","carry out":"Carryout","takeout":"Takeout",
    "to go":"To-Go","to-go":"To-Go"
}
SECTION_SYMBOLS = {"Delivery","Dine-In","Dine In","To-Go","To Go","Carryout","Carry Out","Takeout"}

def norm_section(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    return SECTION_ALIASES.get(s.strip().lower(), s.strip())

def extract_period_labels(text: str, expected: int = 14) -> List[str]:
    for ln in text.splitlines():
        if ("Reason for Contact" in ln) and ("Total" in ln):
            parts = re.findall(r"P\d{1,2}\s+\d{2}", ln)
            if len(parts) >= expected-1:
                labs = [p.replace(" ","_") for p in parts[:expected-1]]
                labs.append("Total"); return labs
    return [f"col{i:02d}" for i in range(1, expected+1)]

def metric_line_pattern(ncols: int = 14) -> re.Pattern:
    nums = r"\s+".join([fr"(?P<c{i:02d}>\d+)" for i in range(1, ncols+1)])
    return re.compile(fr"^(?P<metric>[A-Za-z][A-Za-z0-9/'&()\-\s]+?)\s*:?\s+{nums}\s*$")

def parse_matrix_blocks(text: str, ncols: int = 14) -> tuple[pd.DataFrame, List[str]]:
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
        vals = [gd[f"c{i:02d}"] for i in range(1, ncols+1)]
        row = {"section": norm_section(section), "metric": metric_name}
        for j, lab in enumerate(labels):
            v = vals[j]
            row[lab] = int(v) if v and v.isdigit() else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty: df = df[["section","metric"] + labels]
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
    return any((p.lower()==name.lower()) if ci else (p==name) for p in pats)

def section_allowed(section: Optional[str], rule: Dict[str, Any]) -> bool:
    allowed = rule.get("sections", ["*"])
    if not allowed or "*" in allowed: return True
    sec = norm_section(section) if section else section
    canon = set(norm_section(s) for s in allowed)
    return sec in canon

def apply_mapping(df: pd.DataFrame, labels: List[str], mapping: Dict[str, Any]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    for lab in labels:
        if lab in df.columns: df[lab] = pd.to_numeric(df[lab], errors="coerce")
    ci_default = bool(mapping.get("case_insensitive", False))
    outputs: List[pd.DataFrame] = []
    for idx, rule in enumerate(mapping.get("metrics", []), start=1):
        label = rule.get("label", f"Rule {idx}")
        mask = df.apply(lambda r: match_metric(r["metric"], rule, ci_default) and section_allowed(r["section"], rule), axis=1)
        matched = df[mask].copy()
        use_cols = [c for c in labels if c in matched.columns]
        if matched.empty:
            totals = pd.Series({c: 0 for c in use_cols})
            agg = pd.DataFrame([totals]); agg.insert(0, "label", label)
        else:
            totals = matched[use_cols].sum(min_count=1)
            agg = pd.DataFrame([totals]); agg.insert(0, "label", label)
        outputs.append(agg)
    out = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(columns=["label"] + labels)
    out = out.groupby(["label"], dropna=False).sum(min_count=1).reset_index()
    ordered = ["label"] + [c for c in labels if c in out.columns]
    return out[ordered]

def pick_latest_period_label(labels: List[str]) -> Optional[str]:
    cand = [lab for lab in labels if lab.lower() != "total"]
    if not cand: return None
    def key(lbl: str) -> int:
        m = re.match(r"^P(\d{1,2})_(\d{2})$", lbl)
        if not m: return -1
        return int(m.group(2))*100 + int(m.group(1))
    return max(cand, key=key)

# ---------------------- UI ----------------------
st.markdown("# 📊 Greeno Bad Three — Scoreboard")
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
period_label = st.selectbox("Period", options=period_choices, index=period_choices.index(latest_label) if latest_label in period_choices else 0)

# Muted corporate palette (7 colors; repeats per row)
PALETTE: List[Tuple[str, str]] = [
    ("linear-gradient(135deg,#1e3a8a,#3b82f6)", "#ffffff"),  # navy → blue
    ("linear-gradient(135deg,#0f766e,#22d3ee)", "#ffffff"),  # teal
    ("linear-gradient(135deg,#4338ca,#818cf8)", "#ffffff"),  # indigo
    ("linear-gradient(135deg,#334155,#64748b)", "#ffffff"),  # slate
    ("linear-gradient(135deg,#065f46,#34d399)", "#ffffff"),  # emerald
    ("linear-gradient(135deg,#92400e,#f59e0b)", "#1a1200"),  # amber (dark text)
    ("linear-gradient(135deg,#9f1239,#f472b6)", "#ffffff")   # rose
]

st.markdown(SCOREBOARD_CSS, unsafe_allow_html=True)
mapping_cfg = load_mapping_constant()
metric_labels = [m["label"] for m in mapping_cfg["metrics"]]

# Build fixed 3×7 grid
rows = [st.columns(7) for _ in range(3)]
placeholders: Dict[str, Tuple[st.delta_generator.DeltaGenerator, str, str]] = {}

# Skeleton tiles (spinner centered)
for idx, lab in enumerate(metric_labels[:21]):
    r, c = divmod(idx, 7)
    bg, fg = PALETTE[c]
    with rows[r][c]:
        ph = st.empty()
        placeholders[lab] = (ph, bg, fg)
        ph.markdown(
            f"""
            <div class="tile" style="background:{bg}; color:{fg}">
              <div class="tile-inner">
                <p class="title">{lab}</p>
                <div class="loading"><div class="spinner"></div><div>loading…</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Compute totals once
with st.spinner("Computing totals…"):
    result_df = apply_mapping(df_wide, labels, mapping_cfg)

# Fill values (centered/large)
for lab in metric_labels[:21]:
    ph, bg, fg = placeholders[lab]
    val = 0
    if (not result_df.empty) and (lab in result_df["label"].values) and (period_label in result_df.columns):
        val = int(result_df.loc[result_df["label"] == lab, period_label].sum())
    ph.markdown(
        f"""
        <div class="tile" style="background:{bg}; color:{fg}">
          <div class="tile-inner">
            <p class="title">{lab}</p>
            <div class="value-wrap"><p class="value">{val}</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
