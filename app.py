# path: app.py
# Greeno Bad Three — Polished scoreboard (3×7 rows by category, centered totals, shimmer "Roll Tide…", non-sticky header)

from __future__ import annotations

import io, os, re, json, unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------------- Page config (first Streamlit call) ----------------
st.set_page_config(page_title="Greeno Bad Three", page_icon="🐘", layout="wide")

# ---------------- Dependencies ----------------
try:
    import pdfplumber
except Exception:
    st.error("Missing dependency: pdfplumber. Add `pdfplumber` to requirements.txt.")
    st.stop()

# ---------------- Mapping (locked 21 metrics) ----------------
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

# ---------------- Appearance (CSS only) ----------------
UI_CSS = """
<style>
:root{
  --tile-radius:12px; --tile-gap:10px;
  --shadow: 0 6px 18px rgba(0,0,0,.10), 0 2px 6px rgba(0,0,0,.06);
  --header-bg:#0f172a; --header-fg:#ffffff; --divider:rgba(255,255,255,.06);
}

/* Header */
.header-wrap{display:flex;align-items:center;gap:16px;background:var(--header-bg);
  color:var(--header-fg);padding:10px 14px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.18);
  margin-bottom:8px;border-bottom:1px solid var(--divider);}
.logo{width:140px;height:auto;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.18);background:#1e293b;}
.title{font-size:1.25rem;font-weight:700;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

/* Row headers */
.row-header{display:flex;align-items:center;gap:10px;height:34px;margin:12px 0 8px 0;}
.swatch{width:14px;height:14px;border-radius:4px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.2);}
.row-title{font-size:0.95rem;font-weight:700;margin:0;color:#0f172a;}
.row-divider{height:1px;background:rgba(0,0,0,.08);margin-top:6px}

/* Scroll container for narrow viewports */
.row-scroll{overflow-x:auto; padding-bottom:2px;}
.row-inner{display:grid;grid-template-columns:repeat(7,1fr);gap:var(--tile-gap);min-width:980px;}

/* Tile */
.tile{position:relative;width:100%;aspect-ratio:1/1;border-radius:var(--tile-radius);overflow:hidden;box-shadow:var(--shadow);}
.tile-inner{position:absolute;inset:0;display:grid;grid-template-rows:auto 1fr;padding:10px 12px;}
.title-small{font-size:.78rem;font-weight:800;margin:0;color:#fff;opacity:.98;text-shadow:0 1px 2px rgba(0,0,0,.18);
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.value-wrap{display:flex;align-items:center;justify-content:center;}
.value{font-weight:900;line-height:1;margin:0;letter-spacing:-0.3px;font-variant-numeric:tabular-nums;
  font-size:clamp(2.8rem, 6vw, 3.8rem);text-shadow:0 2px 4px rgba(0,0,0,.22);}

/* Inner border to reduce banding */
.tile::after{content:"";position:absolute;inset:0;border-radius:var(--tile-radius);box-shadow:inset 0 0 0 1px rgba(255,255,255,.12);pointer-events:none;}

/* Shimmer placeholder with 'Roll Tide…' */
.skel{position:relative;width:78%;max-width:280px;height:1.6em;border-radius:10px;opacity:.95;
  background:linear-gradient(90deg, rgba(255,255,255,.14) 25%, rgba(255,255,255,.32) 37%, rgba(255,255,255,.14) 63%);
  background-size:400% 100%;animation:shimmer 1.3s linear infinite;}
.skel-text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:1.1rem;opacity:.9;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.28);}
@keyframes shimmer {0%{background-position:100% 0}100%{background-position:-100% 0}}
@media (prefers-reduced-motion: reduce){ .skel{animation:none;background:rgba(255,255,255,.18);} }

/* Hover elevation */
.tile:hover{box-shadow:0 10px 22px rgba(0,0,0,.14), 0 4px 10px rgba(0,0,0,.08);}

/* Category gradients */
.bg-missing{background:linear-gradient(135deg,#155e75,#22d3ee);}
.bg-attitude{background:linear-gradient(135deg,#3730a3,#818cf8);}
.bg-other{background:linear-gradient(135deg,#334155,#64748b);}
/* Text colors on gradients (AA) */
.fg-light{color:#ffffff;}
.fg-dark{color:#0e0a00;}
</style>
"""

st.markdown(UI_CSS, unsafe_allow_html=True)

# ---------------- Extraction / cleaning ----------------
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
    if cfg.normalize_unicode: s = unicodedata.normalize("NFKC", s)
    if cfg.hyphenation_fix:   s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = s.replace("\r\n","\n").replace("\r","\n")
    if cfg.collapse_whitespace: s = "\n".join(re.sub(r"[ \t]+"," ", ln) for ln in s.split("\n"))
    if cfg.remove_empty_lines:  s = "\n".join(ln for ln in s.split("\n") if ln.strip())
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
        t = ln.strip()
        if re.fullmatch(r"Page\s+\d+(?:\s*/\s*\d+)?", t, flags=re.I): continue
        if re.fullmatch(r"\d{1,4}", t): continue
        keep.append(ln)
    return "\n".join(keep)

# ---------------- Parser (matrix of 14 numbers) ----------------
SECTION_ALIASES = {
    "delivery": "Delivery","dine in": "Dine-In","dine-in": "Dine-In",
    "carryout": "Carryout","carry out": "Carryout","takeout": "Takeout",
    "to go": "To-Go","to-go": "To-Go"
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
                labels = [p.replace(" ","_") for p in parts[:expected-1]]
                labels.append("Total"); return labels
    return [f"col{i:02d}" for i in range(1, expected+1)]

def metric_line_pattern(ncols: int = 14) -> re.Pattern:
    nums = r"\s+".join([fr"(?P<c{i:02d}>\d+)" for i in range(1, ncols+1)])
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
        vals = [gd[f"c{i:02d}"] for i in range(1, ncols+1)]
        row = {"section": norm_section(section), "metric": metric_name}
        for j, lab in enumerate(labels):
            v = vals[j]
            row[lab] = int(v) if v and v.isdigit() else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty: df = df[["section","metric"] + labels]
    return df, labels

# ---------------- Mapping engine ----------------
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

def compute_presence(df: pd.DataFrame, mapping: Dict[str, Any]) -> Dict[str, bool]:
    """For appearance: identify metrics that are present in the PDF (so we can show '—' if absent)."""
    if df.empty: return {}
    ci_default = bool(mapping.get("case_insensitive", False))
    present = {}
    for idx, rule in enumerate(mapping.get("metrics", []), start=1):
        label = rule.get("label", f"Rule {idx}")
        mask = df.apply(lambda r: match_metric(r["metric"], rule, ci_default) and section_allowed(r["section"], rule), axis=1)
        present[label] = bool(mask.any())
    return present

# ---------------- Category definitions (7 / 7 / 7) ----------------
MISSING = [
    "Missing food","Order wrong","Missing condiments","Out of menu item",
    "Missing bev","Missing ingredients","Packaging to-go complaint"
]
ATTITUDE = [
    "Unprofessional/Unfriendly","Manager directly involved","Manager not available",
    "Manager did not visit","Negative mgr-employee exchange","Manager did not follow up",
    "Argued with guest"
]
OTHER = [
    "Long hold/no answer","No/insufficient compensation offered","Did not attempt to resolve",
    "Guest left without ordering","Unknowledgeable","Did not open on time","No/poor apology"
]
ALL_21 = MISSING + ATTITUDE + OTHER

CATEGORY_META = {
    "To-go Missing Complaints": {"labels": MISSING,  "bg": "bg-missing",  "fg": "fg-light", "swatch": "#155e75"},
    "Attitude":                   {"labels": ATTITUDE, "bg": "bg-attitude", "fg": "fg-light", "swatch": "#3730a3"},
    "Other":                      {"labels": OTHER,    "bg": "bg-other",    "fg": "fg-light", "swatch": "#334155"}
}

# ---------------- Header (non-sticky) ----------------
def render_header():
    logo_path = "greenoosu.webp"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64 = f.read()
        # Streamlit: show via st.image for simplicity (keeps alt text)
        cols = st.columns([1,5])
        with cols[0]:
            st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
            st.image(logo_path, caption=None, use_container_width=False, width=140)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(
                f"""
                <div class="header-wrap">
                  <div class="title">Greeno Bad Three</div>
                </div>
                """, unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div class="header-wrap">
              <div class="logo"></div>
              <div class="title">Greeno Bad Three</div>
            </div>
            """, unsafe_allow_html=True
        )

render_header()

# ---------------- Upload ----------------
pdf_file = st.file_uploader("Upload PDF report", type=["pdf"])
if not pdf_file:
    st.stop()

# ---------------- Pre-render tiles with shimmer ----------------
# Build row headers + 7-square grids; placeholders keyed by metric label
placeholders: Dict[str, st.delta_generator.DeltaGenerator] = {}

def row_header_html(title: str, swatch_hex: str) -> str:
    return f"""
    <div class="row-header">
      <div class="swatch" style="background:{swatch_hex}"></div>
      <div class="row-title">{title}</div>
    </div>
    <div class="row-divider"></div>
    """

st.markdown("### Scoreboard")
for row_title, meta in CATEGORY_META.items():
    st.markdown(row_header_html(row_title, meta["swatch"]), unsafe_allow_html=True)
    st.markdown('<div class="row-scroll"><div class="row-inner">', unsafe_allow_html=True)
    for lab in meta["labels"]:
        ph = st.empty()
        placeholders[lab] = ph
        st.markdown(
            f"""
            <div class="tile {meta['bg']} {meta['fg']}" title="{lab}">
              <div class="tile-inner">
                <p class="title-small">{lab}</p>
                <div class="value-wrap">
                  <div class="skel"><div class="skel-text">Roll Tide…</div></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div></div>', unsafe_allow_html=True)

# ---------------- Parse + aggregate ----------------
cfg = ExtractConfig()
with st.spinner("Parsing PDF and computing totals…"):
    raw_bytes = io.BytesIO(pdf_file.read())
    raw_text, _pages = extract_pdf_text(raw_bytes)
    txt = strip_headers_footers(raw_text, cfg)
    if cfg.remove_page_numbers: txt = remove_page_numbers(txt)
    txt = normalize_text(txt, cfg)
    df_wide, labels = parse_matrix_blocks(txt, ncols=14)

if df_wide.empty:
    st.error("No matrix rows matched. Ensure header includes 'Reason for Contact' and rows end with 14 numbers.")
    st.stop()

mapping_cfg = load_mapping_constant()
result_df = apply_mapping(df_wide, labels, mapping_cfg)
present_map = compute_presence(df_wide, mapping_cfg)

# Select latest period by default
period_choices = [c for c in labels if c.lower() != "total"]
latest = pick_latest_period_label(labels) or (period_choices[-1] if period_choices else None)
sel_idx = period_choices.index(latest) if latest in period_choices else 0
period_label = st.selectbox("Period", options=period_choices, index=sel_idx)

# ---------------- Fill tiles ----------------
def get_value_for(label: str, period: str) -> Optional[int]:
    if result_df.empty or (label not in result_df["label"].values) or (period not in result_df.columns):
        return None
    return int(result_df.loc[result_df["label"] == label, period].sum())

for row_title, meta in CATEGORY_META.items():
    for lab in meta["labels"]:
        val = get_value_for(lab, period_label)
        has_data = present_map.get(lab, False)
        text_val = "—" if (not has_data) else str(int(val or 0))
        placeholders[lab].markdown(
            f"""
            <div class="tile {meta['bg']} {meta['fg']}" title="{lab}">
              <div class="tile-inner">
                <p class="title-small">{lab}</p>
                <div class="value-wrap"><p class="value">{text_val}</p></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
