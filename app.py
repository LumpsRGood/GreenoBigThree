# path: app.py
# Greeno Bad Three — polished scoreboard (fixed 7×3 grid, centered totals, “Roll Tide…” shimmer, non-sticky header)

from __future__ import annotations

import io, os, re, json, unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import base64

# ---------------- Page config (must be first) ----------------
st.set_page_config(page_title="Greeno Bad Three", page_icon="🐘", layout="wide")

# ---------------- Deps ----------------
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
    { "label": "Missing food", "patterns": ["Item \\(Food\\)"], "regex": true, "sections": ["To-Go","Dine-In","Delivery"] },
    { "label": "Order wrong", "patterns": ["Order Wrong"], "regex": false, "sections": ["To-Go","Delivery"] },
    { "label": "Missing condiments", "patterns": ["Missing Condiments"], "regex": false, "sections": ["To-Go","Delivery"] },
    { "label": "Out of menu item", "patterns": ["Out Of Menu Item"], "regex": false, "sections": ["To-Go","Delivery"] },
    { "label": "Missing bev", "patterns": ["Item \\(Bev\\)"], "regex": true, "sections": ["To-Go","Delivery"] },
    { "label": "Missing ingredients", "patterns": ["Ingredient \\(Food\\)"], "regex": true, "sections": ["To-Go","Delivery"] },
    { "label": "Packaging to-go complaint", "patterns": ["Packaging To Go Complaint"], "regex": false, "sections": ["To-Go","Delivery"] },

    { "label": "Unprofessional/Unfriendly", "patterns": ["Unfriendly Attitude","Unprofessional Behavior"], "regex": false, "sections": ["*"] },
    { "label": "Manager directly involved", "patterns": ["^Manager\\s*Directly"], "regex": true, "sections": ["*"] },
    { "label": "Manager not available", "patterns": ["Management Not Available"], "regex": false, "sections": ["*"] },
    { "label": "Manager did not visit", "patterns": ["Not Visit","Manager Did Not Visit"], "regex": true, "sections": ["*"] },
    { "label": "Negative mgr-employee exchange", "patterns": ["Negative Manager","Negative Manager-Employee Interaction"], "regex": true, "sections": ["*"] },
    { "label": "Manager did not follow up", "patterns": ["Manager Did Not Follow","Manager Did Not Follow Up"], "regex": true, "sections": ["*"] },
    { "label": "Argued with guest", "patterns": ["Argued","Argued With Guest"], "regex": true, "sections": ["*"] },

    { "label": "Long hold/no answer", "patterns": ["Long Hold","No Answer","Hung Up","Long Hold/No Answer/Hung Up"], "regex": true, "sections": ["*"] },
    { "label": "No/insufficient compensation offered", "patterns": ["No/Unsatisfactory","No/Unsatisfactory Compensation Offered By Restaurant"], "regex": true, "sections": ["*"] },
    { "label": "Did not attempt to resolve", "patterns": ["Resolve","Did Not Attempt To Resolve"], "regex": true, "sections": ["*"] },
    { "label": "Guest left without ordering", "patterns": ["Guest Left","Guest Left Without Dining or Ordering"], "regex": true, "sections": ["*"] },
    { "label": "Unknowledgeable", "patterns": ["Unknowledgeable"], "regex": false, "sections": ["*"] },
    { "label": "Did not open on time", "patterns": ["Open/close","Didn’t Open/close On Time","Didn't Open/close On Time"], "regex": true, "sections": ["*"] },
    { "label": "No/poor apology", "patterns": ["No/Poor Apology"], "regex": false, "sections": ["*"] }
  ]
}
"""

# ---------------- Appearance (CSS) ----------------
UI_CSS = """
<style>
:root{
  --tile-radius:12px; --tile-gap:10px;
  --shadow: 0 6px 18px rgba(0,0,0,.10), 0 2px 6px rgba(0,0,0,.06);
}

/* ===== Hero header (non-sticky) ===== */
.hero{display:flex;align-items:flex-end;gap:14px;margin:4px 0 10px 0;padding:0;}
.hero-title{font-size:clamp(1.8rem, 3vw, 2.4rem);font-weight:900;margin:0;color:#e5e7eb;line-height:1;}
.hero-sub{font-size:.95rem;color:#cbd5e1;opacity:.85;margin-top:2px}

/* Fixed background watermark (stays put while scrolling) */
.bg-mark{position:fixed;top:18px;left:18px;width:min(9vw, 220px);opacity:.10;
  filter:saturate(1) contrast(1.05);z-index:0;pointer-events:none;}
/* Lift app content above the watermark */
[data-testid="stAppViewContainer"] .main .block-container{position:relative;z-index:1}

/* ===== Row headers ===== */
.row-header{display:flex;align-items:center;gap:8px;height:32px;margin:10px 0 8px 0;}
.swatch{width:12px;height:12px;border-radius:4px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.25);}
.row-title{font-size:.95rem;font-weight:700;margin:0;color:#e5e7eb;}

/* ===== Grid (exactly 7 columns) ===== */
.row-inner{display:grid;grid-template-columns:repeat(7, minmax(110px, 1fr));gap:var(--tile-gap);}
@media (max-width:1100px){ .row-wrap{overflow-x:auto;padding-bottom:2px;} .row-inner{min-width:900px;} }

/* ===== Tiles (square) ===== */
.tile{position:relative;width:100%;aspect-ratio:1/1;border-radius:var(--tile-radius);overflow:hidden;box-shadow:var(--shadow);}
.tile-inner{position:absolute;inset:0;display:grid;grid-template-rows:auto 1fr;padding:10px 12px;}
.title-small{font-size:.78rem;font-weight:800;margin:0;color:#fff;opacity:.98;text-shadow:0 1px 2px rgba(0,0,0,.18);
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.value-wrap{display:flex;align-items:center;justify-content:center;}
.value{font-weight:900;line-height:1;margin:0;letter-spacing:-0.3px;font-variant-numeric:tabular-nums;
  font-size:clamp(2.6rem, 5.8vw, 3.6rem);text-shadow:0 2px 4px rgba(0,0,0,.22);}

/* Inner border to reduce banding & hover */
.tile::after{content:"";position:absolute;inset:0;border-radius:var(--tile-radius);box-shadow:inset 0 0 0 1px rgba(255,255,255,.12);pointer-events:none;}
.tile:hover{box-shadow:0 10px 22px rgba(0,0,0,.14), 0 4px 10px rgba(0,0,0,.08);}

/* Shimmer with 'Roll Tide…' */
.skel{position:relative;width:78%;max-width:280px;height:1.6em;border-radius:10px;opacity:.95;
  background:linear-gradient(90deg, rgba(255,255,255,.14) 25%, rgba(255,255,255,.32) 37%, rgba(255,255,255,.14) 63%);
  background-size:400% 100%;animation:shimmer 1.3s linear infinite;}
.skel-text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:1.1rem;opacity:.9;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.28);}
@keyframes shimmer {0%{background-position:100% 0}100%{background-position:-100% 0}}
@media (prefers-reduced-motion: reduce){ .skel{animation:none;background:rgba(255,255,255,.18);} }

/* Category gradients */
.bg-missing{background:linear-gradient(135deg,#155e75,#22d3ee);}
.bg-attitude{background:linear-gradient(135deg,#3730a3,#818cf8);}
.bg-other{background:linear-gradient(135deg,#334155,#64748b);}
.fg-light{color:#ffffff;}
</style>
"""
st.markdown(UI_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
/* Centered, responsive watermark */
.bg-mark{
  position: fixed;
  top: 14px;                 /* push down a touch so it's never clipped */
  left: 50%;                 /* center horizontally */
  transform: translateX(-50%);
  width: clamp(320px, 36vw, 680px);   /* scales with screen, but bounded */
  max-height: 34vh;          /* never taller than ~1/3 of the viewport */
  opacity: .16;              /* muted but visible */
  filter: brightness(1.1) saturate(1.05) contrast(1.1)
          drop-shadow(0 6px 16px rgba(0,0,0,.25));
  z-index: 0;
  pointer-events: none;
}
.bg-mark img{
  display:block;
  width:100%;
  height:auto;
  object-fit: contain;       /* prevent any cropping inside the box */
  border-radius: 12px;       /* optional: rounds it slightly */
}

/* Tune for small screens */
@media (max-width: 900px){
  .bg-mark{
    top: 8px;
    width: clamp(240px, 60vw, 520px);
    max-height: 28vh;
    opacity: .18;
  }
}
</style>
""", unsafe_allow_html=True)

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
            pages.append(page.extract_text(layout=True) or "")
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
        t=ln.strip()
        if re.fullmatch(r"Page\s+\d+(?:\s*/\s*\d+)?", t, flags=re.I): continue
        if re.fullmatch(r"\d{1,4}", t): continue
        keep.append(ln)
    return "\n".join(keep)

# ---------------- Parser (rows ending with 14 numbers) ----------------
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

# ---------------- Categories (7 per row) ----------------
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
CATEGORY_META = {
    "To-go Missing Complaints": {"labels": MISSING,  "bg": "bg-missing",  "swatch": "#155e75"},
    "Attitude":                   {"labels": ATTITUDE, "bg": "bg-attitude", "swatch": "#3730a3"},
    "Other":                      {"labels": OTHER,    "bg": "bg-other",    "swatch": "#334155"}
}

# ---------------- Header (simple, non-sticky) ----------------
def render_header():
    # Background watermark (fixed)
    if os.path.exists("greenoosu.webp"):
        with open("greenoosu.webp", "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"<div class='bg-mark'><img src='data:image/webp;base64,{b64}' alt='Greeno watermark'/></div>",
            unsafe_allow_html=True
        )

    # Compact hero title (no inline image; watermark handles the brand)
    st.markdown(
        """
        <div class="hero">
          <h1 class="hero-title">Greeno Bad Three</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

render_header()

# ---------------- Upload + period UI ----------------
pdf_file = st.file_uploader("Upload PDF report", type=["pdf"])
if not pdf_file:
    st.stop()

# Row placeholders (entire row at once to preserve grid)
row_ph: Dict[str, st.delta_generator.DeltaGenerator] = {title: st.empty() for title in CATEGORY_META.keys()}

def render_row_html(title: str, bg_class: str, swatch_hex: str, items: List[Tuple[str, Optional[int]]], loading: bool) -> str:
    tiles_html = ""
    for label, val in items:
        if loading:
            inner = '<div class="skel"><div class="skel-text">Roll Tide…</div></div>'
        else:
            text_val = "—" if (val is None) else str(int(val))
            inner = f'<p class="value">{text_val}</p>'
        tiles_html += f"""
        <div class="tile {bg_class} fg-light" title="{label}">
          <div class="tile-inner">
            <p class="title-small">{label}</p>
            <div class="value-wrap">{inner}</div>
          </div>
        </div>
        """
    return f"""
      <div class="row-header">
        <div class="swatch" style="background:{swatch_hex}"></div>
        <div class="row-title">{title}</div>
      </div>
      <div class="row-wrap">
        <div class="row-inner">{tiles_html}</div>
      </div>
    """

# Initial shimmer rows
for title, meta in CATEGORY_META.items():
    items = [(lab, None) for lab in meta["labels"]]
    row_ph[title].markdown(render_row_html(title, meta["bg"], meta["swatch"], items, loading=True), unsafe_allow_html=True)

# ---------------- Parse + aggregate ----------------
cfg = ExtractConfig()
with st.spinner("Parsing PDF and computing totals…"):
    raw = io.BytesIO(pdf_file.read())
    raw_text, _pages = extract_pdf_text(raw)
    txt = strip_headers_footers(raw_text, cfg)
    if cfg.remove_page_numbers: txt = remove_page_numbers(txt)
    txt = normalize_text(txt, cfg)
    df_wide, labels = parse_matrix_blocks(txt, ncols=14)

if df_wide.empty:
    st.error("No matrix rows matched. Make sure rows end with 14 numbers and the header line contains the period labels.")
    st.stop()

mapping_cfg = load_mapping_constant()
result_df = apply_mapping(df_wide, labels, mapping_cfg)

period_choices = [c for c in labels if c.lower() != "total"]
latest = pick_latest_period_label(labels) or (period_choices[-1] if period_choices else None)
sel_idx = period_choices.index(latest) if latest in period_choices else 0
period_label = st.selectbox("Period", options=period_choices, index=sel_idx)

def val_for(metric_label: str, period: str) -> Optional[int]:
    if result_df.empty or (metric_label not in result_df["label"].values) or (period not in result_df.columns):
        return None
    return int(result_df.loc[result_df["label"] == metric_label, period].sum())

# Fill rows with values (single re-render per row to keep the grid intact)
for title, meta in CATEGORY_META.items():
    items = [(lab, val_for(lab, period_label)) for lab in meta["labels"]]
    row_ph[title].markdown(render_row_html(title, meta["bg"], meta["swatch"], items, loading=False), unsafe_allow_html=True)
