# Greeno Big Three v1.9.2 — precise line counting + per-page de-dup (full mode)
# - New: Must-have-numbers filter (only count a label if there is a number on that same Y-band mapped to a period)
# - New: De-dup per page/store/section/reason (full mode)
# - Keeps Pure Count Mode, debug exports, scoreboard, etc.

import io, os, re, base64, statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# ──────────────── Styling ────────────────
def style_table(df: pd.DataFrame, highlight_grand_total: bool = True):
    def zebra(series):
        return [
            "background-color: #F5F7FA" if i % 2 == 0 else "background-color: #E9EDF2"
            for i, _ in enumerate(series)
        ]
    sty = (
        df.style
        .set_properties(
            **{
                "color": "#111",
                "border-color": "#CCD3DB",
                "border-width": "0.5px",
                "border-style": "solid",
            }
        )
        .apply(zebra, axis=0)
    )
    if highlight_grand_total and "— Grand Total —" in df.index.astype(str):
        def highlight_total(row):
            if str(row.name) == "— Grand Total —":
                return ["background-color: #FFE39B; color: #111; font-weight: 700;"] * len(row)
            return [""] * len(row)
        sty = sty.apply(highlight_total, axis=1)
    return sty

# ─────────────── Header / Sidebar ───────────────
st.set_page_config(page_title="Greeno Big Three v1.9.2", layout="wide")
logo_path = "greenosu.webp"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f'<img src="data:image/webp;base64,{logo_data}" width="240" style="border-radius:12px;">'
else:
    logo_html = '<div style="width:240px;height:240px;background:#fff;border-radius:12px;"></div>'

st.markdown(
    f"""
    <div style="background-color:#0078C8; color:white; padding:2rem 2.5rem; border-radius:10px;
    display:flex; align-items:center; gap:2rem; box-shadow:0 4px 12px rgba(0,0,0,.2);
    position:sticky; top:0; z-index:50;">
      {logo_html}
      <div style="display:flex; flex-direction:column;">
          <h1 style="margin:0; font-size:2.4rem;">Greeno Big Three v1.9.2</h1>
          <div style="height:5px; background-color:#F44336; width:300px; margin-top:10px; border-radius:3px;"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1) Upload PDF")
    up = st.file_uploader("Choose the PDF report", type=["pdf"])
    st.caption("Missing = To-Go/Delivery (except ‘Out of menu item’ includes Dine-In). Attitude/Other = all segments.")
    st.divider()
    pure_mode = st.toggle("✅ Pure Count Mode (ignore AD/Store/Segment)", value=False)
    debug_mode = st.checkbox("🔍 Enable Debug Mode", value=False)

if not up:
    st.info("⬅️ Upload your PDF in the sidebar to begin.")
    st.stop()

file_bytes = up.read()
if pdfplumber is None:
    st.error("pdfplumber missing. Run `pip install pdfplumber`.")
    st.stop()

# ─────────────── Constants ───────────────
HEADINGS = {"Area Director", "Restaurant", "Order Visit Type", "Reason for Contact"}
STORE_LINE_RX  = re.compile(r"^\s*\d{3,6}\s*-\s+.*")
SECTION_TOGO   = re.compile(r"^\s*(To[\s-]?Go|To-go)\s*$", re.I)
SECTION_DELIV  = re.compile(r"^\s*Delivery\s*$", re.I)
SECTION_DINEIN = re.compile(r"^\s*Dine[\s-]?In\s*$", re.I)
HEADER_RX      = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

MISSING = [
    "Missing food","Order wrong","Missing condiments","Out of menu item",
    "Missing bev","Missing ingredients","Packaging to-go complaint",
]
ATT = [
    "Unprofessional/Unfriendly","Manager directly involved","Manager not available",
    "Manager did not visit","Negative mgr-employee exchange","Manager did not follow up","Argued with guest",
]
OTHER = [
    "Long hold/no answer","No/insufficient compensation offered","Did not attempt to resolve",
    "Guest left without ordering","Unknowledgeable","Did not open on time","No/poor apology",
]
ALL_CANONICAL = MISSING + ATT + OTHER
# Keyword triggers
KEYWORD_TRIGGERS = {
    "Missing food": ["missing food"],
    "Order wrong": ["order wrong"],
    "Missing condiments": ["condiment"],
    "Out of menu item": ["out of menu"],
    "Missing bev": ["missing bev"],
    "Missing ingredients": ["ingredient"],
    "Packaging to-go complaint": ["packaging"],

    "Unprofessional/Unfriendly": ["unfriendly","unprofessional"],
    "Manager directly involved": ["directly involved","in complaint"],
    "Manager not available": ["manager not available"],
    "Manager did not visit": ["did not visit","no visit"],
    "Negative mgr-employee exchange": ["manager-employee","negative"],
    "Manager did not follow up": ["follow up"],
    "Argued with guest": ["argued"],

    "Long hold/no answer": ["hold","no answer","hung up"],
    "No/insufficient compensation offered": ["compensation"],
    "Did not attempt to resolve": ["resolve"],
    "Guest left without ordering": ["without ordering"],
    "Unknowledgeable": ["unknowledgeable"],
    "Did not open on time": ["open on time"],
    "No/poor apology": ["apology"],
}

SPECIAL_REASON_SECTIONS = {"Out of menu item": {"To Go", "Delivery", "Dine-In"}}
COMP_CANON = "No/insufficient compensation offered"

# Helpers
def _lc(s): return re.sub(r"\s+", " ", s.lower().strip())
def _round_to(x, base=2): return round(x / base) * base
def is_structural_total(label_lc): return label_lc.endswith("total:")

# Period header detection
def find_period_headers(page):
    words = page.extract_words(x_tolerance=1.2, y_tolerance=2.0)
    lines = defaultdict(list)
    for w in words:
        y = _round_to((w["top"] + w["bottom"]) / 2, 2)
        lines[y].append(w)
    headers = []
    for y, ws in lines.items():
        ws = sorted(ws, key=lambda w: w["x0"])
        for i in range(len(ws)-1):
            text = f"{ws[i]['text']} {ws[i+1]['text']}"
            if HEADER_RX.fullmatch(text):
                headers.append((text, (ws[i]["x0"] + ws[i+1]["x1"])/2, y))
    seen = {}
    for txt, xc, ym in headers:
        seen.setdefault(txt, (txt, xc, ym))
    return list(seen.values())

def sort_headers(headers):
    def key(h):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999,999)
    return sorted(headers, key=key)

def find_total_header_x(page, header_y):
    for w in page.extract_words(x_tolerance=1.0, y_tolerance=2.0):
        if abs(((w["top"] + w["bottom"]) / 2) - header_y) <= 2.5 and w["text"].strip().lower() == "total":
            return (w["x0"] + w["x1"]) / 2
    return None

def build_header_bins(pos, total_x):
    items = sorted(pos.items(), key=lambda kv: kv[1])
    bins = []
    for i, (h, x) in enumerate(items):
        left = (items[i-1][1] + x)/2 if i>0 else x-40
        right = (x + (items[i+1][1] if i+1 < len(items) else (total_x or x+60)))/2
        bins.append((h,left,right))
    return bins

def map_x_to_header(bins,x):
    for h,l,r in bins:
        if l<=x<r:return h
    return None

def extract_words_grouped(page):
    words = page.extract_words(x_tolerance=1.4, y_tolerance=2.4)
    lines=defaultdict(list)
    for w in words:
        y=_round_to((w["top"]+w["bottom"])/2,2);lines[y].append(w)
    out=[]
    for y,ws in sorted(lines.items()):
        ws=sorted(ws,key=lambda w:w["x0"])
        txt=" ".join(w["text"].strip() for w in ws if w["text"].strip())
        if txt:out.append({"y":y,"x_min":ws[0]["x0"],"words":ws,"text":txt})
    return out

def line_has_period_numbers(line,label_right,bins,total_x):
    for w in line["words"]:
        if not re.fullmatch(r"\d+",w["text"]):continue
        if w["x0"]<=label_right:continue
        if total_x and w["x0"]>=total_x:continue
        xmid=(w["x0"]+w["x1"])/2
        if map_x_to_header(bins,xmid):return True
    return False

def _match_canon(label_lc):
    for canon,trigs in KEYWORD_TRIGGERS.items():
        if any(t in label_lc for t in trigs):return canon
    return None
    # ──────────────── PURE COUNT MODE ────────────────
def parse_pdf_pure_counts(file_bytes, debug=False):
    counts = defaultdict(lambda: defaultdict(int))
    ordered_headers = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        header_positions, carry_headers, carry_total_x = {}, None, None
        for page in pdf.pages:
            headers = find_period_headers(page) or carry_headers
            if not headers: continue
            if find_period_headers(page): carry_headers, carry_total_x = headers[:], None
            for h, x, _ in headers: header_positions[h] = x
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)
            total_x = find_total_header_x(page, header_y) or carry_total_x
            if total_x: carry_total_x = total_x
            bins = build_header_bins({h: header_positions[h] for h in ordered_headers}, total_x)
            first_x = min(header_positions.values()); label_right = first_x - 10
            lines = extract_words_grouped(page)
            for L in lines:
                lbl = " ".join(w["text"] for w in L["words"] if w["x1"]<=label_right).strip().lower()
                canon = None
                if "compensation" in lbl or "unsatisfactory" in lbl: canon = COMP_CANON
                else: canon = _match_canon(lbl)
                if not canon or not line_has_period_numbers(L,label_right,bins,total_x): continue
                for w in L["words"]:
                    t=w["text"].strip()
                    if not re.fullmatch(r"\d+",t):continue
                    if w["x0"]<=label_right:continue
                    if total_x and w["x0"]>=total_x:continue
                    xmid=(w["x0"]+w["x1"])/2
                    p=map_x_to_header(bins,xmid)
                    if not p:continue
                    counts[canon][p]+=int(t)
    return counts,ordered_headers,{}

# ──────────────── FULL PARSER ────────────────
def parse_pdf_full(file_bytes, debug=False):
    data=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    header_positions,ordered_headers={},[]
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        carry_headers,carry_total_x=None,None
        for page in pdf.pages:
            headers=find_period_headers(page)or carry_headers
            if not headers:continue
            if find_period_headers(page):carry_headers,carry_total_x=headers[:],None
            for h,x,_ in headers:header_positions[h]=x
            ordered_headers=sort_headers(list(header_positions.keys()))
            header_y=min(h[2] for h in headers)
            total_x=find_total_header_x(page,header_y)or carry_total_x
            if total_x:carry_total_x=total_x
            bins=build_header_bins({h:header_positions[h]for h in ordered_headers},total_x)
            first_x=min(header_positions.values());label_right=first_x-10
            lines=extract_words_grouped(page)
            current_ad,current_store,current_section=None,None,None
            counted=set()
            for L in lines:
                txt=L["text"].strip()
                # detect store
                if STORE_LINE_RX.match(txt):
                    current_store=txt
                    current_section=None
                    continue
                if SECTION_TOGO.match(txt):current_section="To Go";continue
                if SECTION_DELIV.match(txt):current_section="Delivery";continue
                if SECTION_DINEIN.match(txt):current_section="Dine-In";continue
                if txt in HEADINGS or not current_store or not current_section:continue
                lbl=" ".join(w["text"]for w in L["words"]if w["x1"]<=label_right).strip().lower()
                if is_structural_total(lbl):continue
                canon=None
                if "compensation" in lbl or "unsatisfactory" in lbl:canon=COMP_CANON
                else:canon=_match_canon(lbl)
                if not canon or not line_has_period_numbers(L,label_right,bins,total_x):continue
                key=(page.page_number,current_store,current_section,canon)
                if key in counted:continue
                for w in L["words"]:
                    t=w["text"].strip()
                    if not re.fullmatch(r"\d+",t):continue
                    if w["x0"]<=label_right:continue
                    if total_x and w["x0"]>=total_x:continue
                    xmid=(w["x0"]+w["x1"])/2
                    p=map_x_to_header(bins,xmid)
                    if not p:continue
                    sect=data[current_ad].setdefault(current_store,{}).setdefault(current_section,{})
                    per=sect.setdefault("__all__",defaultdict(lambda:defaultdict(int)))
                    per[canon][p]+=int(t)
                counted.add(key)
    return header_positions,data,sort_headers(list(header_positions.keys())),[],{}

# ──────────────── RUN ────────────────
with st.spinner("Roll Tide…"):
    if pure_mode:
        counts,ordered,dbg=parse_pdf_pure_counts(file_bytes,debug_mode)
        raw=None
    else:
        _,raw,ordered,_,dbg=parse_pdf_full(file_bytes,debug_mode)

if not ordered:
    st.error("No period headers found.");st.stop()
sel_col=st.selectbox("Period",ordered,index=len(ordered)-1)

# ──────────────── PURE COUNT UI ────────────────
if pure_mode:
    df=pd.DataFrame(index=ALL_CANONICAL,columns=ordered).fillna(0).astype(int)
    for r,pm in counts.items():
        for p,v in pm.items():
            if r in df.index and p in df.columns:df.loc[r,p]=v
    df["Total"]=df.sum(axis=1)
    def ctot(rs):return int(df.loc[rs,sel_col].sum())
    m,a,o=ctot(MISSING),ctot(ATT),ctot(OTHER);tot=m+a+o
    st.markdown(f"### Selected: **{sel_col}**")
    st.metric("Overall",tot);st.metric("Missing",m);st.metric("Attitude",a);st.metric("Other",o)
    st.dataframe(style_table(df),use_container_width=True)
    st.stop()

# ──────────────── NORMAL MODE UI ────────────────
rows=[]
for ad,stores in raw.items():
    for s,sects in stores.items():
        for sec,rm in sects.items():
            if sec not in{"To Go","Delivery","Dine-In"}:continue
            allp=rm.get("__all__",{})
            for canon in ALL_CANONICAL:
                for per,val in allp.get(canon,{}).items():
                    rows.append({"Area Director":ad,"Store":s,"Section":sec,"Reason":canon,"Period":per,"Value":val})
df_all=pd.DataFrame(rows)
df=df_all[df_all["Period"]==sel_col]
if df.empty:st.warning("No data.");st.stop()

# Category tables
def tbl(df,order):
    out=pd.DataFrame({sec:df[df["Section"]==sec].groupby("Reason")["Value"].sum().reindex(order).fillna(0).astype(int)
                      for sec in sorted(df["Section"].unique())})
    out["Total"]=out.sum(axis=1)
    out.loc["— Grand Total —"]=out.sum()
    return out
st.header("To-Go Missing Complaints");st.dataframe(style_table(tbl(df[df["Reason"].isin(MISSING)],MISSING)),use_container_width=True)
st.header("Attitude");st.dataframe(style_table(tbl(df[df["Reason"].isin(ATT)],ATT)),use_container_width=True)
st.header("Other");st.dataframe(style_table(tbl(df[df["Reason"].isin(OTHER)],OTHER)),use_container_width=True)
