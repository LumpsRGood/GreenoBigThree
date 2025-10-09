# Greeno Big Three v1.5.5 (hotfix)
# Run: streamlit run app.py

import io
import re
import unicodedata
from collections import OrderedDict
from statistics import median

import pandas as pd
import streamlit as st
import pdfplumber

# ========== Normalizers & Helpers ==========

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("–", "-")
           .replace("—", "-")
           .replace("ﬁ", "fi")
           .replace("ﬂ", "fl"))
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _only_digits(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(,\d{3})*|\d+", s or ""))

def _clean_int(s: str) -> int:
    try:
        return int((s or "0").replace(",", ""))
    except Exception:
        return 0

# Locked reasons (exactly 7)
ALLOWED_REASONS = {
    _norm("Missing Item (Food)"):       "Missing food",
    _norm("Order Wrong"):               "Order wrong",
    _norm("Missing Condiments"):        "Missing condiments",
    _norm("Out Of Menu Item"):          "Out of menu item",
    _norm("Missing Item (Bev)"):        "Missing bev",
    _norm("Missing Ingredient (Food)"): "Missing ingredients",
    _norm("Packaging To Go Complaint"): "Packaging to-go complaint",
}

# Sections
SECTION_PATTERNS = [r"\bto\s*-?\s*go\b", r"\bdelivery\b"]  # "To Go", "To-Go", "ToGo"; "Delivery"

# Periods — tolerate P7 25, P07 25, P7'25, P7/25, P7 2025, P07 2025, etc.
PERIOD_PATTERN = re.compile(
    r"\bP\s*0?(\d{1,2})\s*(?:['/]|-|-)?\s*(?:20)?(\d{2})\b", re.IGNORECASE
)

def normalize_period_label(text: str) -> str:
    """
    Normalize various forms to a canonical label like 'P7 25'
    """
    m = PERIOD_PATTERN.search(text or "")
    if not m:
        return None
    p = int(m.group(1))
    yy = int(m.group(2))
    return f"P{p} {yy:02d}"

# ========== Low-level PDF word/line utilities ==========

def group_lines_by_y(words):
    """
    Build [(y, text, [word_boxes])] in reading order.
    """
    lines = {}
    for w in words:
        y = round(w["top"], 1)
        lines.setdefault(y, []).append(w)
    rows = []
    for y in sorted(lines.keys()):
        row = sorted(lines[y], key=lambda ww: ww["x0"])
        text = " ".join([r["text"] for r in row])
        rows.append((y, text, row))
    return rows

def merge_wrapped_labels(rows):
    """
    Merge common wraps for 'Packaging To Go' + 'Complaint'
    """
    out = []
    i = 0
    while i < len(rows):
        y, text, boxes = rows[i]
        if i + 1 < len(rows):
            y2, text2, boxes2 = rows[i + 1]
            if _norm(text) in (_norm("Packaging To Go"), _norm("Packaging To-Go")) and _norm(text2) == _norm("Complaint"):
                out.append((y, f"{text} {text2}", boxes + boxes2))
                i += 2
                continue
        out.append((y, text, boxes))
        i += 1
    return out

def find_sections(words):
    """
    Find To Go / Delivery sections with y-bounds.
    """
    headers = []
    for w in words:
        txt = _norm(w.get("text", ""))
        for pat in SECTION_PATTERNS:
            if re.search(pat, txt):
                nm = "to go" if re.search(r"\bto\s*-?\s*go\b", txt) else "delivery"
                headers.append({"name": nm, "y0": w["top"]})
                break
    headers.sort(key=lambda h: h["y0"])
    if not headers:
        return []
    sections = []
    for i, h in enumerate(headers):
        y1 = words[-1]["bottom"] if i == len(headers) - 1 else headers[i + 1]["y0"] - 1
        sections.append({"name": h["name"], "y0": h["y0"], "y1": y1})
    return sections

# ========== Column binning (header or fallback) ==========

def build_period_bins_from_header(header_boxes):
    """
    From a line's word boxes, detect period labels and return bins [{label, x0, x1}].
    Merge fragments and normalize labels to 'P# YY'.
    """
    candidates = []
    for b in header_boxes:
        t = b.get("text", "")
        lbl = normalize_period_label(t)
        if lbl:
            candidates.append({"label": lbl, "x0": b["x0"], "x1": b["x1"]})
        elif re.search(r"\btotal\b", t, re.IGNORECASE):
            candidates.append({"label": "TOTAL", "x0": b["x0"], "x1": b["x1"]})

    # Merge overlaps
    candidates.sort(key=lambda k: k["x0"])
    merged = []
    for c in candidates:
        if not merged:
            merged.append(c)
        else:
            last = merged[-1]
            if c["x0"] - last["x1"] < 6:
                last["x1"] = max(last["x1"], c["x1"])
                if last["label"] == "TOTAL" and c["label"] != "TOTAL":
                    last["label"] = c["label"]
            else:
                merged.append(c)

    # Drop TOTAL bin
    bins = [b for b in merged if b["label"] != "TOTAL"]
    return sorted(bins, key=lambda b: b["x0"])

def build_period_bins_by_clustering(rows):
    """
    Fallback when no explicit header row is found:
    - collect numeric boxes for lines that match allowed reasons,
    - cluster by x-center into ~N columns (median-based),
    - infer labels from any text above/near each cluster; if none, make generic 'P? ??' slots.
    """
    # Gather numeric centers per y row that looks like a reason line
    numeric_points = []
    reason_rows = []
    for y, text, boxes in rows:
        txtn = _norm(re.sub(r"\s+total.*$", "", text, flags=re.IGNORECASE))
        if any(txtn.startswith(k) for k in ALLOWED_REASONS.keys()):
            reason_rows.append((y, text, boxes))
            for b in boxes:
                if _only_digits(b.get("text", "")):
                    cx = (b["x0"] + b["x1"]) / 2
                    numeric_points.append(cx)

    if not numeric_points:
        return []

    numeric_points.sort()
    # Heuristic: estimate column count by looking at gaps
    gaps = [numeric_points[i+1] - numeric_points[i] for i in range(len(numeric_points)-1)]
    if not gaps:
        return []
    big_gap = median(gaps) * 1.8
    clusters = [[numeric_points[0]]]
    for g, x in zip(gaps, numeric_points[1:]):
        if g > big_gap:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    bins = []
    for cl in clusters:
        cx = sum(cl) / len(cl)
        # approximate width from spread
        w = max(8.0, (max(cl) - min(cl)) if len(cl) > 1 else 12.0)
        bins.append({"label": None, "x0": cx - w/2, "x1": cx + w/2})

    # Try to label clusters using nearby text lines that contain a period pattern
    labels = []
    for y, text, boxes in rows:
        lbl = normalize_period_label(text)
        if not lbl:
            continue
        # approximate this line’s text extents per word to align labels to clusters
        for b in boxes:
            lb = normalize_period_label(b.get("text", ""))
            if not lb:
                continue
            cx = (b["x0"] + b["x1"]) / 2
            labels.append((cx, lb))

    labels.sort(key=lambda t: t[0])
    for i, bin_ in enumerate(sorted(bins, key=lambda b: b["x0"])):
        # nearest label by x
        if labels:
            j = min(range(len(labels)), key=lambda k: abs(labels[k][0] - (bin_["x0"] + bin_["x1"]) / 2))
            bin_["label"] = labels[j][1]
        else:
            bin_["label"] = f"P? {i+1:02d}"

    # Attempt to find and exclude a TOTAL column: look for a 'Total' token and drop its nearest bin
    total_x = []
    for y, text, boxes in rows:
        for b in boxes:
            if re.search(r"\btotal\b", b.get("text", ""), re.IGNORECASE):
                total_x.append((b["x0"] + b["x1"]) / 2)
    if total_x:
        tx = median(total_x)
        idx = min(range(len(bins)), key=lambda k: abs(((bins[k]["x0"] + bins[k]["x1"]) / 2) - tx))
        bins.pop(idx)

    return sorted(bins, key=lambda b: b["x0"])

def nearest_bin(cx, bins):
    if not bins:
        return None
    return min(range(len(bins)), key=lambda k: abs(cx - (bins[k]["x0"] + bins[k]["x1"]) / 2))

# ========== Extraction per section ==========

def extract_reason_rows(page, section):
    """
    Return list of {section, reason, period_values{label:int}} for this section y-range.
    """
    words = [
        w for w in page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if section["y0"] <= w["top"] <= section["y1"]
    ]
    if not words:
        return []

    rows = group_lines_by_y(words)
    rows = merge_wrapped_labels(rows)

    # Try to locate a header row with period codes
    header_idx = None
    for i, (_, text, _) in enumerate(rows):
        if normalize_period_label(text):
            header_idx = i
            break
    # Fallback: any line mentioning multiple period codes
    if header_idx is None:
        for i, (_, text, _) in enumerate(rows):
            if len(PERIOD_PATTERN.findall(text)) >= 2:
                header_idx = i
                break

    # Build bins
    if header_idx is not None:
        period_bins = build_period_bins_from_header(rows[header_idx][2])
    else:
        period_bins = build_period_bins_by_clustering(rows)

    parsed = []
    for y, text, boxes in rows:
        t = re.sub(r"\s+total.*$", "", text, flags=re.IGNORECASE)
        key = _norm(t)
        for allowed_key, out_label in ALLOWED_REASONS.items():
            if key.startswith(allowed_key):
                numeric_boxes = [b for b in boxes if _only_digits(b.get("text", ""))]
                row = {"section": section["name"], "reason": out_label, "period_values": {}}
                for nb in numeric_boxes:
                    cx = (nb["x0"] + nb["x1"]) / 2
                    j = nearest_bin(cx, period_bins)
                    if j is not None:
                        lbl = period_bins[j]["label"]
                        if lbl:  # skip if couldn't label
                            row["period_values"].setdefault(lbl, 0)
                            row["period_values"][lbl] += _clean_int(nb["text"])
                parsed.append(row)
                break
    return parsed

# ========== AD / Store detection (unchanged simple heuristic) ==========

def detect_store_and_ad(lines_before_idx):
    store = None
    ad = None
    for y, text, _ in reversed(lines_before_idx):
        m = re.search(r"\b(\d{4})\b", text)
        if m:
            store = m.group(1)
            break
    if store is not None:
        passed_store = False
        for y, text, _ in reversed(lines_before_idx):
            if store in text and not passed_store:
                passed_store = True
                continue
            t = text.strip()
            if t and not re.search(r"\d", t) and len(t) <= 40:
                ad = t
                break
    return ad, store

# ========== Page-level extraction ==========

def extract_page_data(page):
    all_words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not all_words:
        return []

    sections = find_sections(all_words)
    if not sections:
        return []

    all_rows = group_lines_by_y(all_words)
    out = []
    for sec in sections:
        if sec["name"] not in ("to go", "delivery"):
            continue
        sec_rows = extract_reason_rows(page, sec)

        # Anchor above the section header for AD/Store
        idx_anchor = max([i for i, (yy, _, _) in enumerate(all_rows) if yy <= sec["y0"]], default=0)
        ad, store = detect_store_and_ad(all_rows[:idx_anchor + 1])

        for r in sec_rows:
            r["ad"] = ad or "Unknown AD"
            r["store"] = store or "Unknown Store"
            out.append(r)
    return out

# ========== Aggregation ==========

def aggregate_results(records):
    detail_rows = []
    for rec in records:
        for period, val in rec.get("period_values", {}).items():
            detail_rows.append({
                "Area Director": rec["ad"],
                "Store": rec["store"],
                "Section": "To Go" if rec["section"] == "to go" else "Delivery",
                "Reason": rec["reason"],
                "Period": period,          # already normalized like 'P7 25'
                "Count": val
            })
    if not detail_rows:
        return pd.DataFrame(), pd.DataFrame()

    detail_df = pd.DataFrame(detail_rows)
    totals_df = (detail_df
                 .groupby(["Area Director", "Store", "Reason", "Period"], as_index=False)["Count"]
                 .sum())
    return detail_df, totals_df

# ========== UI (minimal; preserves your original look) ==========

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_records = []
    debug_pages = []

    for f in uploaded_files:
        try:
            data = f.read()
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for pageno, page in enumerate(pdf.pages, start=1):
                    page_recs = extract_page_data(page)
                    if not page_recs:
                        debug_pages.append((f.name, pageno, page))
                    all_records.extend(page_recs)
        except Exception as e:
            st.error(f"Error opening {f.name}: {e}")

    if not all_records:
        st.error("No To Go/Delivery matches found. Expand debug below to inspect raw text.")
        with st.expander("Debug: raw text by page"):
            for fname, pageno, page in debug_pages:
                st.markdown(f"**{fname} — Page {pageno}**")
                try:
                    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
                    st.text("\n".join(w["text"] for w in words))
                except Exception:
                    st.text("(Unable to extract words.)")
        st.stop()

    detail_df, totals_df = aggregate_results(all_records)
    if detail_df.empty or totals_df.empty:
        st.error("No aggregated rows after parsing. Open the debug expander below to verify period headers/columns.")
        with st.expander("Debug: parsed records (pre-aggregation)"):
            st.write(pd.DataFrame(all_records))
        st.stop()

    # Period filter
    all_periods = sorted(set(totals_df["Period"].tolist()),
                         key=lambda x: (int(re.search(r"P(\d{1,2})", x).group(1)) if re.search(r"P(\d{1,2})", x) else 99,
                                        x))
    sel_periods = st.multiselect("Select Period(s)", options=all_periods, default=all_periods)

    filt_totals = totals_df[totals_df["Period"].isin(sel_periods)].copy()
    filt_detail = detail_df[detail_df["Period"].isin(sel_periods)].copy()

    # Totals grid
    st.subheader("Totals by AD → Store → Reason")
    pivot = (filt_totals
             .pivot_table(index=["Area Director", "Store", "Reason"],
                          columns="Period",
                          values="Count",
                          aggfunc="sum",
                          fill_value=0)
             .reset_index())
    st.dataframe(pivot, use_container_width=True)

    # Drill-down
    st.subheader("Drill-down")
    ads = ["(All)"] + sorted(filt_detail["Area Director"].dropna().unique().tolist())
    stores = ["(All)"] + sorted(filt_detail["Store"].dropna().unique().tolist())
    reasons = ["(All)"] + list(OrderedDict.fromkeys([
        "Missing food",
        "Order wrong",
        "Missing condiments",
        "Out of menu item",
        "Missing bev",
        "Missing ingredients",
        "Packaging to-go complaint"
    ]).keys())

    sel_ad = st.selectbox("Area Director", options=ads, index=0)
    sel_store = st.selectbox("Store", options=stores, index=0)
    sel_reason = st.selectbox("Reason", options=reasons, index=0)

    dd = filt_detail.copy()
    if sel_ad != "(All)":
        dd = dd[dd["Area Director"] == sel_ad]
    if sel_store != "(All)":
        dd = dd[dd["Store"] == sel_store]
    if sel_reason != "(All)":
        dd = dd[dd["Reason"] == sel_reason]

    if dd.empty:
        st.info("No rows for this selection.")
    else:
        st.dataframe(dd.sort_values(["Area Director", "Store", "Reason", "Period"]),
                     use_container_width=True)

    # Export
    def to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    def to_xlsx_bytes(dfs: dict):
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            for sheet, frame in dfs.items():
                frame.to_excel(writer, index=False, sheet_name=sheet[:31])
        bio.seek(0)
        return bio.read()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV (Detail & Totals)",
            file_name="greeno_big_three_v1_5_5.csv",
            data=to_csv_bytes(pd.concat([
                filt_detail.assign(_sheet="Detail"),
                filt_totals.assign(_sheet="Totals")
            ])),
            mime="text/csv"
        )
    with c2:
        xlsx_bytes = to_xlsx_bytes({
            "Detail": filt_detail,
            "Totals": filt_totals,
            "Pivot": pivot
        })
        st.download_button(
            "Download Excel (Detail/Totals/Pivot)",
            file_name="greeno_big_three_v1_5_5.xlsx",
            data=xlsx_bytes,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Upload one or more PDF files to begin.")
