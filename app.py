# app.py — Greeno Big Three v1.5.4 (strict labels + robust column bins + drill-down)
# Run: streamlit run app.py

import io
import re
import unicodedata
from collections import defaultdict, OrderedDict

import pandas as pd
import streamlit as st
import pdfplumber

st.set_page_config(page_title="Greeno Big Three v1.5.4", layout="wide")
st.title("Greeno Big Three v1.5.4")

# =========================
# Utilities & Normalizers
# =========================
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # Normalize common PDF quirks
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

# Locked reasons (7 only)
ALLOWED_REASONS = {
    _norm("Missing Item (Food)"):       "Missing food",
    _norm("Order Wrong"):               "Order wrong",
    _norm("Missing Condiments"):        "Missing condiments",
    _norm("Out Of Menu Item"):          "Out of menu item",
    _norm("Missing Item (Bev)"):        "Missing bev",
    _norm("Missing Ingredient (Food)"): "Missing ingredients",
    _norm("Packaging To Go Complaint"): "Packaging to-go complaint",
}

SECTION_PATTERNS = [r"\bto\s*-?\s*go\b", r"\bdelivery\b"]  # To Go, To-Go, ToGo; Delivery

PERIOD_PATTERN = re.compile(r"\bP\s*\d{1,2}\s*\d{2}\b")     # e.g., P7 25, P10 24


# =========================
# PDF Parsing Helpers
# =========================
def find_sections(words):
    """
    words: list from page.extract_words(use_text_flow=True)
    Return sections as [{name, y0, y1}], where name in {'to go', 'delivery'}
    """
    headers = []
    for w in words:
        txt = _norm(w.get("text", ""))
        for pat in SECTION_PATTERNS:
            if re.search(pat, txt):
                sec_name = "to go" if re.search(r"\bto\s*-?\s*go\b", txt) else "delivery"
                headers.append({"name": sec_name, "y0": w["top"]})
                break
    headers.sort(key=lambda h: h["y0"])
    if not headers:
        return []

    sections = []
    for i, h in enumerate(headers):
        y1 = words[-1]["bottom"] if i == len(headers) - 1 else headers[i + 1]["y0"] - 1
        sections.append({"name": h["name"], "y0": h["y0"], "y1": y1})
    return sections

def group_lines_by_y(words):
    """
    Rebuild text lines in reading order from word boxes.
    Returns: [(y, line_text, [word_boxes])]
    """
    lines = {}
    for w in words:
        y = round(w["top"], 1)  # gentle grouping
        lines.setdefault(y, []).append(w)

    rows = []
    for y in sorted(lines.keys()):
        row = sorted(lines[y], key=lambda ww: ww["x0"])
        text = " ".join([r["text"] for r in row])
        rows.append((y, text, row))
    return rows

def merge_wrapped_labels(rows):
    """
    rows: list of (y, text, boxes). Merge common wraps for 'Packaging To Go' + 'Complaint'
    Returns same structure, possibly combined.
    """
    out = []
    i = 0
    while i < len(rows):
        y, text, boxes = rows[i]
        t_norm = _norm(text)

        if i + 1 < len(rows):
            y2, text2, boxes2 = rows[i + 1]
            if _norm(text) in (_norm("Packaging To Go"), _norm("Packaging To-Go")) and _norm(text2) == _norm("Complaint"):
                merged_text = f"{text} {text2}"
                merged_boxes = boxes + boxes2
                out.append((y, merged_text, merged_boxes))
                i += 2
                continue
        out.append((y, text, boxes))
        i += 1
    return out

def build_period_bins(header_boxes):
    """
    From a line's word boxes, identify columns with P# YY and return ordered bins: [{label, x0, x1}]
    """
    bins = []
    for b in header_boxes:
        t = b.get("text", "")
        if PERIOD_PATTERN.search(t):
            lbl = _norm(t)
            bins.append({"label": lbl, "x0": b["x0"], "x1": b["x1"]})
    bins.sort(key=lambda k: k["x0"])
    # Merge overlapping bins (some PDFs split P7 and 25 into separate boxes)
    merged = []
    for b in bins:
        if not merged:
            merged.append(b)
        else:
            last = merged[-1]
            # if close/overlap, extend
            if b["x0"] - last["x1"] < 6:  # small gap threshold
                last["x1"] = max(last["x1"], b["x1"])
                last["label"] = last["label"]  # keep first label (already normalized)
            else:
                merged.append(b)
    return merged

def nearest_bin(cx, bins):
    if not bins:
        return None
    idx = min(range(len(bins)), key=lambda k: abs(cx - (bins[k]["x0"] + bins[k]["x1"]) / 2))
    return idx

def extract_reason_rows(page, section):
    """
    Extract rows of the 7 reasons within a section y-range and attach numbers to period bins.
    Returns list of dicts: {section, reason, period_values{period_label: int}}
    """
    words = [
        w for w in page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if section["y0"] <= w["top"] <= section["y1"]
    ]
    if not words:
        return []

    rows = group_lines_by_y(words)
    rows = merge_wrapped_labels(rows)

    # Find a header line with period codes OR fall back to first numeric-rich line
    header_idx = None
    for i, (_, text, _) in enumerate(rows):
        if PERIOD_PATTERN.search(text):
            header_idx = i
            break
    if header_idx is None:
        for i, (_, text, _) in enumerate(rows):
            if len(re.findall(r"\b\d[\d,]*\b", text)) >= 3:
                header_idx = i
                break
    if header_idx is None:
        header_idx = 0

    period_bins = build_period_bins(rows[header_idx][2])  # use boxes of header line

    parsed = []
    for y, text, boxes in rows:
        # Strip obvious total suffix if present in same line (we ignore totals line by rule)
        t = re.sub(r"\s+total.*$", "", text, flags=re.IGNORECASE)
        key = _norm(t)
        for allowed_key, out_label in ALLOWED_REASONS.items():
            # start-with helps when PDFs jam numbers into the same line after label
            if key.startswith(allowed_key):
                numeric_boxes = [b for b in boxes if _only_digits(b.get("text", ""))]
                row = {"section": section["name"], "reason": out_label, "period_values": {}}
                for nb in numeric_boxes:
                    cx = (nb["x0"] + nb["x1"]) / 2
                    j = nearest_bin(cx, period_bins)
                    if j is not None:
                        lbl = period_bins[j]["label"]
                        row["period_values"].setdefault(lbl, 0)
                        row["period_values"][lbl] += _clean_int(nb["text"])
                parsed.append(row)
                break
    return parsed

def detect_store_and_ad(lines_before_idx):
    """
    Very simple heuristic:
    - Store line typically contains a 4-digit store number.
    - AD name often appears above stores, without numbers.
    We scan upward lines to find nearest store, then the nearest AD above it.
    """
    store = None
    ad = None
    # Find nearest store-like token going upward
    for y, text, _ in reversed(lines_before_idx):
        m = re.search(r"\b(\d{4})\b", text)
        if m:
            store = m.group(1)
            break
    # Find nearest non-numeric, likely AD, above that
    if store is not None:
        for y, text, _ in reversed(lines_before_idx):
            if store in text:
                # start looking above the store line
                continue
            # name-ish: letters/spaces, not a period code, not just numbers
            t = text.strip()
            if t and not re.search(r"\d", t) and len(t) <= 40:
                ad = t
                break
    return ad, store

def extract_page_data(page):
    """
    For a page, extract rows for To Go / Delivery sections and attach AD/Store by scanning above each reason line.
    Return list of dicts: {ad, store, section, reason, period_values{period: count}}
    """
    all_words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not all_words:
        return []

    sections = find_sections(all_words)
    if not sections:
        return []

    # Precompute lines for store/AD detection
    all_rows = group_lines_by_y(all_words)

    page_rows = []
    for sec in sections:
        if sec["name"] not in ("to go", "delivery"):
            continue
        sec_rows = extract_reason_rows(page, sec)

        # For each matched line, re-scan upwards from that y to get AD/Store
        for r in sec_rows:
            # find the row index with the same y (approx)
            y_candidates = [i for i, (yy, _, _) in enumerate(all_rows)]
            # We'll attach nearest line above the section header for AD/Store context
            # Use section y0 as anchor; everything above can be scanned
            idx_anchor = max([i for i, (yy, _, _) in enumerate(all_rows) if yy <= sec["y0"]], default=0)
            ad, store = detect_store_and_ad(all_rows[:idx_anchor + 1])
            r["ad"] = ad or "Unknown AD"
            r["store"] = store or "Unknown Store"
            page_rows.append(r)

    return page_rows


# =========================
# Aggregation
# =========================
def aggregate_results(records):
    """
    records: list of {ad, store, section, reason, period_values{period: count}}
    Returns:
      detail_df: each row per AD/Store/Section/Reason/Period
      totals_df: totals aggregated at AD & Store levels for the 7 reasons
    """
    detail_rows = []
    for rec in records:
        ad = rec["ad"]
        store = rec["store"]
        section = rec["section"]
        reason = rec["reason"]
        for period, val in rec["period_values"].items():
            detail_rows.append({
                "Area Director": ad,
                "Store": store,
                "Section": "To Go" if section == "to go" else "Delivery",
                "Reason": reason,
                "Period": period.upper().replace(" ", ""),  # P7 25 -> p7 25 normalized; display later
                "Count": val
            })
    if not detail_rows:
        return pd.DataFrame(), pd.DataFrame()

    detail_df = pd.DataFrame(detail_rows)

    # Create pivot totals per AD/Store/Reason/Period (Section already implied in Reason by business rule; we keep Section too)
    totals_df = (detail_df
                 .groupby(["Area Director", "Store", "Reason", "Period"], as_index=False)["Count"]
                 .sum())

    return detail_df, totals_df


# =========================
# UI
# =========================
st.caption("Parses IHOP PDF(s), extracts ONLY **To Go** and **Delivery** for the 7 locked reasons, grouped by **AD → Store → Period**.")

uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_records = []
    debug_pages = []
    with st.spinner("Parsing PDFs…"):
        for f in uploaded_files:
            try:
                with pdfplumber.open(io.BytesIO(f.read())) as pdf:
                    for pageno, page in enumerate(pdf.pages, start=1):
                        page_recs = extract_page_data(page)
                        if not page_recs:
                            debug_pages.append((f.name, pageno, page))
                        all_records.extend(page_recs)
            except Exception as e:
                st.error(f"Error opening {f.name}: {e}")

    if not all_records:
        st.error("Parser fallback engaged. No matching To Go/Delivery reasons found. Expand debug below to inspect raw text.")
        with st.expander("Debug: Show raw text by page"):
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
        st.warning("No aggregated rows after parsing. Check PDF formatting or open the debug expander above.")
        st.stop()

    # Period selection (from discovered periods)
    all_periods = sorted(set(totals_df["Period"].tolist()))
    sel_periods = st.multiselect("Select Period(s)", options=all_periods, default=all_periods)

    filt_totals = totals_df[totals_df["Period"].isin(sel_periods)].copy()
    filt_detail = detail_df[detail_df["Period"].isin(sel_periods)].copy()

    # Wide layout: left summary, right drill-down
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Totals by Area Director → Store → Reason")
        # Build a pivot friendly view
        pivot = (filt_totals
                 .pivot_table(index=["Area Director", "Store", "Reason"],
                              columns="Period",
                              values="Count",
                              aggfunc="sum",
                              fill_value=0)
                 .reset_index())
        st.dataframe(pivot, use_container_width=True)

        # Export buttons
        def to_csv_bytes(df):
            return df.to_csv(index=False).encode("utf-8")

        def to_xlsx_bytes(dfs: dict):
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
                for sheet, frame in dfs.items():
                    frame.to_excel(writer, index=False, sheet_name=sheet[:31])
            bio.seek(0)
            return bio.read()

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                label="Download CSV (Detail & Totals)",
                file_name="greeno_big_three_v1_5_4.csv",
                data=to_csv_bytes(pd.concat([
                    filt_detail.assign(_sheet="Detail"),
                    filt_totals.assign(_sheet="Totals")
                ])),
                mime="text/csv"
            )
        with exp_col2:
            xlsx_bytes = to_xlsx_bytes({
                "Detail": filt_detail,
                "Totals": filt_totals,
                "Pivot": pivot
            })
            st.download_button(
                label="Download Excel (Detail/Totals/Pivot)",
                file_name="greeno_big_three_v1_5_4.xlsx",
                data=xlsx_bytes,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with col2:
        st.subheader("Drill-down")
        ads = ["(All)"] + sorted(filt_detail["Area Director"].dropna().unique().tolist())
        stores = ["(All)"] + sorted(filt_detail["Store"].dropna().unique().tolist())
        reasons = ["(All)"] + list(OrderedDict.fromkeys([  # keep display order stable
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
            st.info("No matching rows for this selection.")
        else:
            st.dataframe(dd.sort_values(["Area Director", "Store", "Reason", "Period"]),
                         use_container_width=True)

    # Optional: raw debug view (collapsed)
    with st.expander("Advanced Debug (raw parsed rows)"):
        st.write(pd.DataFrame(all_records))

else:
    st.info("Upload one or more PDF files to begin.")
