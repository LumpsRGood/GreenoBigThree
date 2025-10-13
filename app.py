# Greeno Big Three v1.8.2 — simplified keyword triggers
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from collections import defaultdict
import re
from io import BytesIO

st.set_page_config(page_title="Greeno Big Three v1.8.2", layout="wide")
st.title("Greeno Big Three v1.8.2")

# --- Sidebar upload ---
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload the NGC report PDF", type=["pdf"])

# --- Helper functions ---
def _lc(x): return x.lower() if isinstance(x, str) else ""

def extract_text_blocks(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    data = []
    for page in doc:
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            data.append((page.number + 1, text))
    return data

# --- New simplified keyword triggers ---
KEYWORD_TRIGGERS = {
    # TO-GO MISSING COMPLAINTS
    "Missing food": ["missing food"],
    "Order wrong": ["order wrong"],
    "Missing condiments": ["condiments"],
    "Out of menu item": ["out of menu"],
    "Missing bev": ["missing bev"],
    "Missing ingredients": ["ingredient"],
    "Packaging to-go complaint": ["packaging"],

    # ATTITUDE
    "Unprofessional/Unfriendly": ["unfriendly"],
    "Manager directly involved": ["involved"],
    "Manager not available": ["manager not available"],
    "Manager did not visit": ["visit"],
    "Negative mgr-employee exchange": ["exchange"],
    "Manager did not follow up": ["follow up"],
    "Argued with guest": ["argued"],

    # OTHER
    "Long hold/no answer": ["hold"],
    "No/insufficient compensation offered": ["compensation"],
    "Did not attempt to resolve": ["resolve"],
    "Guest left without ordering": ["without ordering"],
    "Unknowledgeable": ["unknowledgeable"],
    "Did not open on time": ["open on time"],
    "No/poor apology": ["apology"],
}

# --- Simple match helper ---
def _matches_keyword(line_text: str) -> str | None:
    text = _lc(line_text)
    for canon, triggers in KEYWORD_TRIGGERS.items():
        for trig in triggers:
            if trig in text:
                return canon
    return None

# --- Parse PDF into lines grouped by AD, store, and section ---
def parse_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    current_ad, current_store, current_section = None, None, None
    rows = []

    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # sort by y,x position

        for b in blocks:
            text = b[4].strip()
            if not text:
                continue

            # Detect Area Director
            if re.match(r"^[A-Z][a-z]+\s[A-Z][a-z]+$", text) and "IHOP" not in text:
                current_ad = text
                continue

            # Detect store pattern
            if re.match(r"^\d{4}\s-\s", text):
                current_store = text
                continue

            # Detect segment section
            if text in ["Dine-In", "To Go", "Delivery"]:
                current_section = text
                continue

            # Ignore total lines
            if "Total:" in text or text.startswith("—"):
                continue

            # Look for metric triggers
            match = _matches_keyword(text)
            if match:
                rows.append({
                    "page": page.number + 1,
                    "ad": current_ad,
                    "store": current_store,
                    "section": current_section,
                    "metric": match,
                    "text": text
                })

    df = pd.DataFrame(rows)
    return df

# --- Totals ---
def summarize(df):
    summary = (
        df.groupby("metric")
        .size()
        .reset_index(name="count")
        .sort_values("metric")
    )
    total = pd.DataFrame({"metric": ["Category Grand Total"], "count": [summary["count"].sum()]})
    summary = pd.concat([summary, total], ignore_index=True)
    return summary

# --- UI Logic ---
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    with st.spinner("Roll Tide… Parsing PDF"):
        df = parse_pdf(pdf_bytes)

    if df.empty:
        st.error("No matching metrics found.")
    else:
        st.success("PDF parsed successfully!")

        tabs = st.tabs(["To-go Missing Complaints", "Attitude", "Other"])

        # Category: To-go Missing Complaints
        with tabs[0]:
            subset = df[df["metric"].isin([
                "Missing food", "Order wrong", "Missing condiments",
                "Out of menu item", "Missing bev", "Missing ingredients",
                "Packaging to-go complaint"
            ])]
            if not subset.empty:
                st.subheader("4️⃣ Reason totals — To-go Missing Complaints (selected period)")
                st.dataframe(summarize(subset), use_container_width=True)
            else:
                st.info("No matching 'To-go Missing Complaints' reasons found.")

        # Category: Attitude
        with tabs[1]:
            subset = df[df["metric"].isin([
                "Unprofessional/Unfriendly", "Manager directly involved",
                "Manager not available", "Manager did not visit",
                "Negative mgr-employee exchange", "Manager did not follow up",
                "Argued with guest"
            ])]
            if not subset.empty:
                st.subheader("5️⃣ Reason totals — Attitude (selected period)")
                st.dataframe(summarize(subset), use_container_width=True)
            else:
                st.info("No matching 'Attitude' reasons found.")

        # Category: Other
        with tabs[2]:
            subset = df[df["metric"].isin([
                "Long hold/no answer", "No/insufficient compensation offered",
                "Did not attempt to resolve", "Guest left without ordering",
                "Unknowledgeable", "Did not open on time", "No/poor apology"
            ])]
            if not subset.empty:
                st.subheader("6️⃣ Reason totals — Other (selected period)")
                st.dataframe(summarize(subset), use_container_width=True)
            else:
                st.info("No matching 'Other' reasons found.")

        # Optional download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download raw parsed data (CSV)", csv, "parsed_metrics.csv", "text/csv")
else:
    st.markdown("⬅️ Upload a PDF to get started.")
