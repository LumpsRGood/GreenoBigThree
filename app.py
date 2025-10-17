# path: app.py
# Streamlit app: PDF → CSV with regex-based parsing and repeatable "recipe" configs.

from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional imports guarded at runtime (why: avoid hard fail if not installed)
try:
    import pdfplumber  # type: ignore
except Exception as e:
    st.error("Missing dependency: pdfplumber. Install with `pip install pdfplumber`.")
    raise

# OCR is optional
try:
    import pytesseract  # type: ignore
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

# ---------- Config & Models ----------

@dataclass
class ParserRecipe:
    name: str
    regex: str
    flags_ignorecase: bool = False
    flags_multiline: bool = False
    flags_dotall: bool = True
    collapse_whitespace: bool = True
    normalize_unicode: bool = True
    remove_page_numbers: bool = True
    remove_empty_lines: bool = True
    drop_header_lines: int = 0
    drop_footer_lines: int = 0
    hyphenation_fix: bool = False
    use_ocr: bool = False
    date_columns: List[str] = None
    numeric_columns: List[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "ParserRecipe":
        data = json.loads(s)
        # Backward-safe defaults
        data.setdefault("flags_dotall", True)
        data.setdefault("collapse_whitespace", True)
        data.setdefault("normalize_unicode", True)
        data.setdefault("remove_page_numbers", True)
        data.setdefault("remove_empty_lines", True)
        data.setdefault("drop_header_lines", 0)
        data.setdefault("drop_footer_lines", 0)
        data.setdefault("hyphenation_fix", False)
        data.setdefault("use_ocr", False)
        data.setdefault("date_columns", [])
        data.setdefault("numeric_columns", [])
        return ParserRecipe(**data)


DEFAULT_REGEX = (
    r"(?s)"                              # DOTALL by default
    r"Item:\s*(?P<item>.*?)\n"           # example multi-line block
    r"ID:\s*(?P<id>[A-Za-z0-9\-]+)\n"
    r"Qty:\s*(?P<qty>\d+(?:\.\d+)?)\b.*?"
    r"(?:Price:\s*(?P<price>\d+(?:\.\d+)?))?"
)

# ---------- Text Extraction & Cleaning ----------

def extract_pdf_text(file: io.BytesIO, use_ocr: bool) -> Tuple[str, List[str]]:
    """
    Returns (full_text, page_texts). Falls back to non-OCR if OCR unavailable.
    """
    page_texts: List[str] = []

    if use_ocr and not _HAS_TESSERACT:
        st.warning("OCR selected but pytesseract not available. Falling back to pdfplumber text.")
        use_ocr = False

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if use_ocr:
                try:
                    # Render page to raster and OCR (why: image-only PDFs)
                    pil_img = page.to_image(resolution=300).original
                    txt = pytesseract.image_to_string(pil_img)  # type: ignore
                except Exception as ocr_err:
                    st.warning(f"OCR failed on a page; using text extract instead. ({ocr_err})")
                    txt = page.extract_text(layout=True) or ""
            else:
                txt = page.extract_text(layout=True) or ""
            page_texts.append(txt)

    full_text = "\n<<<PAGE_BREAK>>>\n".join(page_texts)
    return full_text, page_texts


def normalize_text(s: str, normalize_unicode: bool, collapse_whitespace: bool,
                   remove_empty_lines: bool, hyphenation_fix: bool) -> str:
    if normalize_unicode:
        s = unicodedata.normalize("NFKC", s)

    if hyphenation_fix:
        # Join words broken across line breaks like "exam-\nple" → "example"
        s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)

    # Unify line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    if collapse_whitespace:
        # Preserve newlines but collapse runs of spaces/tabs
        s = "\n".join(re.sub(r"[ \t]+", " ", line) for line in s.split("\n"))

    if remove_empty_lines:
        s = "\n".join([ln for ln in s.split("\n") if ln.strip() != ""])

    return s


def strip_headers_footers(text: str, drop_header_lines: int, drop_footer_lines: int) -> str:
    chunks = text.split("\n<<<PAGE_BREAK>>>\n")
    new_chunks = []
    for chunk in chunks:
        lines = chunk.split("\n")
        if drop_header_lines > 0:
            lines = lines[drop_header_lines:]
        if drop_footer_lines > 0 and drop_footer_lines < len(lines):
            lines = lines[:-drop_footer_lines]
        new_chunks.append("\n".join(lines))
    return "\n<<<PAGE_BREAK>>>\n".join(new_chunks)


def remove_page_numbers(text: str) -> str:
    # Drop common standalone page number lines
    lines = text.split("\n")
    filtered = []
    for ln in lines:
        if re.fullmatch(r"Page\s+\d+(?:\s*/\s*\d+)?", ln.strip(), flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{1,4}", ln.strip()):
            continue
        filtered.append(ln)
    return "\n".join(filtered)

# ---------- Parsing ----------

def compile_pattern(pattern: str, ignorecase: bool, multiline: bool, dotall: bool) -> re.Pattern:
    flags = 0
    if ignorecase:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    if dotall:
        flags |= re.DOTALL
    return re.compile(pattern, flags)


def parse_records(text: str, pattern: re.Pattern) -> pd.DataFrame:
    matches = list(pattern.finditer(text))
    rows: List[Dict[str, Any]] = []
    for m in matches:
        gd = m.groupdict()
        # Strip whitespace for each captured group
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in gd.items()})
    return pd.DataFrame(rows)


def coerce_columns(df: pd.DataFrame, numeric_cols: List[str], date_cols: List[str]) -> pd.DataFrame:
    df_out = df.copy()
    for col in numeric_cols:
        if col in df_out.columns:
            df_out[col] = (
                df_out[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
            )
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    for col in date_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_datetime(df_out[col], errors="coerce", infer_datetime_format=True)
    return df_out

# ---------- UI Helpers ----------

def download_button_for_df(df: pd.DataFrame, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def default_recipe() -> ParserRecipe:
    return ParserRecipe(
        name="My PDF Parser",
        regex=DEFAULT_REGEX,
        flags_ignorecase=False,
        flags_multiline=False,
        flags_dotall=True,
        collapse_whitespace=True,
        normalize_unicode=True,
        remove_page_numbers=True,
        remove_empty_lines=True,
        drop_header_lines=0,
        drop_footer_lines=0,
        hyphenation_fix=False,
        use_ocr=False,
        date_columns=[],
        numeric_columns=["qty", "price"],
    )

# ---------- Streamlit App ----------

st.set_page_config(page_title="PDF → CSV (Regex Parser)", page_icon="📄", layout="wide")

st.title("📄→📊 PDF to CSV (Regex, reproducible ‘recipe’)")

with st.sidebar:
    st.header("⚙️ Options")
    if "recipe" not in st.session_state:
        st.session_state["recipe"] = default_recipe()

    # Recipe load
    load_recipe_file = st.file_uploader("Load Recipe (JSON)", type=["json"], key="recipe_loader")
    if load_recipe_file:
        try:
            recipe_text = load_recipe_file.read().decode("utf-8")
            st.session_state["recipe"] = ParserRecipe.from_json(recipe_text)
            st.success("Recipe loaded.")
        except Exception as e:
            st.error(f"Failed to load recipe: {e}")

    recipe: ParserRecipe = st.session_state["recipe"]

    recipe.name = st.text_input("Recipe Name", value=recipe.name)
    recipe.use_ocr = st.toggle("Use OCR (pytesseract)", value=recipe.use_ocr,
                               help="Enable if your PDF is image-based. Requires Tesseract installed.")

    st.subheader("Cleaning")
    recipe.normalize_unicode = st.checkbox("Normalize Unicode", value=recipe.normalize_unicode)
    recipe.collapse_whitespace = st.checkbox("Collapse extra spaces", value=recipe.collapse_whitespace)
    recipe.remove_empty_lines = st.checkbox("Remove empty lines", value=recipe.remove_empty_lines)
    recipe.hyphenation_fix = st.checkbox("Fix hyphenation across line breaks (ex-periment → experiment)",
                                         value=recipe.hyphenation_fix)

    recipe.drop_header_lines = st.number_input("Drop header lines (per page)", min_value=0, max_value=50,
                                               value=recipe.drop_header_lines, step=1)
    recipe.drop_footer_lines = st.number_input("Drop footer lines (per page)", min_value=0, max_value=50,
                                               value=recipe.drop_footer_lines, step=1)
    recipe.remove_page_numbers = st.checkbox("Remove page number lines", value=recipe.remove_page_numbers)

    st.subheader("Regex Flags")
    recipe.flags_dotall = st.checkbox("DOTALL ( . matches newlines )", value=recipe.flags_dotall)
    recipe.flags_multiline = st.checkbox("MULTILINE ( ^/$ match per line )", value=recipe.flags_multiline)
    recipe.flags_ignorecase = st.checkbox("IGNORECASE", value=recipe.flags_ignorecase)

    st.subheader("Type Coercion")
    numeric_cols_csv = st.text_input("Numeric columns (comma-separated)", value=",".join(recipe.numeric_columns or []))
    recipe.numeric_columns = [c.strip() for c in numeric_cols_csv.split(",") if c.strip()]

    date_cols_csv = st.text_input("Date columns (comma-separated)", value=",".join(recipe.date_columns or []))
    recipe.date_columns = [c.strip() for c in date_cols_csv.split(",") if c.strip()]

    # Save recipe
    recipe_json = recipe.to_json()
    st.download_button("💾 Download Recipe JSON", data=recipe_json.encode("utf-8"),
                       file_name=f"{recipe.name or 'recipe'}.json", mime="application/json", use_container_width=True)

st.markdown("**Step 1 — Upload PDF**")
pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])

if pdf_file is None:
    st.info("Upload a PDF to begin. Use the sidebar to tweak extraction & parsing.")
    st.stop()

# Read PDF bytes into BytesIO to allow re-reading
pdf_bytes = io.BytesIO(pdf_file.read())

with st.spinner("Extracting text..."):
    raw_text, _pages = extract_pdf_text(pdf_bytes, use_ocr=st.session_state["recipe"].use_ocr)

# Apply cleaning
cleaned = raw_text
cleaned = strip_headers_footers(cleaned, recipe.drop_header_lines, recipe.drop_footer_lines)
if recipe.remove_page_numbers:
    cleaned = remove_page_numbers(cleaned)
cleaned = normalize_text(
    cleaned,
    normalize_unicode=recipe.normalize_unicode,
    collapse_whitespace=recipe.collapse_whitespace,
    remove_empty_lines=recipe.remove_empty_lines,
    hyphenation_fix=recipe.hyphenation_fix,
)

# Preview text
st.markdown("**Step 2 — Preview Extracted/Cleaned Text**")
with st.expander("Show text preview", expanded=False):
    st.text_area("Extracted Text", value=cleaned[:20000], height=300)  # cap preview length for performance
    st.caption("Preview capped at ~20k chars for performance.")

# Parser config
st.markdown("**Step 3 — Define Regex with Named Groups**")
st.write(
    "Use a single regex that matches one entire record, with **named groups** for each column. "
    "Example groups: `(?P<id>...)`, `(?P<qty>...)`. The parser will run `finditer` to build rows."
)

regex_input = st.text_area("Regex (use named groups)", value=recipe.regex, height=160,
                           placeholder="(?s)Your pattern with (?P<field>...) groups")

apply_btn = st.button("Parse with Regex", type="primary", use_container_width=True)

if apply_btn:
    try:
        pat = compile_pattern(
            regex_input,
            ignorecase=recipe.flags_ignorecase,
            multiline=recipe.flags_multiline,
            dotall=recipe.flags_dotall,
        )
    except re.error as e:
        st.error(f"Regex compile error: {e}")
        st.stop()

    with st.spinner("Parsing records..."):
        try:
            df = parse_records(cleaned, pat)
        except Exception as e:
            st.error(f"Parsing failed: {e}")
            st.stop()

    if df.empty:
        st.warning("No matches. Tips: ensure DOTALL if records span lines; test smaller pieces; inspect text preview.")
        # Heuristic: show first 20 lines to help user eyeball anchors
        sample_lines = "\n".join(cleaned.splitlines()[:20])
        with st.expander("First 20 lines of cleaned text"):
            st.code(sample_lines)
        st.stop()

    # Coerce types
    df = coerce_columns(df, numeric_cols=recipe.numeric_columns or [], date_cols=recipe.date_columns or [])

    st.success(f"Parsed {len(df)} record(s).")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("**Step 4 — Download CSV**")
    suggested_name = (recipe.name or "parsed") + ".csv"
    download_button_for_df(df, filename=suggested_name)

    # Persist latest regex back into recipe
    st.session_state["recipe"].regex = regex_input

else:
    st.info("Enter your regex, then click **Parse with Regex**. Use named groups for CSV columns.")

# Footer help
with st.expander("🧪 Quick tips for resilient regex"):
    st.markdown(
        """
- Prefer **named groups**: `(?P<col>...)` → becomes a CSV column.
- Use `(?s)` or enable **DOTALL** when records span multiple lines.
- Anchor recurring labels: e.g., `Item:\\s*(?P<item>.*?)\\nID:\\s*(?P<id>\\S+)`.
- Make optional fields non-fatal: `(?:Label:\\s*(?P<opt>.*?))?`.
- If headers repeat per page, set **Drop header/footer lines** in the sidebar.
- Convert numbers/dates via **Type Coercion** in the sidebar.
"""
    )
