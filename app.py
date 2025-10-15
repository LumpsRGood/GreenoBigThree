import streamlit as st
import pandas as pd
import pdfplumber
import io
import re

# --- Core PDF Extraction Function ---
def extract_pdf_tables_pdfplumber(uploaded_file):
    """Reads PDF data using pdfplumber and combines all extracted tables."""
    # Read the uploaded file into memory as bytes
    pdf_bytes = uploaded_file.read()
    all_data = []

    # These settings are optimized for finding tables without visible borders
    # (like the one in your screenshot) based on white space and text alignment.
    table_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "text_tolerance": 5,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
    }

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            st.info(f"Processing {len(pdf.pages)} pages...")
            
            for i, page in enumerate(pdf.pages):
                # Extract tables using custom settings
                tables = page.extract_tables(table_settings=table_settings)
                
                for table_data in tables:
                    if table_data and len(table_data[0]) > 1: # Ensure table has at least one row and multiple columns
                        df = pd.DataFrame(table_data)
                        df['Source_Page'] = i + 1 # Add page number for reference
                        all_data.append(df)

    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

    if not all_data:
        return pd.DataFrame({'Status': ['No tables were successfully extracted.']})
    
    # Concatenate all tables into one raw DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# --- Data Cleaning and Aggregation Function ---
def clean_and_aggregate_data(raw_df):
    """Locates target columns robustly and performs final calculations."""
    
    # Clean up column values by stripping whitespace and converting to string
    raw_df = raw_df.applymap(lambda x: str(x).strip() if x is not None else None)
    
    # --- Robust Column Index Mapping (AVOIDS 'out-of-bounds' error) ---
    p9_col_index = None
    reason_col_index = None

    # Iterate through columns to find the correct indices based on header text
    for col in raw_df.columns:
        # Check for 'P9 25' or 'Total' headers in the first few rows (handles multi-row headers)
        header_text = "".join(raw_df[col].iloc[0:3].dropna().astype(str))
        
        # Identify the P9 25 column (must be numeric for aggregation)
        if re.search(r'P9\s*25|TOTAL', header_text, re.IGNORECASE) and raw_df[col].astype(str).str.isnumeric().any():
            if p9_col_index is None:
                p9_col_index = col
        
        # Identify the Reason for Contact column (typically the first non-numeric column)
        if reason_col_index is None and any('Reason' in s for s in header_text) and not raw_df[col].astype(str).str.isnumeric().any():
            reason_col_index = col
        
        # Fallback for the first column (likely 'Reason for Contact')
        if reason_col_index is None and col == 0:
            reason_col_index = col

    if p9_col_index is None or reason_col_index is None:
        st.error("Could not reliably locate 'Reason' or 'P9 25' columns. Check raw data.")
        return None, 0

    # 1. Slice and Rename
    processed_df = raw_df[[reason_col_index, p9_col_index]].copy()
    processed_df.columns = ['Reason', 'P9_25_Count']
    
    # 2. Clean Data (Remove header rows and convert to numeric)
    
    # Remove rows that are clearly headers or blank (assuming headers are in the first few rows)
    processed_df = processed_df[~processed_df['Reason'].str.contains(r'Reason|Cold Food|Delivery Total', na=False)].copy()

    # Convert counts to numeric (coerces non-numeric entries like headers to NaN)
    processed_df['P9_25_Count'] = pd.to_numeric(processed_df['P9_25_Count'], errors='coerce')
    
    # Final cleanup of blank rows and rows with NaN counts
    processed_df.dropna(subset=['Reason', 'P9_25_Count'], inplace=True)
    
    # 3. Calculate total
    total = processed_df['P9_25_Count'].sum()
    
    return processed_df, total

# --- Streamlit Application ---
st.set_page_config(page_title="PDF Table Extractor", layout="wide")
st.title("📄 Free PDF Table Extractor (for Streamlit Cloud)")
st.markdown("Upload your PDF to extract metrics from tables and calculate the total from the target column.")

uploaded_file = st.file_uploader("Upload your PDF file (max ~200MB):", type="pdf")

if uploaded_file:
    if st.button("Extract and Analyze P9 25 Data"):
        
        # --- Extraction ---
        raw_df = extract_pdf_tables_pdfplumber(uploaded_file)
        
        if raw_df is not None:
            st.subheader("Raw Extracted Data (for debugging)")
            st.dataframe(raw_df.head(10))

            # --- Cleaning and Aggregation ---
            with st.spinner('Cleaning data and calculating totals...'):
                processed_df, total = clean_and_aggregate_data(raw_df)
            
            if processed_df is not None:
                st.success(f"✅ Extraction Complete! Total Occurrences in P9 25: **{int(total)}**")
                
                # Filter for metrics with positive counts
                metrics_df = processed_df[processed_df['P9_25_Count'] > 0].sort_values(by='P9_25_Count', ascending=False)
                
                st.subheader("Results: Sorted Metrics from P9 25 Column")
                st.dataframe(metrics_df, use_container_width=True)

                # --- Download Button ---
                csv = metrics_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Processed Metrics as CSV",
                    data=csv,
                    file_name='p9_25_metrics_report.csv',
                    mime='text/csv',
                    key='download_csv'
                )
