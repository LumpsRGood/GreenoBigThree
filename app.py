import streamlit as st
import pandas as pd
import pdfplumber
import io
import re

# --- 1. Configuration ---
# Define the exact metric names you want to filter for (based on your screenshot)
TARGET_METRICS = [
    "Cold Food",
    "Long Hold/No Answer/Hung Up",
    "Missing Condiments",
    "Missing Item (Food)",
    "Tough Food",
    "Unprofessional Behavior"
]

# --- 2. Core PDF Extraction Function ---
def extract_pdf_tables_pdfplumber(uploaded_file):
    """Reads PDF data using pdfplumber and combines all extracted tables."""
    pdf_bytes = uploaded_file.read()
    all_data = []

    # Settings optimized for finding tables without visible borders
    table_settings = {
        "vertical_strategy": "text",   # Use text boundaries to determine vertical lines
        "horizontal_strategy": "lines",# Use line boundaries to determine horizontal lines
        "snap_tolerance": 3,
        "text_tolerance": 5,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
    }

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            st.info(f"Processing {len(pdf.pages)} pages...")
            
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables(table_settings=table_settings)
                
                for table_data in tables:
                    if table_data and len(table_data[0]) > 1:
                        df = pd.DataFrame(table_data)
                        df['Source_Page'] = i + 1
                        all_data.append(df)

    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

    if not all_data:
        return pd.DataFrame({'Status': ['No tables were successfully extracted.']})
    
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# --- 3. Data Cleaning and Aggregation Function ---
def clean_and_aggregate_data(raw_df):
    """Locates target columns robustly, filters by exact metric, and performs final calculations."""
    
    # Clean up column values by stripping whitespace and converting to string
    raw_df = raw_df.applymap(lambda x: str(x).strip() if x is not None else None)
    
    # --- Robust Column Index Mapping ---
    p9_col_index = None
    reason_col_index = None

    for col in raw_df.columns:
        # Check the first few rows for header text
        header_text = "".join(raw_df[col].iloc[0:3].dropna().astype(str))
        
        # Find P9 25 column (by name and ensuring it contains numbers)
        if re.search(r'P9\s*25', header_text, re.IGNORECASE) and raw_df[col].astype(str).str.isnumeric().any():
            if p9_col_index is None:
                p9_col_index = col
        
        # Find Reason for Contact column (by name)
        if reason_col_index is None and any('Reason' in s for s in header_text):
            reason_col_index = col
        
        # Fallback: Assume Reason for Contact is the first column if not explicitly found
        if reason_col_index is None and col == 0:
            reason_col_index = col

    if p9_col_index is None or reason_col_index is None:
        st.error("Could not reliably locate 'Reason' or 'P9 25' columns. Check raw output.")
        return None, 0

    # 1. Slice and Rename
    processed_df = raw_df[[reason_col_index, p9_col_index]].copy()
    processed_df.columns = ['Reason', 'P9_25_Count']
    
    # 2. Convert counts to numeric
    processed_df['P9_25_Count'] = pd.to_numeric(processed_df['P9_25_Count'], errors='coerce')
    
    # 3. CRITICAL FILTERING STEP: Filter by exact metric wording
    # This eliminates all scattered and summary rows like "Delivery Total"
    filtered_df = processed_df[
        processed_df['Reason'].isin(TARGET_METRICS)
    ].copy()

    # 4. Final cleanup
    filtered_df['P9_25_Count'] = filtered_df['P9_25_Count'].fillna(0).astype(int)
    
    # 5. Group by Reason (in case a single metric appears on multiple pages)
    final_metrics_df = filtered_df.groupby('Reason')['P9_25_Count'].sum().reset_index()
    final_metrics_df.rename(columns={'P9_25_Count': 'P9_25_Occurrences'}, inplace=True)
    
    # Calculate total
    total = final_metrics_df['P9_25_Occurrences'].sum()
    
    return final_metrics_df, total

# --- 4. Styling Function for Conditional Formatting ---
def style_metrics(df):
    """Applies conditional formatting/styling to highlight non-zero counts."""
    
    # Highlight rows where occurrences are greater than 0
    def highlight_positive(s):
        is_pos = s['P9_25_Occurrences'] > 0
        return ['background-color: #e0f2f1' if is_pos else '' for v in s] # Light teal/blue

    # Apply the styling and set general table properties
    styled_df = df.style.apply(
        highlight_positive, 
        axis=1
    ).set_properties(**{'text-align': 'left'}).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4DB6AC'), ('color', 'white')]}
    ])

    return styled_df

# =========================================================
# --- Streamlit Application ---
# =========================================================

st.set_page_config(page_title="PDF Table Extractor", layout="wide")
st.title("📄 Free PDF Table Extractor (for Streamlit Cloud)")
st.markdown("Upload your PDF to **filter by specific metrics** and calculate the total occurrences in the **P9 25** column.")

uploaded_file = st.file_uploader("Upload your PDF file:", type="pdf")

if uploaded_file:
    if st.button("Extract and Analyze P9 25 Data"):
        
        # --- Extraction ---
        raw_df = extract_pdf_tables_pdfplumber(uploaded_file)
        
        if raw_df is not None and not raw_df.empty:
            st.subheader("Raw Extracted Data Preview (for debugging)")
            st.dataframe(raw_df.head(10))

            # --- Cleaning and Aggregation ---
            with st.spinner('Cleaning data and calculating totals...'):
                processed_df, total = clean_and_aggregate_data(raw_df)
            
            if processed_df is not None:
                st.success(f"✅ Extraction Complete! Total Occurrences in P9 25: **{int(total)}**")
                
                # Sort the metrics DataFrame before displaying
                sorted_metrics_df = processed_df.sort_values(by='P9_25_Occurrences', ascending=False)
                
                st.subheader("Results: Filtered & Styled Metrics")
                
                # Display the styled DataFrame
                styled_table = style_metrics(sorted_metrics_df)
                st.dataframe(styled_table, use_container_width=True)

                # --- Download Button ---
                csv = sorted_metrics_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Processed Metrics as CSV",
                    data=csv,
                    file_name='p9_25_metrics_report.csv',
                    mime='text/csv',
                    key='download_csv'
                )
