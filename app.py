import streamlit as st
import pandas as pd
import pdfplumber
import io
import re
# Assuming these imports were necessary and working previously (due to system packages)
import tabula # Used for extraction if pdfplumber fails, or as an alternative
import camelot # Used for extraction if pdfplumber fails, or as an alternative

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
def extract_pdf_tables(uploaded_file):
    """
    Reads PDF data using pdfplumber (recommended for cloud) and combines all extracted tables.
    You could also use tabula-py or camelot-py here if they are configured correctly.
    """
    pdf_bytes = uploaded_file.read()
    all_data = []

    # Settings optimized for finding tables without visible borders
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
            st.info(f"Processing {len(pdf.pages)} pages using pdfplumber...")
            
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables(table_settings=table_settings)
                
                for table_data in tables:
                    if table_data and len(table_data[0]) > 1:
                        df = pd.DataFrame(table_data)
                        df['Source_Page'] = i + 1
                        all_data.append(df)

    except Exception as e:
        st.error(f"PDF extraction error with pdfplumber: {e}")
        st.warning("Attempting to continue with available raw data...")
        return None # Return None if extraction completely fails.

    if not all_data:
        return pd.DataFrame({'Status': ['No tables were successfully extracted.']})
    
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# --- 3. Data Cleaning and Aggregation Function (Revised Loop) ---
def clean_and_aggregate_data(raw_df):
    
    # ... (Lines before the loop remain the same)
    raw_df = raw_df.applymap(lambda x: str(x).strip() if x is not None else None)
    
    # --- Robust Column Index Mapping ---
    p9_col_index = None
    reason_col_index = None

    HEADER_SCAN_ROWS = 5 
    
    # Iterate using the column index (i) and the column label (label)
    for i, label in enumerate(raw_df.columns):
        
        # Use the index 'i' for the comparison that caused the error
        if i > 15:
            break
            
        header_fragments = [
            str(raw_df[label].iloc[r]).strip() for r in range(min(HEADER_SCAN_ROWS, len(raw_df)))
        ]
        header_text = " ".join(filter(None, header_fragments)).upper()

        # Identify the P9 25 column index (use the label for data access, save the label/index)
        if p9_col_index is None and re.search(r'P9\s*25|P8\s*25|TOTAL', header_text, re.IGNORECASE):
            # Also ensure this column contains *some* numbers
            if raw_df[label].astype(str).str.isnumeric().any():
                p9_col_index = label # Save the column label/name
        
        # Identify the Reason for Contact column index
        if reason_col_index is None and re.search(r'REASON|COLD|MISSING|UNPROFESSIONAL', header_text, re.IGNORECASE):
            reason_col_index = label # Save the column label/name
            
    # Fallback for Reason column
    if reason_col_index is None:
        reason_col_index = 0 
        
    # ... (rest of the function, which uses reason_col_index and p9_col_index)
        
    if p9_col_index is None:
        st.error("Could not reliably locate the 'P9 25' count column. Cannot proceed.")
        return None, 0
    
    # --- 1. Filter: Keep only rows that have at least one numeric count ---
    start_count_col = max(0, p9_col_index - 2)
    count_cols = raw_df.columns[start_count_col : p9_col_index + 1] 
    
    temp_df = raw_df.copy()
    for col in count_cols:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

    numeric_mask = temp_df[count_cols].notna().any(axis=1)
    data_rows_df = raw_df[numeric_mask].copy()

    # --- 2. Final Filtering and Aggregation ---
    processed_df = data_rows_df[[reason_col_index, p9_col_index]].copy()
    processed_df.columns = ['Reason', 'P9_25_Count']
    
    # Filter by exact metric wording
    processed_df = processed_df[
        processed_df['Reason'].isin(TARGET_METRICS)
    ].copy()

    # Convert counts to numeric and aggregate
    processed_df['P9_25_Count'] = pd.to_numeric(processed_df['P9_25_Count'], errors='coerce').fillna(0).astype(int)
    
    final_metrics_df = processed_df.groupby('Reason')['P9_25_Count'].sum().reset_index()
    final_metrics_df.rename(columns={'P9_25_Count': 'P9_25_Occurrences'}, inplace=True)
    
    total = final_metrics_df['P9_25_Occurrences'].sum()
    
    return final_metrics_df, total

# --- 4. Styling Function for Conditional Formatting ---
def style_metrics(df):
    """Applies conditional formatting/styling to highlight non-zero counts."""
    
    def highlight_positive(s):
        is_pos = s['P9_25_Occurrences'] > 0
        return ['background-color: #e0f2f1' if is_pos else '' for v in s]

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
st.title("📄 PDF Table Extractor & Analyzer")
st.markdown("Upload your PDF to **filter by specific metrics** and calculate the total occurrences in the **P9 25** column.")

uploaded_file = st.file_uploader("Upload your PDF file:", type="pdf")

if uploaded_file:
    if st.button("Extract and Analyze P9 25 Data"):
        
        # --- Extraction ---
        raw_df = extract_pdf_tables(uploaded_file)
        
        if raw_df is not None and not raw_df.empty:
            st.subheader("Raw Extracted Data Preview (for debugging)")
            st.dataframe(raw_df.head(15)) # Show more rows for better debugging

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
