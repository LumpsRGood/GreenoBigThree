# --- Data Cleaning and Aggregation Function (FINAL REVISION) ---
def clean_and_aggregate_data(raw_df):
    """
    Locates target columns robustly, uses numeric validation to find data rows,
    filters by exact metric, and performs final calculations.
    """
    
    # Clean up column values by stripping whitespace and converting to string
    raw_df = raw_df.applymap(lambda x: str(x).strip() if x is not None else None)
    
    # --- Robust Column Index Mapping (Find P9 25 and Reason) ---
    p9_col_index = None
    reason_col_index = None

    HEADER_SCAN_ROWS = 5
    
    for col in raw_df.columns:
        # Stop searching if we check too many columns (e.g., beyond column 15)
        if col > 15: 
            break
            
        header_fragments = [
            str(raw_df[col].iloc[r]).strip() for r in range(min(HEADER_SCAN_ROWS, len(raw_df)))
        ]
        header_text = " ".join(filter(None, header_fragments)).upper()

        # Identify the P9 25 column index
        if p9_col_index is None and re.search(r'P9\s*25|P8\s*25|TOTAL', header_text, re.IGNORECASE):
            # Also ensure this column contains *some* numbers
            if raw_df[col].astype(str).str.isnumeric().any():
                p9_col_index = col
        
        # Identify the Reason for Contact column index (typically column 0 or the first text column)
        if reason_col_index is None and re.search(r'REASON|COLD|MISSING|UNPROFESSIONAL', header_text, re.IGNORECASE):
            reason_col_index = col
            
    # Fallback for Reason column
    if reason_col_index is None:
        reason_col_index = 0 
        
    if p9_col_index is None:
        st.error("Could not reliably locate the 'P9 25' count column. Cannot proceed.")
        return None, 0
    
    # --- 1. Slice and Preliminary Numeric Check ---
    # Create a temporary DataFrame with the data columns of interest (P9 25 and the two preceding it for context)
    # This range handles the common scenario where P9 25 is column N, and counts start around column N-2
    start_count_col = max(0, p9_col_index - 2)
    count_cols = raw_df.columns[start_count_col : p9_col_index + 1] 
    
    # Temporarily attempt to convert these columns to numeric (coercing errors to NaN)
    temp_df = raw_df.copy()
    for col in count_cols:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

    # --- 2. CRITICAL ROW FILTER: Keep only rows that have at least one numeric count ---
    # We create a boolean mask: a row is kept if the sum of all numeric columns in the count range is not NaN (i.e., it had a number)
    # This automatically discards all pure text header rows, which is what you need.
    numeric_mask = temp_df[count_cols].notna().any(axis=1)
    
    # Filter the original DataFrame using this mask
    data_rows_df = raw_df[numeric_mask].copy()

    # --- 3. Final Filtering and Aggregation ---
    
    # Slice the filtered data to just the two columns we care about
    processed_df = data_rows_df[[reason_col_index, p9_col_index]].copy()
    processed_df.columns = ['Reason', 'P9_25_Count']
    
    # Filter by exact metric wording (guarantees clean results)
    processed_df = processed_df[
        processed_df['Reason'].isin(TARGET_METRICS)
    ].copy()

    # Convert counts to numeric (should work perfectly now)
    processed_df['P9_25_Count'] = pd.to_numeric(processed_df['P9_25_Count'], errors='coerce').fillna(0).astype(int)
    
    # Group by Reason to handle duplicates across pages and calculate final total
    final_metrics_df = processed_df.groupby('Reason')['P9_25_Count'].sum().reset_index()
    final_metrics_df.rename(columns={'P9_25_Count': 'P9_25_Occurrences'}, inplace=True)
    
    total = final_metrics_df['P9_25_Occurrences'].sum()
    
    return final_metrics_df, total
