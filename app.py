import streamlit as st
import pandas as pd
import pdfplumber
import io
import os

# --- Extraction Function using pdfplumber ---
def extract_pdf_tables_pdfplumber(uploaded_file):
    # Use io.BytesIO to read the uploaded file directly in memory
    pdf_bytes = uploaded_file.read()
    all_data = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        st.write(f"Processing {len(pdf.pages)} pages...")
        
        for i, page in enumerate(pdf.pages):
            # Extract tables using default settings. 
            # If the tables are complex, you may need to define table_settings here.
            tables = page.extract_tables() 
            
            # tables is a list of lists (the data rows) for each table found on the page
            for table_data in tables:
                if table_data:
                    # Convert the table data (list of lists) to a DataFrame
                    df = pd.DataFrame(table_data)
                    
                    # Tag the data with the page number for organization
                    df['Source_Page'] = i + 1 
                    all_data.append(df)

    if not all_data:
        return pd.DataFrame({'Status': ['No tables were successfully extracted.']})
    
    # Combine all extracted tables/pages into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

# --- Streamlit App Interface ---
st.title("Streamlit Cloud PDF Table Extractor")
st.markdown("Uses the free, pure-Python **`pdfplumber`** library.")

uploaded_file = st.file_uploader("Upload your PDF file:", type="pdf")

if uploaded_file:
    if st.button("Extract Data"):
        with st.spinner('Extracting tables from PDF...'):
            extracted_df = extract_pdf_tables_pdfplumber(uploaded_file)
            
            # --- Data Cleaning and Aggregation ---
            
            # This step requires knowing your final column structure.
            # Assuming the columns are 'Reason' (index 0) and 'P9 25' (index 14)
            try:
                # 1. Select the relevant columns (adjust indices as needed)
                # Note: Headers often appear as the first row.
                processed_df = extracted_df.iloc[:, [0, 14]].copy() 
                processed_df.columns = ['Reason', 'P9_25_Count']
                
                # 2. Convert counts to numeric
                processed_df['P9_25_Count'] = pd.to_numeric(processed_df['P9_25_Count'], errors='coerce')
                
                # 3. Clean up empty/header rows
                processed_df.dropna(subset=['P9_25_Count'], inplace=True)
                
                # 4. Calculate total
                total = processed_df['P9_25_Count'].sum()
                
                st.success(f"Extraction Complete. Total Occurrences in P9 25: **{int(total)}**")
                
                st.subheader("Final Processed Data (Occurrences > 0)")
                # Show only metrics that have an occurrence
                metrics_df = processed_df[processed_df['P9_25_Count'] > 0]
                st.dataframe(metrics_df)

                # Download Button
                csv = metrics_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed CSV",
                    data=csv,
                    file_name='p9_25_metrics_report.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"Error during data processing/cleaning. Check column indices. Details: {e}")
                st.dataframe(extracted_df.head(20)) # Show raw data for debugging
