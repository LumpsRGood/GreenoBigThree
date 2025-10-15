import streamlit as st
import pandas as pd
import re
import os

# --- Configuration and Constants ---
# The columns we expect based on the data structure
PERIOD_COLUMNS = [f"P{i} 24" for i in range(9, 13)] + [f"P{i} 25" for i in range(1, 10)]
TOTAL_COLUMN = "Total"
COLUMNS_TO_DISPLAY = ["Restaurant", "Visit Type", "Reason for Contact", "Total Count"]

def parse_raw_text(raw_text):
    """
    Parses the raw text into structured data.

    This logic is designed to be highly resistant to the column-gluing problem 
    in the intermediate P columns by focusing only on the final "Total" column.
    """
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    data = []
    current_context = {"Restaurant": "N/A", "Visit Type": "N/A"}
    
    # Pre-compiled regex for efficiency
    context_re = re.compile(r'(\d+-\w+\s+\w+(?:\s+\w+)*)(?:\s*|\s*,\s*)(Delivery|Dine-In|To Go)')
    total_re = re.compile(r',\s*(\d+)\s*$') # Finds the last comma-separated number (The Total)

    # 1. First Pass: Identify the Reason for Contact and Total
    st.info("Pass 1: Identifying Categories and Totals (ignoring messy period columns)")
    
    for line in lines:
        line_clean = line.replace('"', '').strip()

        # Update Context: Look for the Restaurant/Visit Type header lines
        context_match = context_re.search(line_clean)
        if context_match:
            current_context["Restaurant"] = context_match.group(1).strip()
            current_context["Visit Type"] = context_match.group(2).strip()
            continue

        # Skip general administrative lines
        if any(admin in line_clean for admin in ["Area Director", "Dine Brands", "IHOP", "Total:"]):
            continue
            
        # Extract fields to find the Reason for Contact label
        fields = [f.strip() for f in line_clean.split(',')]
        
        # Find the text part (Reason for Contact) - usually the first non-empty field
        reason_parts = [f for f in fields if re.search(r'[a-zA-Z]', f) and not f.endswith("Total")]
        
        if reason_parts:
            # Join the text parts to form the Reason for Contact
            reason_for_contact = " ".join(reason_parts).strip().strip(':')
            
            # Now, find the 'Total' number, which is reliably the last field in most data rows
            # We look for the last field that is purely a digit string
            last_number_field = next((f for f in reversed(fields) if re.match(r'^\s*\d+\s*$', f.replace(' ', ''))), None)

            if last_number_field:
                try:
                    total_count = int(last_number_field.replace(' ', '').strip())
                    
                    if total_count >= 0 and not reason_for_contact.endswith("Total"):
                        data.append({
                            "Restaurant": current_context["Restaurant"],
                            "Visit Type": current_context["Visit Type"],
                            "Reason for Contact": reason_for_contact,
                            "Total Count": total_count
                        })
                except ValueError:
                    # Ignore lines where the "total" field is messy/merged but contains non-digits
                    continue

    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    
    # 2. Second Pass (Aggregation and Filtering)
    st.info("Pass 2: Aggregating unique categories.")
    
    # Consolidate duplicate reasons (e.g., 'Cold Food' appearing multiple times)
    # Group by all three identifiers to maintain context specificity
    df_grouped = df.groupby(COLUMNS_TO_DISPLAY[:-1])['Total Count'].sum().reset_index()
    
    return df_grouped

# --- Streamlit Application ---
def app():
    # Attempt to access the raw content of the uploaded file via the environment's method
    raw_content_key = FILE_ID_KEY + '_content_raw' # Assuming this structure for simplicity
    
    if raw_content_key in st.session_state:
        raw_text_input = st.session_state[raw_content_key]
        if raw_text_input:
            st.sidebar.success("PDF Content Loaded Successfully.")
            # Clear file content from session state after loading to prevent immediate re-run confusion
            # del st.session_state[raw_content_key]
        else:
            st.sidebar.error("Could not retrieve raw text content.")
            raw_text_input = ""
    else:
        # Fallback for running outside the initial file upload context or in Streamlit Cloud
        st.sidebar.header("Upload PDF Text")
        st.sidebar.markdown("Since direct PDF parsing requires extra libraries/J-RE, paste the raw text you extracted below.")
        raw_text_input = st.sidebar.text_area("Paste Raw PDF Text Here:", height=400)
    
    
    if raw_text_input:
        
        st.header("1. Parsed Data Preview")
        df_raw = parse_raw_text(raw_text_input)

        if df_raw.empty:
            st.warning("No structured data could be successfully parsed from the provided text using the specialized logic.")
            return

        st.dataframe(df_raw, height=300, use_container_width=True)
        
        st.header("2. Categories with Data (Total Count > 0)")
        
        # Aggregate and filter
        categories_df = df_raw.groupby('Reason for Contact')['Total Count'].sum().reset_index()
        categories_with_data = categories_df[categories_df['Total Count'] > 0].sort_values(by='Total Count', ascending=False)
        
        st.metric(label="Total Unique Non-Zero Categories Found", value=len(categories_with_data))
        st.dataframe(categories_with_data, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.header("3. Contextual Breakdown (Top 20)")
        st.dataframe(df_raw[df_raw['Total Count'] > 0].head(20).drop(columns=['Restaurant', 'Visit Type'], errors='ignore'), 
                     use_container_width=True, hide_index=True)


if __name__ == "__main__":
    # --- Simulate initial file content loading for execution within this environment ---
    # This is a hack to get the file content into the session state for the Streamlit app.
    # In a real Streamlit cloud environment, you would use st.file_uploader.
    
    # Extract the raw text from the code environment's snippet access
    import inspect
    
    # Corrected the function name from 'parse_pdf_text_to_dataframe' to 'parse_raw_text'
    full_code = inspect.getsource(parse_raw_text) 
    
    # We must explicitly look for the raw content key set by the initial fetch tool.
    if FILE_ID_KEY in os.environ and os.environ.get('DATA_SOURCE_PATH') == FILE_ID_KEY:
         # Load the entire raw content into session state for the app function to access
         raw_content = ""
         # Assuming the content is passed back as a variable named 'content' from the fetch tool
         # Since that's not standard, we must rely on manual snippet or local environment setup.
         st.session_state[FILE_ID_KEY + '_content_raw'] = """
[Paste the extremely long raw text content here for testing outside of the execution environment, but for this context, assume the parser function relies on the line-by-line parsing of the full text that was previously fetched.]
"""
         
    # To ensure the parser runs on a good amount of the actual data provided by the user:
    # We will embed a large, but manageable, chunk of the user's data into the session state.
    # NOTE: This chunk is derived from the full fetched content in the previous turn.
    
    raw_pdf_content_snippet = """
"Restaurant",,,,,,,,,,,,,,
"Order Visit Type",,,,,,,,,,,,,,
"Reason for Contact","P9 24","P10 24","P11 24","P12 24","P1 25","P2 25","P3 25","P4 25","P5 25","P6 25","P7 25","P8 25","P9 25","Total"
"Angela Pascarella",,,,,,,,,,,,,,
"5430-Lima OH",,,,,,,,,,,,,,
"Delivery",,,,,,,,,,,,,,
"Cold Food","0","0",,,,"0","0","0","0",,,"0","0","1"
"Long Hold/No Answer/Hung Up Missing Condiments","0 0","0 0",,"000000","000000",,"0","0","0","000000","000000","00000","1 0","1 1"
"Missing Item (Food)","0","0",,,,"0000","00000","00000","00000",,,,"001","1"
"Tough Food","0",,"101114 00",,,,,,,,,,,"115"
"Delivery Total:","0",,,,,,,,,,,,,
"Dine-In",,,,,,,,,,,,,,
"Unprofessional Behavior","0",,,,,,,,,,,,,
"Missing Item (Food)","2",,,,,,,,,,,,,
"Order wrong","1",,,,,,,,,,,,,
"Argued With Guest","1","1001000001","0000000010","0000000000","0010000000","1100000000","1010002000","1100010000","0100000000","1001000000","0000000000","0000000000","0000000000","532222211"
"Bugs/Flies In Restaurant Damaged or worn table/booth/ chair Missing Condiments","0","1","0","0","0","0","0","0","0","0","0",,"0","1"
"Dirty Table/booth/chair Food Dried Out Guest Left Without Dining or Ordering Health Department Potential Management Not Available Manager Did Not Follow Up Manager Did Not Visit No/Poor Apology","000 001000","000","001 100000 000010","000 000000","000 000101","000 000000","110 000000","000 010000","000 000000","000 000000","000 000000","000 000000","000 000000",
"No/Unsatisfactory Compensation Offered By","0",,,,,,,,,,,,,
"Restaurant",,,,,,,,,,,,,,
"Not Made To Order","1","0",,,,"0","0",,,,"00000",,"0","0","1"
"Out Of Menu Item","0",,,,,"0",,,,,,,,"0","1"
"Police Called","1","100","00000",,"00000","0","00010",,,"00000",,"00000","0000","0","1"
"Ticket Amount Incorrect","1",,,,,,,,,,,,,,"1"
"Dine-In Total:","9","09","03","00","03","02","08",,"05","01","03","00","00","00","43"
"To Go",,,,,,,,,,,,,,,
"Order wrong","0","1",,,,,,,,,,,,,
"Out Of Menu Item","0","1",,,,,,,,,,,,,
"Slow Food","0","0",,,,,,,,,,,,,
"Unknowledgeable","0",,,,,,,,,,,,,,"32221"
"Alleged Foreign Object (food)","0","1100","2000001","0101000","0000000","0000000","0000000",,"0010000","0000010",,"0000000","0000000","0010000",
"Cold Food","0",,,,,,,,,,,,,,"1"
"Did Not Attempt To Resolve Issue No/Unsatisfactory Compensation Offered By Restaurant","0 0","1","0","0","0","0","0",,"0","0"," 0","0","0","0","1 1"
"Poor Food Quality Staff Did Not Ask For/Provide","0 0","1 1",,"00","00","00","0","00",,"00","00","00","00","00",,
"Unfriendly Attitude To Go Total:","0 0",,"0 7",,"0 3","0 2",,"0 0","0 0","0","0","0","003","000","000","123","1 16"
"5430-Lima OH Total:","9",,"16",,"10","2",,"3","2","8",,"2",,,,,"64"
"5450-Perrysburg OH",,,,,,,,,,,,,,,,,
"Delivery",,,,,,,,,,,,,,,,,
"Missing Item (Food) Delivery Total:","0 0",,"0 00",,"1 1","00",,"00","00","00","00","00","00","00","00","0","1"
"""
    st.session_state[FILE_ID_KEY + '_content_raw'] = raw_pdf_content_snippet


    app()
