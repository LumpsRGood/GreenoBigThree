import streamlit as st
import pandas as pd
import re
import os

# --- Configuration and Constants ---
# The columns we expect based on the data structure
FILE_ID_KEY = "uploaded:R781424.pdf" # Key corresponding to the uploaded file ID
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
        # This regex is specifically looking for the restaurant IDs like '5430-Lima OH' and the Visit Type
        context_match = re.search(r'(\d+-\w+\s+\w+.*)(?:,\s*(Delivery|Dine-In|To Go))', line_clean)
        
        # Check for context setting lines
        if "OH" in line_clean or "GA" in line_clean or "VA" in line_clean:
            if "Total:" not in line_clean and any(vt in line_clean for vt in ["Delivery", "Dine-In", "To Go"]):
                 # Simple approach: find the last text block containing a location/visit type
                parts = [p.strip() for p in line_clean.split(',') if p.strip()]
                if parts:
                    # Update Restaurant/Visit Type context
                    current_context["Restaurant"] = parts[0] if "-" in parts[0] else current_context["Restaurant"]
                    
                    if len(parts) > 1 and parts[-1] in ["Delivery", "Dine-In", "To Go"]:
                         current_context["Visit Type"] = parts[-1]
                continue
        
        # Skip general administrative lines and header lines
        if any(admin in line_clean for admin in ["Area Director", "Dine Brands", "IHOP", "Reason for Contact", "P9 24"]):
            continue
            
        # Extract fields to find the Reason for Contact label
        fields = [f.strip() for f in line_clean.split(',')]
        
        # Find the text part (Reason for Contact) - usually the first non-empty field
        # Filter fields to only keep those containing letters and not being a "Total" indicator
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
                    
                    # Only append if the count is valid and it's not a secondary total row
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
            # We don't delete from session state here to allow re-runs/updates without re-pasting
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
    # The file content is manually copied here to ensure the application can run 
    # and demonstrate the parsing logic against the full sample provided by the user.
    # We must replace the placeholder with the actual content provided in the initial fetch.

    # This content is the raw text from the multi-page PDF provided in the conversation history.
    raw_pdf_content_full = """
"Restaurant Order Visit Type Reason for Contact","P9 24","P10 24","P11 24","P12 24","P1 25","P2 25","P3 25","P4 25","P5 25","P6 25","P7 25","P8 25","P9 25","Total"
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
"Slow Food","0",,,,,,,,,,,,,
"Cold Food","0",,,,,,,,,,,,,
"Infrequent Server Visits","0",,,,,,,,,,,,,
"Missing Item (Food)","2",,,,,,,,,,,,,
"Order wrong","1",,,,,,,,,,,,,
"Overcooked/Burnt","0",,,,,,,,,,,,,
"Argued With Guest","1","1001000001","0000000010","0000000000","0010000000","1100000000","1010002000","1100010000","0100000000","1001000000","0000000000","0000000000","0000000000","532222211"
"Bugs/Flies In Restaurant Damaged or worn table/booth/ chair Did Not Attempt To Resolve Issue","0","1","0","0","0","0","0","0","0","0","0",,"0","1"
"Dirty Table/booth/chair Food Dried Out Guest Left Without Dining or Ordering Health Department Potential Management Not Available Manager Did Not Follow Up Manager Did Not Visit No/Poor Apology","000 001000","000","001 100000 000010","000 000000","000 000101","000 000000","110 000000","000 010000","000 000000","000 000000","000 000000","000 000000","000 000000",
"No/Unsatisfactory Compensation Offered By","0",,,,,,,,,,,,,
"Restaurant",,,,,,,,,,,,,,
"Not Made To Order","1","0",,,,"0","0",,,,"00000",,"0","0","1"
"Out Of Menu Item","0",,,,,"0",,,,,,,,"0","1"
"Police Called","1","100","00000",,"00000","0","00010",,,"00000",,"00000","0000","0","1"
"Portion Of Food","0",,,"00000",,"0",,,"00000",,,,,"0","1"
"Presentation/appearance of food","0","1",,,,"0",,,,,,,,"0","1"
"Seating uncomfortable Server Did Not Write Down Order","0 1","00","00","00","00","00","00",,"00","00",,"00","00","00","1 1"
"Slow Check Or Change","0","1","00","00","00","00","00",,"00","00","00","00","00","00","1"
"Staff Did Not Ask For/Provide Necessary Info Table Not Cleared During Visit Temp in restaurant too cold or hot","0 0 0","1 00 0","00","0","00","00","10",,"01","00","00","00","00","0 00","1 1 1"
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
"Dine-In",,,,,,,,,,,,,,,,,
"Bugs/Flies In Restaurant Health Department Potential","0 0",,"00000",,"00000","00011",,,"00000","22210","00000",,,,"00000","0","2"
"Manager Did Not Visit","0",,,,,,,,,,,,,,,"00000",
"No/Poor Apology Did Not Attempt To Resolve Issue","0 0",,"0",,,,,"0000000",,,,"0000000","0000000","0000000",,"0","22221"
"Dirty Floors Guest Left Without Dining or Ordering Infrequent Server Visits","0 0 0",,"00 0",,"00","10",,"00","01","00",,,,"00","00 0","1"
"Manager Directly Involved In Complaint Misquoted Wait - Seating No/Unsatisfactory Compensation Offered By Restaurant","0 0",,"00 0",,"00","01",,"00","00","00","00","10","00","00","00","00"
"Not Made To Order Overcooked/Burnt Poor Attire/Appearance","0 0 0",,"","0000",,"","00000",,"1",,"","00000",,"00000",,"00000",,"00000",,"00000",,"00000",,""
"Portion Of Food","0",,"0",,,,,,,,,,,,,,
"Presentation/appearance of food","0",,"0",,,"1111",,,,,,,,,,,
"Restrooms Unsanitary","0",,"0",,"0","1",,"0","0","0","0","0","0","0","0","0","1"
"Slow Food","0","0",,,,"0",,"0","0",,,"0","0","1"
"Taste/flavor Complaint","0","0",,,,,,,,,,"0","0","1"
"Unfriendly Attitude","0","0",,"0101013","000000","00000","1010112","00000","00001","000000","000000","0000","0","1"
"Unknowledgeable","0","0",,,,,,,,,,,"1"
"Unprofessional Behavior","0","00","000000",,,,,,,,,,"1"
"Dine-In Total:","0",,,,,,,,,,,,"26"
"To Go","Management Not Available Manager Directly Involved In Complaint","0 0","00","00","01","00","00","00","00","00","00","10","00","1"
"Slow Food","0",,,,,,,,,,,,
"Unprofessional Behavior","0",,,,,,,,,,,,"2"
"Missing Condiments","0",,,,,,,,,,,,"2211"
"Missing Ingredient (Food)","0",,,,,,,,,,,,"010001"
"No Silver/Nap/Straws To Go No/Unsatisfactory Compensation Offered By Restaurant","0 0","000000","101010","010000","000000","000000","000000","000100","100000","000000","000000","000000","1 1"
"Order wrong","0",,"00100045","000000215","00000000","00000000",,,,,,"10000011",
"Out Of Menu Item","0",,,,,,,,,,,,
"Overcooked/Burnt","0",,,,,,,,,,,,
"Slow Drink Delivery","0","00000000",,,,,"00000002","00000011","00000012",,,,"11111845"
"Understaffed","0",,,,,,,,,,,,
"Unknowledgeable","0000",,,,,,,,,"00000000","00001133",,"01010066"
"To Go Total:",,,,,,,,,,,,,
"5450 - Perrysburg OH Total:",,,,,,,"12",,,,,,
"5456-Toledo OH (Talmadge Road)",,,,,,,,,,,,,
"Delivery",,,,,,,,,,,,,
"Missing Item (Food)","0","230","000","110",,,,"000","000","000","000","000","553 200"
"Order wrong","0",,,,,,,,,,,,
"Undercooked","0","0",,,,,,,,,,,"3"
"Cold Food","0","1","0",,,"0","0","0","0",,"0","0","0","2"
"Missing Condiments","0","0","000","0000","1001","0","0","200","0","0000","000","0","0","2"
"Not Made To Order","0","0",,,,"0","2",,,,,,"0","2"
"Did Not Attempt To Resolve Issue","0","0",,"0",,"0","0",,"000",,,"00","0","1"
"Hair In Food Manager Did Not Follow Up","0 0",,"000","110","001","000","000","000","000","000","000","000","000","1"
"Manager Directly Involved In Complaint","0","000 0",,,,,,,,,,,"0","1"
"Missing Ingredient (Food) No Silver/Nap/Straws To Go","0 0","0000","0000","0000","0000","0000","0000","1100","0000","0000","0000","0000","0011",
"Packaging To Go Complaint","0",,,,,,,,,,,,,
"Presentation/appearance of food","0",,,,,,,,,,,,,"1"
"Unknowledgeable","0",,,,,,,,,,,,,
"Unprofessional Behavior","0","006","000","004","119","000","002","004","000","000","000","000","004","1120"
"Delivery Total:","0",,,,,,,,,,,,,
"Dine-in",,,,,,,,,,,,,,
"Infrequent Server Visits","2",,,,,,,,,,,,,
"Slow Food","3",,,,,,,,,,,,,
"Unprofessional Behavior","3",,,,,,,,," ","10011",,,
"Cold Food","0",,,,,,,,,,,,,
"Overcooked/Burnt","0",,,,,,,,,,,,,
"Management Not Available","1",,,,,,,,,,,,,
"Missing Ingredient (Food)","2",,,,,,,,,,,,,
"Missing Item (Food) Not Made To Order","1 0","0000000000","0000010000","0010000000","0000000000","0000000000","0000000000","1011000000","1101200000","100120","1101","1000000010","0100000001","7664433333"
"Presentation/appearance of food","1",,,,,,,,,,,,,
"Slow Drink Delivery","1","00","00","00","00","00","02","10","00","00","01","10","00","33"
"Taste/flavor Complaint","0",,,,,,,,,,,,,
"Undercooked","2","0","0",,"0","0","0","1","0","0","0","0","0","3"
"Unfriendly Attitude","1","0","000","0100","000","0","0","0","0","0","001",,"0","3"
"Argued With Guest","1","0",,,,"0","0","00","000","1",,"100","0","2"
"Did Not Attempt To Resolve Issue","0","0",,,,"0","0",,,"11 1",,,"0","2"
"Manager Did Not Visit Manager Directly Involved In","1 0","00 0","00 0","01","00","00","00","00","00","11","00","00","00 0","22 2"
"Complaint",,,,,,,,,,,,,,
"No/Slow Bev Refills No/Unsatisfactory Compensation Offered By Restaurant","0 0","00","01","00","00","00","00","10","00",,"00","00","00 0","22"
"Order wrong","0","0",,,,,,,,,,,,
"Overall Poor Service","0",,,,,,,,,,,,,
"Brought Wrong Beverage Dirty Table/booth/chair Guest Left Without Dining or Ordering","0 0 1","00000","10000","00000","01000","00000","00000","00000","01000","10000","00010","00100","00000","22111"
"Long Hold/No Answer/Hung Up","1",,,,,,,,,,,,,
"No Receipt/Survey Given No/Slow Greet - Host","1 0","0000000000","0000000000","0011000000","0000000000","0000000000","0000000000","0000001011","0000010000","0000000000","0000000100","0000000000","0000000000",
"No/Slow Greet - Server","0",,,,,,,,,,,,,
"Poor Beverage Quality","1","0",,,,,,,,,,,,
"Poor Food Quality","0",,,,,,,,,,,,,
"Portion Of Food","0",,,,,,,,,,,,,
"Seating uncomfortable","0",,,,,,,,,,,,,
"Slow Check Or Change","0",,,,,,,,,,,,,
"Staff Did Not Ask For/Provide Necessary Info","0","0",,,,,,,,,,,,
"Ticket Amount Incorrect Understaffed","0 0","00","10","00","00","00","00","00","00","00","00","01","00",
"Unknowledgeable","1","0","0","05","0","0","0","0","0","0","0","0","0","1"
"Dine-In Total: To Go","24","0","4",,"1","0","2","9","7","15","11","6","2","86"
"Missing Item (Food)","1","0","010","200","700","000","000","400","110","112","000","0","2","18"
"Slow Food","2","0",,,,,,,,,,,"1","7"
"Presentation/appearance of food","1","00",,,,,,,,,,"0","2","75"
"Long Hold/No Answer/Hung Up","0","110","000","100","011","000","000","100","010","101","000","000","0","4"
"Unprofessional Behavior","1",,,,,,,,,,,,"00","43"
"Did Not Attempt To Resolve Issue","1","0",,,,,,,,,,,"0","3"
"Missing Ingredient (Food) No/Unsatisfactory Compensation Offered By","0 00","00","01","00","21","00","00","00","01","00","00","00",,"33"
"Restaurant",,,,,,,,,,,,,,
"Order wrong Understaffed Cold Food","0 0 1","0000 0000 0000 0000 1000 0000 1100 0000 0100 1110 0000 0000 0000","3322"
"Guest Left Without Dining or Ordering","2","0",,,,,,,,,,,,"2"
"Manager Directly Involved In Complaint","1","0","0","0","1",,"0","0","00100","0","0","0","0","2"
"Missing Condiments","0","1000","0010","0000","1011","0000","0000","0000",,"0000","0000","0000","0000","2222"
"No/Slow Greet - To Go","1",,,,,,,,,,,,,
"Out Of Menu Item","0",,,,,,,,,,,,,
"Staff Did Not Ask For/Provide Necessary Info","1",,,,,,,,,,,,,
"Unfriendly Attitude","1",,,"00000",,"00000","01000","00000","00001","10000","00000","00000",,"22111"
"Unknowledgeable","1","00000",,,,,,,,,,,"00000",
"Didn't Open/close On Time","1",,"00000",,,,,,,,,,,
"Employee On Cell Phone","1",,,,"00000",,,,,,,,,
"Food Dried Out","0",,,,,,,,,,,,,
"Management Not Available","0","0","0","1","0","0","0","0","0","0","0","0","0","1"
"Manager Did Not Follow Up","0","0","0","0","0","0","0","0","0","0","0","0","0","1"
"Missing Item (Bev)","0","0","0","0","0","0","0","1","0","100000010122",,"0","0","1"
"Music/noise loud","1","0",,," ","0","0","0",,,,"0","0","1"
"No/Poor Apology Not Made To Order Overcooked/Burnt Portion Of Food Taste/flavor Complaint Undercooked To Go Total:","0 1 0 0 0 0 18","0 0 0 0 0 0","000000037 0000000043 0 0 0 0 17","00000000 00000148 09000000 010000085","","0000000000","0 0 1 1 0 0 8","1 1 1 1 1 1 84"
"5456-Toledo OH (Talmadge Total: Road)","42","0000039",,"13","0727",,,"19",,"27",,"00000017","14","199"
"5461-Holland OH (Airport Hwy)","Delivery",,,,,,,,,,,,,,
"Cold Food","0","0",,,,,,,,,,,,
"Did Not Follow Up Manager","0",,"00000000","11001104","00000011","00100001",,"00000000","00000000","00000000",,"00000000","00000000",
"Order wrong","0",,,,,,,,,,,,,
"Out Of Menu Item","0",,,,,,,,,,,,,
"Overcooked/Burnt","0",,,,,,,,,,,,,
"Undercooked","0",,,,,,,,,,,,,
"Unknowledgeable","0","0000000",,,,,"00010001",,,,"00000000",,,"111117"
"Delivery Total:","0",,,,,,,,,,,,,
"Dine-In",,,,,,,,,,,,,,
"Cold Food","1",,,,,,,,,,,,,
"Slow Food","1",,,,,,,,,,,,,
"Manager Directly Involved In Complaint","0","1","000","421","100","000","212",,,,"000","000","010","985"
"No/Unsatisfactory Compensation Offered By Restaurant","1","1","0","2","0","0","0","0","1","0","0","0","0","5"
"Portion Of Food","0","0","0","1",,"3","0","0","0","1","0","0","0","0","5"
"Presentation/appearance of food","0","0","0","1",,"1","0","0","0",,"1","0","0","1","5"
"No/Slow Bev Refills","0","1",,,,,,,"0",,,,,"0","4"
"Order wrong","0","32","000","201",,"110","000","000","001","000","000","000","000","000","44"
"Staff Did Not Ask For/Provide Necessary Info","0","2",,,,,,,,,,,,,"4"
"Unprofessional Behavior","0","1","00","00",,"01",,"20","00","00","00","00","00","0","4"
"Did Not Attempt To Resolve Issue","0","1",,,,,,,,,,,,"00","3"
"Missing Ingredient (Food)","0","0",,"0001",,,"0000",,,"0010",,"0000","0000","1","3"
"Missing Item (Food)","0","0","0000",,,"0000",,"2310","0000",,"0002",,,"0","3"
"Taste/flavor Complaint","1","00",,,,,,,,,,,,"0","33"
"Temp in restaurant too cold or hot Alleged Foreign Object (food) Food Dried Out Guest Left Without Dining or Ordering","0 0 0 0","000 0","000","001",,"000","001","000","000","010","000","000","000","0 2 210 0","2 222 2"
"Meals Not Served Together","0","0",,"000100001001",,"000000010000","000000100010","100000000000","110000000000","000010000000",,"000000000000","000000000000","0",
"No/Poor Apology","0","01","000000000000",,,,,,,,"001000000000",,,"0","222"
"Overall Poor Service","0","0",,,,,,,,,,,,"1",
"Slow Drink Delivery","0","1",,,,,,,,,,,,"0","2"
"Undercooked","0","1",,,,,,,,,,,,"0","2"
"Understaffed","1",,,,,,,,,,,,,,
"Unfriendly Attitude","0",,,,,,,,,,,,,,"2222"
"Unknowledgeable","0","100000",,,,,,,,,,,,"00010000","2"
"Billing Other","0",,,,,,,,,,,,,,"1"
"Dirty Floors","1",,,,,,,,,,,,,,"1"
"Dirty Utensils","0",,,,,,,,,,,,,,
"Employee Behavior Unsanitary","0",,,,,,,,,,,,,,
"Infrequent Server Visits","0","0","0",,"0","0","0","0","0","00 0",,"0","0","0 1"
"Language barrier/Couldn't understand team member","0","1","0","10 0","0","0","0","0","0",,,"0","0","0 1"
"Long Hold/No Answer/Hung Up","1","0","0","0",,"0000","0011","0","0000","0000",,"0000","0000","0 0000 1"
"Management Not Available","0","0000","0000","0100","0000",,,"0000",,,,,,"1"
"Manager Did Not Visit","0",,,,,,,,,,,,,"1 11"
"Negative Manager-Employee Interaction","0","0","0",,,,,,,,,,,"1"
"No/Slow Greet - Host","0",,,,,,,,,,,,,"1"
"No/Slow Greet - Server Not Made To Order Out Of Menu Item Overcooked/Burnt","0 0 0 0","100010020","00000000","000001123","00100000","01000006","000000016","00000005","00010008","00000004"," 00000000","00000000","00000008"
"Slow Check Or Change Taste/Flavor Of Beverage","0 0"," 07 7",,,,,,,,,,,,,"1 107"
"To Go",,,,,,,,,,,,,,
"Missing Item (Food)","0",,"3011000","1100000","0000000","0011000","2100011","0000000","1111000",,,"0100000",,"0000000"
"Slow Food","0",,,,,,,,,,,,,
"Missing Ingredient (Food)","0","0210210",,,,,,,,"2000000",,,"0000000","96432"
"Portion Of Food","0",,,,,,,,,,,,,
"Missing Item (Bev)","0",,,,,,,,,,,,,
"Unprofessional Behavior","0",,,,,,,,,,,,,"2"
"Carside/Curbside To Go Not Available","0",,,,,,,,,,,,,"1"
"Cold Food","0 00","10","00","00","00","01","00","00","00","00",,"00","00","00"
"Did Not Attempt To Resolve Issue",,,,,,,,,,,,,,
"Food Dried Out","0","0",,,,,,,,,,,,"000"
"Management Not Available Manager Did Not Follow Up","0 0","1 1","000","000","000","000","000","000","000","000",,"100","000","1"
"Missing Condiments","0","0",,,"0","0","0","0","0","0","0","0","0","1"
"Order wrong","0","0","10000","00000","0","0","1","0","0","0000","0","0","0","1"
"Out Of Menu Item","0","1",,,,,,"0",,,,"0","0","1"
"Packaging To Go Complaint","0","0",,,"0000","010","1000","00","0000",,"0001","00","0","1"
"Presentation/appearance of food","0","0",,,,,,,,,,,"0","1"
"Understaffed","0","1",,,,,,"0",,,,,"0","1"
"Unknowledgeable","0","1",,"00223","0001",,,"0005",,,,,"0","1"
"To Go Total: 5461-Holland OH (Airport Total: Hwy)","0 7","12 32","0066 29","11","0041","00623","5","0042","0026","0033","0000","0 8","39 153"
"Angela Pascarella Total:","58","57","28","59","41","13","51","31","31","36","17","8","31","461"
"Ben Hinojosa","0406 - Sandy Springs GA",,,,,,,,,,,,,,
"Delivery",,,,,,,,,,,,,,
"Missing Item (Bev) Presentation/appearance of food Undercooked","0 0 0","0 00 00","00","01 0 00","10 00 00","00 00 00","10 00 00","00 00 00","00 00 00",,"00 00 00","00 00 00",,"00 00 00","2 22 2 1"
"Did Not Attempt To Resolve Issue","","00",,,,,,,"01",,,"00",
"Franchise Did Not Follow Up - 2nd Attempt","0","0","0","0","0","0","1","0","0","0","0","0","0","1"
"Management Not Available","0","0",,,,,,,,,,,,
"Missing Condiments","1",,,,,,,,,,,,,
"Missing Item (Food)","0","00000000","10000000","00000000","00100010","00000000","00000000","00000000","00000001","00000000","00000000","00000000","00000000",
"Not Made To Order","1",,,,,,,,,,,,,
"Overcooked/Burnt","1",,,,,,,,,,,,,
"Packaging To Go Complaint","1",,,,,,,,,,,,,
"Portion Of Food","0",,,,,,,,,,,,,
"Spoiled/Expired Food","0",,,,,,,,,,,,,
"Taste/flavor Complaint","0","0","0",,,,"0","0","0","1",,,"0","0","1"
"Unprofessional Behavior Wouldn't Allow Substitution","0 0","0 0","1","0000",,"0004","00","00","000","005","0000","0001","000","00","1 1"
"Delivery Total:","4","0","113",,,,"0","2",,,,,,"0","19"
"Dine-In",,,,,,,,,,,,,,,
"Guest Left Without Dining or Ordering","0","0","2","1",,"0","0","0","0","0","1","0","0","0","4"
"Overcooked/Burnt","0","0",,,,,,,,,,,,,"4"
"Unprofessional Behavior","1","000","001",,"121",,"011","200","100","000","000","000","000","000","4"
"Manager Directly Involved In Complaint Did Not Attempt To Resolve Issue No/Slow Greet - Server","0 0 0","0 0","1","0",,"0000 000","1","0","0","0","0","0","0","0 0","3 2 2"
"Portion Of Food","0","000",,,,,,,,,,,,"000","2"
"Presentation/appearance of food","0",,"101","000",,,"000","000","001","000","120","000","000",,"2"
"Employee Behavior Unsanitary","1","0",,,,,,,,,,,,"0","1"
"Hair In Food","0",,,,,,,,,,,,,"0","1"
"No/Poor Apology","0",,,,,,,,,,,,,"0","1"
"No/Slow Greet - Host Order wrong","0 0","0000","010000000018","000000000106",,"000000000000","000000000003","000000001003","001000000003","000000000000","000100010006","000010100002","000000000000","0 0","1 1"
"Out Of Menu Item","1","0",,,,,,,,,,,,"0","1"
"Slow Drink Delivery","0","0",,,,,,,,,,,,"0","1"
"Slow Food","0","0",,,,,,,,,,,,"0","1"
"Taste/flavor Complaint","0","0",,,,,,,,,,,,"0","1"
"Ticket Amount Incorrect","0","0",,,,,,,,,,,,,"1"
"Undercooked","0","0",,,,,,,,,,,,"000","1"
"Dine-In Total: To Go","3","00",,,,,,,,,,,,,"34"
"Missing Item (Food)","0","0","0",,"0","0","0","3","1","0","0","0","0","4"
"No/Slow Greet - To Go"," 0",,,,,,,,,,,,,
"Manager Directly Involved In Complaint Presentation/appearance of food","0 0","0 0","0 0","0 0",,"0 1","1 0","1 0","0 0","0 0","0 1","0 0","0 0","0 0","2 2 0"
"Slow Food","0",,,,,,,,,,,,,
"Unfriendly Attitude","0","0000","0000",,"0000","0100","0100","1001","0000","1000","0010","0000","0000","22 0000"
"Cold Food","0",,,,,,,,,,,,,"1"
"Did Not Attempt To Resolve Issue","0",,,,,,,,,,,,,"1"
"Didn't Open/close On Time Employee Behavior Unsanitary","0 0","00",,,,,,,,,,,,"00"
"Extra Ingredient in Food","0","0",,,,,,,,,,,,"0 1"
"Long Hold/No Answer/Hung Up Missing Condiments","0 0","0000","0000000","1000000",,"0010010","0000100","0101000","0000000","0000001","0000000","0000000","0000000","1 1"
"No/Unsatisfactory Compensation Offered By Restaurant","0",,,,,,,,,,,,,"1"
"Order wrong Portion Of Food Staff Did Not Ask For/Provide Necessary Info","0 0 0 0","000","000","000",,"010","000","001 010","000","000","000","000","000","1"
"Unprofessional Behavior","0",,,,,,,,,,,,,
"To Go Total: 0406-Sandy Springs GA Total:","0 7","0000","0001","0017",,"0059","0036","15","0014","0027","1039","0003","0000","0011"
"0409 - Norcross GA (Jimmy Dine-In No/Slow Greet - Server Carter)","0 0 0",,"0","0","1","0",,"1","0",,"0","0 2"
"Unprofessional Behavior","0","00","0",,,"0",,,"010",,,,,"2"
"Alleged Race Discrimination","0",,"0",,,,"1",,,,,,,"0 1"
"Guest Left Without Dining or Ordering","0","0",,"0","0",,"0","1","0","0",,"0","0",,"0","0",,"0","1"
"Infrequent Server Visits","0","0",,"0",,"0","0","0","0","10 1",,"0","0",,"0","0",,"0","1"
"Manager Directly Involved In Complaint","0","0",,"0",,"0","0","1","0","0",,"0","0",,"0","0",,"0","1"
"Ticket Amount Incorrect","0","00",,"00","00",,"00","05","0",,,"0","00",,"11","00",,"00",
"Dine-In Total:","0",,,,,,,,"00","02",,"01",,,"1",,,,"19"
"To Go","Order wrong","0",,,,,,,,,,,,,,,
"Management Not Available","0",,,,,,,,,,,,,,,
"Missing Item (Food)","0","000000",,"000000","110133",,"000000","000005","000000","000002",,"001012","100011",,"000001","000000",,"000000","2111514"
"Out Of Menu Item","0",,,,,,,,,,,,,,,
"To Go Total: 0409 - Norcross GA (Jimmy Total: Carter)","0 0",,,,,,,,,,,,,,,
"3775- Evansville IN Dine-In","Cold Food","0","0",,,,,,,,,,,,,,,,,
"Infrequent Server Visits","0","0",,"000","213",,,,,,,"330","100",,"000","010",,"010","966"
"Presentation/appearance of food","0","1",,"៖",,,,,,,,,,,,,,,
"Manager Directly Involved In Complaint","0","1",,"0",,"0","0","2","0","0",,"1","0",,"0","1",,"0","کا"
"Slow Food","0","0",,,,,,,,,,,,,,,,,
"Undercooked","0",,,,,,,,,,,,,,,,,,
"No/Slow Bev Refills No/Unsatisfactory Compensation Offered By Restaurant","0 0","0000",,"0000",,"0211","0000","0102","0200","0000",,"501T","0010",,"0000","0010",,"0000","5544"
"Slow Check Or Change Unprofessional Behavior","0 0","000",,"000","110",,,,,,,,,,,,,,
"Manager Did Not Visit","0","0",,,,,,,,,,,,,,,,,
"Argued With Guest"," 0",,,,,,,,,,,,,
"No/Poor Apology","0","0 0",,,"0",,,"0","1",,"0","1","0","3"
"No/Slow Leftover Box/Cup","0","0",,,,,,"0",,,,"0","0","3"
"Obnoxious (loud & Unruly) Out Of Menu Item","0 0","0 0",,"01000","0000","10010","00121","0000","2200","00000","0000","000","0 0","3 3"
"Language barrier/Couldn't understand team member Missing Item (Food) Negative Manager-Employee Interaction","0 0 0","1 0 0","00",,"00","00","00","00","1","00","00","10","0 0 0","2 22 2"
"Overcharged On Debit/Credit Card (Posted) Taste/flavor Complaint","0 0","1 0 1",,"0","0","0","0","0","1","0","0","0","0","2 22 2"
"Unfriendly Attitude","0","0","000",,"000","000","100","000","001","020","000","000","000","2"
"Alleged Dangerous Environment","0","0",,,,,,,,,,,,"1"
"Bugs/Flies In Restaurant Did Not Attempt To Resolve Issue","00 0","000 0","001",,"000","000","000","000","100","000","000","010","000",
"Employee Behavior Unsanitary General Seating Complaint Guest Left Without Dining or Ordering Hair In Food","0 000 0","000 000 00","000",,"000","000","000","000","000","000","010","100","001",
"Long Hold/No Answer/Hung Up","0","00000000",,,,,,,,,,,,
"Meals Not Served Together No Silver/Nap/Straws DineIn- No/Slow Greet - Host No/Slow Greet - Server","0 0 0 0","0 0","10000000",,"00000000","00000000","00000001","00000000","01100100","00010000","00000000","00000010","00001000",
"Not Made To Order","0","0000",,,,,,,,,,,,
"Order wrong","0",,,,,,,,,,,,,
"Overcooked/Burnt","0","0","0",,"0","0","1","0","0","0","0","0","0","1"
"Poor Food Quality","0","0","0000","00000","00100","0","0","0","0","1000","0000","0001","0","1"
"Portion Of Food","0","0",,,,,,,,,,,"0","1"
"Rushed Svc","0","0",,,,"0000","0000",,,,,,"0","1"
"Server Did Not Write Down Order","0","0",,,,,,"0000","0010",,,,"0","1"
"Slow Drink Delivery","0","00","00","00","00","00","00","00","01","00","00","00","10","1"
"Staff Did Not Ask For/Provide Necessary Info","0",,,,,,,,,,,,"0","1"
"Tough Food Understaffed","000 0","106","001","0015","002","001","1000","000","01735","006","001","002","004","103"
"Missing Item (Food)","0","0",,"1000","1100","2110","0000","0001",,,"0000","0000","2010","7321"
"Order wrong","0","0000","1100",,,,,,"0000","0000",,,,
"Cold Food","000",,,,,,,,,,,,,
"Did Not Attempt To Resolve Issue Guest Left Without Dining or Ordering","0 0","0 0","0","0","0","0","0","1","0","0","0","0","0","1"
"Hair In Food Manager Did Not Follow Up Missing Condiments No Silver/Nap/Straws To Go Slow Food Spoiled/Expired Food Unprofessional Behavior To Go Total: 3775- Evansville IN Total:","0 0 0 0 0 0 0 0 0","00 0 0 0 0 0 0 0 06","010000034 00000001 000000024 001100067 000000000 0000000035 000000006 000000001 000001012 13","1 0 0 0 1 0 1 1 21 124"
"""

    st.session_state[FILE_ID_KEY + '_content_raw'] = raw_pdf_content_full

    # We must explicitly look for the raw content key set by the initial fetch tool.
    # The check for os.environ is specific to the sandbox environment initialization
    if FILE_ID_KEY in os.environ and os.environ.get('DATA_SOURCE_PATH') == FILE_ID_KEY:
         st.session_state[FILE_ID_KEY + '_content_raw'] = raw_pdf_content_full # Ensure session state is populated

    app()
