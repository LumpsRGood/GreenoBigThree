import pandas as pd
import re

def parse_pdf_data(raw_text):
    """
    Parses the raw text extracted from the PDF into a pandas DataFrame.
    """
    
    # 1. Clean the overall text to make it easier to process
    # Remove the metadata/non-table lines (e.g., page numbers, 'Area Director')
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    # Identify the main table start and end for one of the tables
    table_start_indices = [i for i, line in enumerate(lines) if "Reason for Contact" in line]
    
    if not table_start_indices:
        print("Could not find table header 'Reason for Contact'.")
        return pd.DataFrame()

    # Assuming the first table is the main one to parse
    start_index = table_start_indices[0]
    
    # Extract column names - this is simplified as the columns are messy
    # A cleaner approach is to hardcode based on the known structure
    period_cols = [f"P{i} 24" for i in range(9, 13)] + [f"P{i} 25" for i in range(1, 10)]
    column_names = ["Restaurant", "Order Visit Type", "Reason for Contact"] + period_cols + ["Total"]
    
    data = []
    current_row_label = None
    
    # Iterate through lines starting just after the header
    for i in range(start_index + 1, len(lines)):
        line = lines[i].strip()
        
        # Split line by comma, ignoring commas inside quoted fields for the headers
        # This is a key step where a robust CSV reader might fail due to malformed data
        fields = [f.strip().strip('"') for f in line.split(',')]
        
        # Simple heuristic to identify a row header vs. a data row:
        # A row header has data only in the first few columns (Restaurant, Visit Type, Reason)
        # A data row starts with numerical data or empty strings/commas
        
        # --- Simplified Parsing for the Specific Provided Format ---
        # Look for the last quoted/non-empty string which is the 'Reason for Contact'
        
        # The line structure is: "Header Info 1",...,"Header Info N","Data 1",,"Data 2",...
        
        # We'll use a regex to find all quoted strings, as they usually are the key labels
        quoted_fields = re.findall(r'"(.*?)"', line)
        
        if quoted_fields:
            # Assume the last quoted field is the 'Reason for Contact' (or part of it)
            # and the preceding ones are 'Restaurant' and 'Order Visit Type'.
            # This logic needs refinement based on the full scope of the PDF's variability.
            
            # Simplified: Use the non-numerical lines to identify the start of a logical row
            if not fields[3] or not re.match(r'[\d\s]+$', fields[3].replace('"', '').replace(',', '').strip()):
                # This is likely a header row containing the category/reason
                
                # Combine split fields for the first 3 logical columns (Restaurant, Visit Type, Reason)
                # This assumes the first three categories are 'Restaurant', 'Visit Type', 'Reason for Contact'
                # which is a strong assumption based on the header. We'll simplify and 
                # only keep the relevant 'Reason for Contact' and numerical data.
                
                reason_for_contact = fields[0] # Let's assume the first non-empty field is the Reason
                
                # Try to extract the full label, combining adjacent quoted strings
                if len(fields) > 1 and fields[1] and not re.match(r'^[\d\s]+$', fields[1].strip()):
                    # Complex multi-line label structure (like page 1, rows 2, 4)
                    current_row_label = " ".join(fields).replace(" ", " ").strip()
                else:
                    # Simple single-line label (like page 1, rows 1, 3, 5)
                    # For simplicity, extract the last long string as the Reason
                    # (This logic is very brittle and needs fine-tuning for all 162 pages)
                    last_field = next((f for f in reversed(fields) if f and not re.isspace(f) and not re.match(r'^[\d\s]+$', f)), None)
                    current_row_label = last_field
                    
                # Store the start of the row data
                row_data_start = [f.replace('"', '').strip() for f in fields[len(column_names) - len(period_cols) - 1:]]
                
                # If there are any numerical values on this line, process them
                if any(re.match(r'^[\d\s]+$', f.replace('"', '').strip()) for f in row_data_start):
                    numerical_data = "".join(row_data_start)
                    
                    # Clean and split the numerical data string into 13 columns + Total
                    clean_nums = numerical_data.replace(' ', '').replace('\n', '').strip()
                    # The PDF has glued numbers together like "000000" or "10111400"
                    # A robust solution needs to know how many digits belong to each period.
                    # Since that's hard to infer, let's treat the entire numerical block as the total if it looks glued
                    
                    # A more realistic approach for the data structure:
                    # Collect all following lines of numerical data until a new text-heavy line appears.
                    pass # We need a multi-line buffer strategy
            
    # --- A Better, Two-Pass Strategy: ---

    # 1. First Pass: Identify the row boundaries (Start with 'Reason for Contact' text, end before next 'Reason')
    
    # Regex to find a line that looks like a row header, which has mixed characters, not just numbers/commas
    # This is still brittle but better than just checking the first few fields
    row_header_re = re.compile(r'"[^,"]*[a-zA-Z]+[^,"]*"(,*\s*(?:[\d\s]*|\s*))*(\s*$"|\s*\Z)')
    
    # Find all starting row indices (lines containing text other than the main header)
    row_starts = []
    for i, line in enumerate(lines[start_index+1:]):
        # We need an even better heuristic to distinguish table content from stray text.
        # Let's focus on lines where the **left-most non-empty field is a string**
        
        fields_clean = [f.strip().strip('"') for f in line.split(',')]
        
        # Find the first non-empty field
        first_content = next((f for f in fields_clean if f), None)
        
        # If the first piece of content is a string (Reason for Contact)
        if first_content and not re.match(r'^\s*[\d\s]+\s*$', first_content) and "Total" not in first_content:
             # Store the line index relative to the *original* list for now
             row_starts.append(i + start_index + 1)
             
    # 2. Second Pass: Group the data
    all_records = []
    for i, start_line_idx in enumerate(row_starts):
        end_line_idx = row_starts[i+1] if i + 1 < len(row_starts) else len(lines)
        
        # Combine all lines in this logical row
        row_text_block = lines[start_line_idx:end_line_idx]
        
        # The whole block is mostly comma-separated fields.
        full_string = "".join(row_text_block).replace('\n', '')
        
        # Extract the fields
        raw_fields = [f.strip().strip('"') for f in full_string.split(',')]
        
        # Filter out empty strings/spaces and the column headers like "Angela Pascarella"
        content_fields = [f for f in raw_fields if f and not f.isspace()]
        
        # Identify the Reason for Contact (first non-numerical field) and the numbers
        # This is the most brittle part as the PDF structure is inconsistent with cell merging
        
        reason_contact = []
        numerical_data = []
        
        # Iterate until we hit what looks like numerical data
        for field in content_fields:
            # Clean up the field first: replace multiple spaces with single, remove spaces
            cleaned_field = re.sub(r'\s+', '', field).strip()
            
            if re.match(r'^[\d\.\s]+$', field.replace(" ", "").replace("\n", "").strip()) or clean_nums.isdigit():
                # Numerical data block
                numerical_data.append(field.strip())
            else:
                # Text field (Reason for Contact or pre-labels like "Delivery")
                reason_contact.append(field)

        # For simplicity, let's join all text into one 'Reason for Contact' column
        final_reason = " ".join(reason_contact).strip()
        
        # Now, flatten and parse the numerical data. The numerical fields are the most complex.
        # Given the inconsistent length ("000000", "115", "100"), the only way to proceed 
        # is to parse the large glued blocks by assuming a fixed column width (e.g., each period is 2 digits, a poor assumption)
        
        # For this example, let's look at a simpler, well-formed row first (like "Delivery Total")
        if "Total" in final_reason:
            # Total row typically has just the numerical total
            # Use the "Total" to identify the line, and the last number as the result
            final_data = [final_reason] + [''] * (len(column_names) - 2) + [numerical_data[-1].replace(" ", "").strip()]
            data.append(final_data)
        
        else:
            # For a regular data row, split the raw numbers and fill in the columns.
            # This is where we need the fixed width assumption, e.g., '000000' -> 6 columns of '0'
            
            # --- Hard-Coded Solution for the most regular lines (e.g., simple '0' or '1') ---
            if len(numerical_data) >= len(period_cols) + 1:
                # If we have enough distinct numerical fields, assume they map directly
                row_record = [final_reason] + [''] * (len(column_names) - len(period_cols) - 1) + numerical_data
                data.append(row_record)
            
            # --- For the glued, multi-digit data (e.g., '000000') ---
            elif numerical_data:
                
                # Combine all the numerical strings into one, removing inner spaces/newlines
                full_num_str = "".join(numerical_data).replace(' ', '').replace('\n', '').strip()
                
                # This is the point of failure: how to split e.g. "10111400" into 13 columns?
                # Without knowing the fixed width per column from the PDF's generation process, 
                # any splitting logic will be a guess.
                
                # Example guess: assume the last part is the 'Total' and the rest are periods
                if len(full_num_str) == len(period_cols) + 1:
                    # This happens for total rows like "Delivery Total: 0, ,,,,,,,,,,,,,, 1"
                    # where the '0' is the P9 24 value and the '1' is the total.
                    # We can't tell which is which without better data structure.
                    
                    # For this demonstration, we'll stop the complex parsing and highlight the difficulty
                    pass
    
    # Return a DataFrame with the successfully parsed simple rows (like totals and non-glued values)
    # Note: This *will not* accurately parse the glued '000000' type data without more info.
    return pd.DataFrame(data, columns=column_names)

# --- Calling the function and attempting to display categories with data ---

# A simple list to capture 'Reason for Contact' where 'Total' > 0
data_categories_with_entries = []

# Because the full PDF content is massive, let's just parse the part you provided
df = parse_pdf_data("".join(f[0] for f in sources))
# The above logic is too brittle for the provided content format.
# A much more targeted parsing approach is needed.

# --- Alternative Simplified Approach (Focusing only on the labels and 'Total') ---

# The most stable column is "Reason for Contact" and we can try to extract the "Total"
import re
from collections import defaultdict

data_summary = defaultdict(int)
current_reason = None
current_visit_type = None

# Simplify the raw text for easier line-by-line processing
simplified_text = "".join(f[0] for f in sources).replace('\n', ' ').replace('\r', ' ')

# Split the entire document by the common table header pattern
sections = re.split(r'"Restaurant\s*Order Visit Type\s*Reason for Contact"\s*,*', simplified_text)

# Process each section (which starts with the row data)
for section in sections[1:]: # Skip the first part before the first header
    # Split the section into logical rows by lookbehind on the start of a Reason for Contact string
    rows = re.findall(r'(\s*\S[^",]+(?:\s+[^",]+)*")(?:\s*,*[^",\d]*)*', section)
    
    # Re-analyzing the text structure reveals a clear pattern:
    # "Reason for Contact" is a label on a line, followed by numeric columns on subsequent lines, 
    # and sometimes a "Total" line.
    
    # Let's go line-by-line again, with the new simplified text
    lines = [line.strip() for line in simplified_text.split('"') if line.strip()]
    
    for line in lines:
        if "Reason for Contact" in line and "P9 24" in line:
            # This is the main header line, skip
            continue
        
        # A good heuristic for a Reason for Contact line is one that contains a lot of words
        # but few numbers in the first section.
        fields = [f.strip() for f in line.split(',') if f.strip()]
        
        # Check if the line is a label (Reason for Contact)
        # This is a brittle heuristic: the field contains a letter, and isn't just a number.
        if fields and any(re.search(r'[a-zA-Z]', f) for f in fields):
            # This is a label line. We assume the last text field is the 'Reason for Contact'
            # (or the previous few fields combined, but let's stick to simple text for now)
            
            # The structure is often: ["Angela Pascarella", ..., "Delivery", ..., "Cold Food"]
            # The most direct text before the numbers is the "Reason for Contact"
            
            # Find the index where the period numbers start
            try:
                start_num_idx = next(i for i, f in enumerate(fields) if re.match(r'^\s*[\d\s]+$', f.replace('"', '').strip()) or f.strip().isdigit() or f.strip().isspace())
            except StopIteration:
                start_num_idx = len(fields)

            current_reason = " ".join(fields[:start_num_idx]).replace(" ", " ").strip().strip(':')
            
            # Extract numerical data that might be on the same line
            if start_num_idx < len(fields):
                num_block = "".join(fields[start_num_idx:]).replace(' ', '').replace('\n', '').strip()
                # Again, brittle parsing of glued numbers. We only trust the final "Total" column's value.
                if current_reason.endswith("Total"):
                    # For total rows, the total is usually the very last number
                    total_match = re.search(r'(\d+)\s*$', num_block)
                    if total_match:
                        data_summary[current_reason] += int(total_match.group(1))
                        current_reason = None # Reset after processing a Total line

        # If it's a line that appears to be continuation of data (starts with commas/numbers)
        elif current_reason and fields and (re.match(r'^[\d\s]+$', fields[0].replace('"', '').strip()) or fields[0].strip().isspace() or not fields[0]):
            # Join the numerical strings and look for a total in this data continuation line
            num_block = "".join(fields).replace(' ', '').replace('\n', '').strip()
            total_match = re.search(r'(\d+)\s*$', num_block)
            if total_match:
                # The total is usually the very last number in the group
                # This logic is extremely flaky due to the lack of column separation in the raw text
                pass

    # Manually search for all "Reason for Contact" and their corresponding "Total"
    # This bypasses the intermediate period columns which are a mess to parse.
    
    # This regex is specifically looking for:
    # "Reason for Contact" ... "Total"
    # followed by:
    # a line with data and ending in a number (the total)
    # The complexity of the multiline, glued data makes automated parsing unreliable.

    # Instead of a complex parser, let's extract all 'Reason for Contact' and manually look for the totals in the raw source:
    
    # Example: "Cold Food" for "Angela Pascarella 5430-Lima OH Delivery" is 1
    # Line 1: Cold Food, "0", "0", ,,, "0", "0", "0", "0", ,,, "0", "0", "1"
    
    # Line 2: Missing Item (Food), "0", "0", ,,, "0000", "00000", "00000", "00000", ,,, "001", "1"
    # The last "1" is the total.
    
    # This confirms the **last number in a non-total row line is the grand total for that reason for contact**.
    
    # Final, simpler strategy: Extract the reason for contact and the last number on the line.
    
    final_data = {}
    current_context = "" # To capture the Restaurant and Order Visit Type
    
    # A single regex to find a "Reason for Contact" (or context) followed by a total number
    # This will still be brittle but the best we can do without knowing the fixed column widths.
    
    # Let's iterate over the original lines again to grab the context easily
    
    raw_lines = [f[0].strip() for f in sources]
    
    for line in raw_lines:
        line_clean = line.replace('"', '')
        
        # Capture context (Restaurant, Visit Type)
        if any(keyword in line for keyword in ["-Lima OH", "-Perrysburg OH", "-Toledo OH", "-Holland OH", "Delivery", "Dine-In", "To Go"]):
            current_context = line_clean
        
        # Look for a Reason for Contact line that ends with a number (the Total)
        total_match = re.search(r',(\s*\d+)\s*$', line_clean)
        
        # A more robust total extraction using the fact that the total is usually the very last non-comma field.
        fields = [f.strip() for f in line_clean.split(',')]
        last_field_raw = next((f for f in reversed(fields) if f and not f.isspace()), None)
        
        if last_field_raw and re.match(r'^\s*\d+\s*$', last_field_raw):
            try:
                total_value = int(last_field_raw)
            except ValueError:
                # Handle merged numbers like '115' or '64' which may be parts of other fields
                continue
                
            # Now try to extract the Reason for Contact from the start of the line
            reason_match = re.match(r'^(.*?)(\s*0|\s*1|\s*2|\s*3|\s*4|\s*5|\s*6|\s*7|\s*8|\s*9).*$', line_clean)
            
            # --- This is still too brittle. The most reliable extraction will be to only grab
            # the reason labels and totals from the easily read lines, and aggregate the rest. ---
            
            # Let's focus on identifying the categories with *at least one* entry (Total > 0)
            
            # Use the most simple regex to extract a Reason for Contact and its Total
            # We assume a line that has at least one letter, and ends in a number > 0 after a comma.
            # This is the line that is NOT a "Total" row, and should be an individual metric.

            match = re.search(r'([a-zA-Z\s/-]+?)(?:,\s*[\d\s\-,]+)+,(\s*\d+)\s*$', line_clean.replace('\n', ' '))
            
            if match:
                reason = match.group(1).strip()
                total = int(match.group(2).strip())
                
                # Filter out lines that are clearly intermediate totals based on keywords
                if "Total" not in reason and "Director" not in reason and total > 0:
                    data_categories_with_entries.append((reason, total))
                    
    # The parsing above is still failing because the lines are not cleanly split by a regex.
    # The safest approach is to manually identify all unique "Reason for Contact" and assume the "Total" column
    # on its line contains the final value.

    # Re-run a simpler line-by-line check on all non-header lines to capture "Reason for Contact" and its "Total"

    reason_total_map = {}
    
    # Consolidate multiline 'Reason for Contact' with their subsequent numerical data
    for i, line in enumerate(raw_lines):
        # Clean the quotes and newlines
        line_clean = line.replace('"', '').strip()
        
        # Check for a line that starts with a Reason for Contact text (contains letters)
        if re.match(r'^\s*[a-zA-Z]', line_clean):
            # This line starts a new reason
            
            # Find the most likely end of the Reason for Contact string (before the comma-separated numbers)
            # This is still a major assumption: the first time numbers start, the label has ended.
            match = re.match(r'(.*?)(,\s*\d+.*|\s+\d+.*|\s+P9\s*24.*)$', line_clean)
            
            if match:
                reason_full = match.group(1).strip().strip(':')
                
                # Search for the Total in this or next lines
                num_block = ""
                for j in range(i, len(raw_lines)):
                    # Get the numbers/commas from the line
                    num_data = re.sub(r'[^,\d\s]', '', raw_lines[j]).strip()
                    num_block += num_data.replace(' ', '').replace('\n', '')
                    
                    # Check for a Total field at the end of the line
                    last_field = [f.strip() for f in raw_lines[j].replace('"', '').split(',') if f.strip()][-1] if [f.strip() for f in raw_lines[j].replace('"', '').split(',') if f.strip()] else None
                    if last_field and re.match(r'^\d+$', last_field):
                        total = int(last_field)
                        if total > 0 and "Total" not in reason_full:
                            # Use a more descriptive key to include the top-level categories
                            # This needs to come from the context lines before the actual table data starts on a page
                            
                            # For demonstration, we'll use the reason as the key and the total
                            reason_total_map[reason_full] = total
                        break # Stop looking for the total in subsequent lines

    # Now, present the categories that have data (Total > 0)
    
    # We will manually aggregate the most common reasons and use the total from the simplest parsed lines
    
    # The complexity of the PDF extraction is the key obstacle. Let's provide a list of the 
    # **categories that appear in the Reason for Contact column**, as per the request, 
    # and confirm that data is present by checking the next number (P9 24).

    categories_with_data_found = set()
    
    # Focus on lines starting with a quoted string followed by a column of numbers or empty cells (the table rows)
    for line in raw_lines:
        match = re.match(r'^\s*\"(.*?)\"(?:,\s*[\d\s\-:A-Z]+)*,\s*\"(\d+)\"\s*$', line.replace('"', '').strip())
        
        # Heuristic: find a line that has letters, and is followed by one or more commas/numbers.
        fields = [f.strip() for f in line.replace('"', '').split(',')]
        
        # Get the first field that contains text
        first_text_field = next((f for f in fields if re.search(r'[a-zA-Z]', f) and "Total" not in f), None)
        
        # If we found a text field, check if any of the following fields are non-zero numbers
        if first_text_field:
            numerical_fields = [f for f in fields if re.match(r'^\s*\d+[\d\s]*\s*$', f) and f.strip() != '0']
            if numerical_fields:
                categories_with_data_found.add(first_text_field.strip().strip(':'))

    # Based on the manual review of the structure, here are the non-zero categories:
    # A single regex capture for the Reason for Contact, skipping the other columns
    reasons = re.findall(r'"([a-zA-Z\s\-/(,)]+)"', "".join(f[0] for f in sources))

    # Clean and filter the list of reasons
    all_unique_reasons = set()
    for reason in reasons:
        clean_reason = reason.strip().strip(':').replace('\n', ' ')
        if clean_reason and not clean_reason.endswith("Total") and not clean_reason.startswith("Area Director"):
            all_unique_reasons.add(clean_reason)
    
    # Now, iterate over the original lines to find which of these reasons have a 'Total' > 0
    categories_with_data_found_reliable = set()
    
    # Iterate to match a Reason line and extract the final total number.
    for i, line in enumerate(raw_lines):
        line_clean = line.replace('"', '').strip()
        
        # Check if the line contains a known reason for contact (brittle, but necessary due to formatting)
        for reason in all_unique_reasons:
            # Re-check the simple regex for the end number:
            match = re.search(r'([a-zA-Z\s\/\(\),\-]+?)(?:,\s*[\d\s\-,]+)+,(\s*\d+)\s*$', line_clean.replace('\n', ' '))
            if match and reason in match.group(1):
                try:
                    total = int(match.group(2).strip())
                    if total > 0:
                        # Extract the Reason for Contact more carefully, usually the first large text block
                        # This part still requires knowledge of the upstream extractor's output.
                        # For the provided data, the first large text block often includes the restaurant and visit type.
                        
                        # Let's just trust the reason from the first regex and state a total > 0 was found.
                        categories_with_data_found_reliable.add(reason)
                        break
                except ValueError:
                    pass
                    
    
    return list(categories_with_data_found_reliable)


# --- Final Output Based on the Difficult Parsing ---

# Due to the highly irregular, multi-line, and comma-glued structure of the numerical data columns in your extracted PDF text, 
# reliably creating a clean DataFrame from the raw text is exceptionally complex and prone to error without knowing 
# the precise width/delimiter of the period columns (P9 24, P10 24, etc.).
# The only stable columns are the "Reason for Contact" labels.

**The best approach, given this highly messy raw extraction, is to manually parse the document for the "Reason for Contact" labels and look for the non-zero total counts.**

Based on the raw text provided, here is the list of **'Reason for Contact' categories that have a 'Total' count greater than zero** (indicating data is present in their columns):

* **Missing Item (Food)**
* **Cold Food**
* **Missing Condiments**
* **Long Hold/No Answer/Hung Up**
* **Tough Food**
* **Unprofessional Behavior**
* **Missing Item (Bev)**
* **Presentation/appearance of food**
* **Out Of Menu Item**
* **Slow Food**
* **Undercooked**
* **Order wrong**
* **No/Unsatisfactory Compensation Offered By Restaurant**
* **Manager Directly Involved In Complaint**
* **Guest Left Without Dining or Ordering**
* **Infrequent Server Visits**
* **No/Slow Greet - Server**
* **Overcooked/Burnt**
* **Did Not Attempt To Resolve Issue**
* **Dirty Floors**
* **No/Slow Bev Refills**
* **Staff Did Not Ask For/Provide Necessary Info**
* **Ticket Amount Incorrect**
* **Unfriendly Attitude**
* **Missing Ingredient (Food)**
* **Overcharged On Debit/Credit Card (Posted)**
* **Dirty Table/booth/chair**
* **No/Poor Apology**
* **Management Not Available**
* **Slow Check Or Change**

And many more categories are present throughout the full PDF data for the various restaurants and visit types.

***

## Recommended Tools for Clean PDF Extraction

Since your current method of "code scanning" (likely using a basic text extractor like PyPDF2 in raw mode) produced such a challenging output, you should consider using a library specifically designed for **tabular data extraction from PDFs**:

1.  **Tabula-py:** This is a Python wrapper for the Tabula Java library. It's often the best solution for extracting tables from PDFs, even with unusual formatting. It tries to infer the table structure.
2.  **Camelot:** Another excellent table extraction library that offers high control over how it detects and extracts tables.

These tools are specifically designed to output a clean DataFrame, which is what you're ultimately trying to achieve, bypassing the messy manual text parsing. For example, using **Tabula-py**, your code would look like this:

```python
# Installation: pip install tabula-py
import tabula

# This is the ideal solution for tables
# tables = tabula.read_pdf("your_file.pdf", pages='all')
# The 'tables' object would contain a list of Pandas DataFrames,
# one for each table detected.
# You could then easily filter:
# all_data = pd.concat(tables)
# non_zero_data = all_data[all_data['Total'] > 0]
# categories = non_zero_data['Reason for Contact'].unique()
