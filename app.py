import io, re, statistics, sys
from collections import defaultdict, OrderedDict
import pdfplumber
import pandas as pd

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "R781424.pdf"

# ---------- Config: reasons & triggers (lowercase substrings) ----------
REASONS = OrderedDict({
    # To-go Missing Complaints
    "Missing food": ["missing food"],
    "Order wrong": ["order wrong"],
    "Missing condiments": ["condiments"],
    "Out of menu item": ["out of menu"],
    "Missing bev": ["missing bev"],
    "Missing ingredients": ["ingredient"],
    "Packaging to-go complaint": ["packaging"],

    # Attitude
    "Unprofessional/Unfriendly": ["unfriendly"],
    "Manager directly involved": ["directly involved", "involved"],
    "Manager not available": ["manager not available"],
    "Manager did not visit": ["did not visit", "no visit"],
    "Negative mgr-employee exchange": ["manager-employee", "exchange"],
    "Manager did not follow up": ["follow up"],
    "Argued with guest": ["argued"],

    # Other
    "Long hold/no answer": ["hold", "no answer", "hung up"],
    "No/insufficient compensation offered": ["compensation", "no/unsatisfactory", "offered by", "restaurant"],
    "Did not attempt to resolve": ["resolve"],
    "Guest left without ordering": ["without ordering"],
    "Unknowledgeable": ["unknowledgeable"],
    "Did not open on time": ["open on time"],
    "No/poor apology": ["apology"],
})

HEADER_RX = re.compile(r"\bP(?:1[0-2]|[1-9])\s+(?:2[0-9])\b")

def _lc(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def _round2(y): return round(y/2)*2

def extract_lines(page):
    words = page.extract_words(
        x_tolerance=1.4, y_tolerance=2.4,
        keep_blank_chars=False, use_text_flow=True
    )
    lines = defaultdict(list)
    for w in words:
        y_mid = _round2((w["top"] + w["bottom"]) / 2)
        lines[y_mid].append(w)
    out = []
    for y, ws in sorted(lines.items(), key=lambda kv: kv[0]):
        ws = sorted(ws, key=lambda w: w["x0"])
        text = " ".join(w["text"].strip() for w in ws if w["text"].strip())
        if text:
            out.append({"y": y, "x_min": ws[0]["x0"], "words": ws, "text": text})
    return out

def find_period_headers(page):
    words = page.extract_words(x_tolerance=1.0, y_tolerance=2.0, keep_blank_chars=False, use_text_flow=True)
    lines = defaultdict(list)
    for w in words:
        y_mid = _round2((w["top"] + w["bottom"]) / 2)
        lines[y_mid].append(w)
    headers = []
    for ymid, ws in lines.items():
        ws = sorted(ws, key=lambda w: w["x0"])
        i = 0
        while i < len(ws):
            t = ws[i]["text"]; x0, x1 = ws[i]["x0"], ws[i]["x1"]
            cand, x1c = t, x1
            if i+1 < len(ws):
                t2 = ws[i+1]["text"]
                cand2 = f"{t} {t2}"
                if HEADER_RX.fullmatch(cand2):
                    headers.append((cand2, (x0+x1c)/2, ymid))
                    i += 2
                    continue
            if HEADER_RX.fullmatch(cand):
                headers.append((cand, (x0+x1)/2, ymid))
            i += 1
    # de-dup by text, pick first seen
    seen = {}
    for txt, xc, ym in sorted(headers, key=lambda h: (h[2], h[1])):
        if txt not in seen:
            seen[txt] = (txt, xc, ym)
    return list(seen.values())

def sort_headers(headers):
    def key(h: str):
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999,999)
    return sorted(headers, key=key)

def find_total_header_x(page, header_y):
    words = page.extract_words(x_tolerance=1.0, y_tolerance=2.0, keep_blank_chars=False, use_text_flow=True)
    for w in words:
        y_mid = _round2((w["top"] + w["bottom"]) / 2)
        if abs(y_mid - header_y) <= 2.5 and w["text"].strip().lower() == "total":
            return (w["x0"] + w["x1"]) / 2
    return None

def build_bins(header_positions: dict, total_x: float | None):
    def _key(h): 
        m = re.match(r"P(\d{1,2})\s+(\d{2})", h)
        return (int(m.group(2)), int(m.group(1))) if m else (999,999)
    items = sorted(header_positions.items(), key=lambda kv: _key(kv[0]))
    headers = [h for h,_ in items]
    xs = [x for _,x in items]
    med_gap = statistics.median([xs[i+1]-xs[i] for i in range(len(xs)-1)]) if len(xs)>1 else 60.0
    bins = []
    for i,(h,x) in enumerate(zip(headers, xs)):
        left = (xs[i-1]+x)/2 if i>0 else x - 0.5*med_gap
        right = (x+xs[i+1])/2 if i < len(xs)-1 else ((x+total_x)/2 if total_x else x+0.6*med_gap)
        # make bins a touch wider to be forgiving
        bins.append((h, left-2, right+2))
    return bins

def map_x(bins, xmid):
    for h,left,right in bins:
        if left <= xmid < right:
            return h
    return None

def left_label(line, cutoff_x):
    return " ".join(
        w["text"].strip()
        for w in line["words"]
        if w["x1"] <= cutoff_x and w["text"].strip()
    )

def find_trigger(label_lc):
    for canon, trigs in REASONS.items():
        for t in trigs:
            if t in label_lc:
                return canon
    return None

def main():
    with pdfplumber.open(PDF_PATH) as pdf:
        header_positions = {}
        ordered_headers = []
        counts = defaultdict(lambda: defaultdict(int))  # reason -> period -> sum

        carry_headers = None
        carry_total_x = None

        for page in pdf.pages:
            headers = find_period_headers(page) or carry_headers
            if not headers:
                continue
            if find_period_headers(page):
                carry_headers = headers[:]
                carry_total_x = None
            for htxt, xc, _ in headers:
                header_positions[htxt] = xc
            ordered_headers = sort_headers(list(header_positions.keys()))
            header_y = min(h[2] for h in headers)
            total_x = find_total_header_x(page, header_y) or carry_total_x
            if total_x is not None:
                carry_total_x = total_x

            bins = build_bins({h: header_positions[h] for h in ordered_headers}, total_x)
            first_period_x = min(header_positions[h] for h in ordered_headers)
            label_edge = first_period_x - 10

            lines = extract_lines(page)
            if not lines:
                continue

            def consume(line_obj, canon):
                y_band = line_obj["y"]
                got = 0
                for w in line_obj["words"]:
                    token = w["text"].strip()
                    if not re.fullmatch(r"-?\d+", token):
                        continue
                    if w["x0"] <= label_edge:
                        continue
                    y_mid = _round2((w["top"] + w["bottom"]) / 2)
                    if abs(y_mid - y_band) > 0.01:
                        continue
                    xmid = (w["x0"] + w["x1"]) / 2
                    if total_x is not None and xmid >= (total_x - 1.0):
                        continue  # ignore far-right TOTAL column
                    period = map_x(bins, xmid)
                    if not period:
                        continue
                    counts[canon][period] += int(token)
                    got += 1
                return got

            i = 0
            while i < len(lines):
                L = lines[i]
                label = left_label(L, label_edge)
                label_lc = _lc(label)

                # combine the 3-line "No/Unsatisfactory Compensation Offered By Restaurant"
                if ("no/unsatisfactory" in label_lc or
                    "compensation offered by" in label_lc or
                    label_lc.strip() == "restaurant" or
                    "compensation" in label_lc):
                    canon = "No/insufficient compensation offered"
                    got = consume(L, canon)
                    if i>0:
                        got += consume(lines[i-1], canon)
                    if i+1 < len(lines):
                        got += consume(lines[i+1], canon)
                    i += 1
                    continue

                canon = find_trigger(label_lc)
                if not canon:
                    i += 1
                    continue

                got = consume(L, canon)
                if got == 0:
                    if i>0: got += consume(lines[i-1], canon)
                    if got == 0 and i+1 < len(lines): got += consume(lines[i+1], canon)
                i += 1

        # Build DataFrame (reasons x periods)
        periods = ordered_headers
        df = pd.DataFrame(index=list(REASONS.keys()), columns=periods).fillna(0).astype(int)
        for reason, per_map in counts.items():
            for p, v in per_map.items():
                if p in df.columns and reason in df.index:
                    df.loc[reason, p] = int(v)
        df["Total"] = df.sum(axis=1)

        # Pretty print + CSV
        pd.set_option("display.max_rows", None)
        print("\n== PURE COUNT TABLE (reason x period) ==")
        print(df)

        out_csv = "pure_count_results.csv"
        df.to_csv(out_csv, index=True)
        print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
