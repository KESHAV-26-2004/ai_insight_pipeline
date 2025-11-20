# utils/text_utils.py
# kept minimal — add helpers if needed later

def label_to_canonical(label):
    if label is None: return None
    l = str(label).strip().lower()
    if l in ("positive","pos","p","+","+1","label_2","2"): return "positive"
    if l in ("neutral","neu","n","0","label_1","1"): return "neutral"
    if l in ("negative","neg","-","-1","label_0","0"): return "negative"
    return None
