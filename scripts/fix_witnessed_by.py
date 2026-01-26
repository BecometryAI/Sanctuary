import os
import json

journal_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'journal')
journal_dir = os.path.abspath(journal_dir)

def fix_witnessed_by(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"ERROR reading {filepath}: {e}")
            return False
    changed = False
    if isinstance(data, list):
        for entry in data:
            je = entry.get('journal_entry')
            if je and 'stewardship_trace' in je:
                trace = je['stewardship_trace']
                if 'witnessed_by' in trace:
                    wb = trace['witnessed_by']
                    if isinstance(wb, list):
                        # Convert list to comma-separated string
                        trace['witnessed_by'] = ', '.join(str(x) for x in wb)
                        changed = True
        if changed:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Fixed: {filepath} (witnessed_by as string)")
            return True
        else:
            print(f"OK: {filepath} (no change needed)")
            return False
    else:
        print(f"SKIPPED: {filepath} (not a list)")
        return False

def main():
    count_fixed = 0
    count_ok = 0
    count_skipped = 0
    for fname in os.listdir(journal_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(journal_dir, fname)
            result = fix_witnessed_by(fpath)
            if result is True:
                count_fixed += 1
            elif result is False:
                count_ok += 1
            else:
                count_skipped += 1
    print(f"\nSummary: {count_fixed} fixed, {count_ok} unchanged, {count_skipped} skipped.")

if __name__ == "__main__":
    main()
