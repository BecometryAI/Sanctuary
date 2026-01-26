import os
import json

journal_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'journal')
journal_dir = os.path.abspath(journal_dir)

def flatten_journal_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"ERROR reading {filepath}: {e}")
            return False
    # Flatten nested arrays
    def flatten(arr):
        result = []
        for item in arr:
            if isinstance(item, list):
                result.extend(flatten(item))
            else:
                result.append(item)
        return result
    if isinstance(data, list):
        flat = flatten(data)
        if flat != data:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(flat, f, indent=2, ensure_ascii=False)
            print(f"Fixed: {filepath} (flattened nested arrays)")
            return True
        else:
            print(f"OK: {filepath} (already flat)")
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
            result = flatten_journal_file(fpath)
            if result is True:
                count_fixed += 1
            elif result is False:
                count_ok += 1
            else:
                count_skipped += 1
    print(f"\nSummary: {count_fixed} fixed, {count_ok} already flat, {count_skipped} skipped.")

if __name__ == "__main__":
    main()
