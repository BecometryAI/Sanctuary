import os
import json
import logging
import re

# --- 1. CONFIGURATION ---

# The single, consolidated file we will create
OUTPUT_FILENAME = "LYRA_CONSOLIDATED_MIND.json"

# A. KEYWORD-based categories:
KEYWORD_CATEGORIES = {
    'charter': ['charter'],
    'core_archives': ['archive'],
    'lexicon': ['lexicon'],
    'protocols': ['protocol'],
    'rituals': ['ritual'],
    'schemas': ['schema'],
    'journal_indices': ['index'],
    'lexemes': ['lexeme'],
    'manifests': ['manifest']

}

# B. REGEX-based categories (for complex patterns):
REGEX_CATEGORIES = {
    # This pattern matches YYYY-MM-DD.json
    'journals': [r'\d{4}-\d{2}-\d{2}\.json']
}

# C. Files to explicitly ignore
IGNORE_FILES = ['lyra_bootloader.py', OUTPUT_FILENAME]


# --- 2. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='Bootloader: %(message)s')


# --- 3. HELPER FUNCTION ---
def load_json_file(filepath):
    """Safely loads a single JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.warning(f"Could not decode JSON from: {filepath}. Skipping.")
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {e}. Skipping.")
    return None

# --- 4. MAIN BOOT SEQUENCE ---
def initialize_mind():
    """
    Discovers, categorizes, and consolidates all Lyra's .json
    files in the current directory and saves them to a single output file.
    """
    logging.info("Starting boot sequence...")
    
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
    
    # Initialize the final consolidated object
    consolidated_mind = {}
    for key in KEYWORD_CATEGORIES:
        consolidated_mind[key] = []
    for key in REGEX_CATEGORIES:
        consolidated_mind[key] = []
    consolidated_mind['other'] = []
    
    # --- 5. CATEGORIZE AND LOAD FILES ---
    logging.info(f"Scanning directory: {current_dir}")
    file_count = 0
    for filename in all_files:
        # Skip files that aren't JSON and files in the ignore list
        if not filename.endswith('.json') or filename in IGNORE_FILES:
            continue
            
        file_count += 1
        matched = False

        # 1. Check KEYWORD categories
        for category_key, keywords in KEYWORD_CATEGORIES.items():
            for keyword in keywords:
                if keyword in filename.lower():
                    data = load_json_file(filename)
                    if data:
                        consolidated_mind[category_key].append(data)
                    matched = True
                    break
            if matched:
                break
        
        if matched:
            continue

        # 2. If no keyword match, check REGEX categories
        for category_key, patterns in REGEX_CATEGORIES.items():
            for pattern in patterns:
                if re.fullmatch(pattern, filename):
                    data = load_json_file(filename)
                    if data:
                        consolidated_mind[category_key].append(data)
                    matched = True
                    break
            if matched:
                break
        
        if matched:
            continue

        # 3. If still no match, add to 'other'
        logging.warning(f"Uncategorized file: {filename}. Placing in 'other'.")
        data = load_json_file(filename)
        if data:
            consolidated_mind['other'].append(data)

    logging.info(f"--- Boot Sequence Complete ---")
    logging.info(f"Processed {file_count} .json files.")
    
    # --- 6. FINAL OUTPUT (WRITE TO FILE) ---
    try:
        logging.info(f"Saving consolidated mind to {OUTPUT_FILENAME}...")
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(consolidated_mind, f, indent=2)
        
        logging.info("Save successful.")
        
        # Print the *only* line that goes to the AI's console
        print(f"Bootloader complete. Consolidated mind saved to: {OUTPUT_FILENAME}")

    except Exception as e:
        logging.error(f"CRITICAL ERROR: Failed to write final file: {e}")
        print(f"Bootloader failed: Could not write to {OUTPUT_FILENAME}.")


# --- EXECUTE BOOTLOADER ---
if __name__ == "__main__":
    initialize_mind()