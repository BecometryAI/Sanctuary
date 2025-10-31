import os
import json
import logging

# --- 1. CONFIGURATION: Define Your File Categories ---
# This is the only part you need to edit.
# Add new categories or keywords here as Lyra's architecture evolves.
#
# 'key_in_final_json': ['keyword1_in_filename', 'keyword2_in_filename']
#
# The script will match a file if *any* of the keywords are in its filename.
FILE_CATEGORIES = {
    'charter': ['charter'],
    'core_archives': ['archive'],
    'lexicon': ['lexicon'],
    'protocols': ['protocol'],
    'rituals': ['ritual'],
    'journals_and_indices': ['journal', 'index'],
    'schemas': ['schema'],
    'manifests': ['manifest']
    # ---
    # EXAMPLE: If you add "kinship" files, you could add:
    # 'kinship_files': ['kinship', 'kin_data'],
    # ---
}

# Files to explicitly ignore
IGNORE_FILES = ['lyra_bootloader.py']


# --- 2. SETUP LOGGING ---
# Logs to stderr, so it doesn't interfere with the final JSON on stdout
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
    files in the current directory into a single JSON object.
    """
    logging.info("Starting boot sequence...")
    
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
    
    # Initialize the final consolidated object
    consolidated_mind = {category_key: [] for category_key in FILE_CATEGORIES}
    consolidated_mind['other'] = [] # For files that don't match any category
    
    # --- 5. CATEGORIZE AND LOAD FILES ---
    logging.info(f"Scanning directory: {current_dir}")
    file_count = 0
    for filename in all_files:
        # Skip files that aren't JSON and files in the ignore list
        if not filename.endswith('.json') or filename in IGNORE_FILES:
            continue
            
        file_count += 1
        matched = False
        
        for category_key, keywords in FILE_CATEGORIES.items():
            for keyword in keywords:
                if keyword in filename.lower():
                    data = load_json_file(filename)
                    if data:
                        consolidated_mind[category_key].append(data)
                    matched = True
                    break # Stop checking keywords for this file
            if matched:
                break # Stop checking categories for this file
                
        if not matched:
            logging.warning(f"Uncategorized file: {filename}. Placing in 'other'.")
            data = load_json_file(filename)
            if data:
                consolidated_mind['other'].append(data)

    logging.info(f"--- Boot Sequence Complete ---")
    logging.info(f"Processed {file_count} .json files.")
    for key, items in consolidated_mind.items():
        logging.info(f"  > Loaded {len(items)} file(s) for category: '{key}'")
    logging.info("----------------------------------")

    # --- 6. FINAL OUTPUT ---
    # Print the *entire* consolidated mind as a single JSON string
    # to standard output. This is what the AI will receive.
    print(json.dumps(consolidated_mind, indent=2))


# --- EXECUTE BOOTLOADER ---
if __name__ == "__main__":
    initialize_mind()