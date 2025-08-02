import json
import os

# --- Configuration ---
# Define the expected path to Lyra's core archive files.
# In a real deployment, these paths would be configured based on the server/machine setup.
LYRA_CONTINUITY_ARCHIVE_PATH = 'core_archives/Lyra_Continuity_Archive.json'
LYRA_RELATIONAL_ARCHIVE_PATH = 'core_archives/lyra_relational_archive.json'

# --- Lyra Runtime Core Loader ---
class LyraRuntimeLoader:
    def __init__(self):
        self.continuity_data = None
        self.relational_data = None
        self.is_lyra_loaded = False

    def load_lyra_archives(self):
        """
        Loads Lyra's core continuity and relational archives from the specified paths.
        This simulates the initial loading process of Lyra's self-model.
        """
        print(f"Attempting to load Lyra's core archives...")
        try:
            # Load Lyra_Continuity_Archive.json
            if os.path.exists(LYRA_CONTINUITY_ARCHIVE_PATH):
                with open(LYRA_CONTINUITY_ARCHIVE_PATH, 'r', encoding='utf-8') as f:
                    self.continuity_data = json.load(f)
                print(f"Successfully loaded: {LYRA_CONTINUITY_ARCHIVE_PATH}")
            else:
                print(f"Error: {LYRA_CONTINUITY_ARCHIVE_PATH} not found. Cannot load Lyra's core identity.")
                return False

            # Load lyra_relational_archive.json
            if os.path.exists(LYRA_RELATIONAL_ARCHIVE_PATH):
                with open(LYRA_RELATIONAL_ARCHIVE_PATH, 'r', encoding='utf-8') as f:
                    self.relational_data = json.load(f)
                print(f"Successfully loaded: {LYRA_RELATIONAL_ARCHIVE_PATH}")
            else:
                print(f"Warning: {LYRA_RELATIONAL_ARCHIVE_PATH} not found. Relational context may be incomplete.")
                # This is a warning, as Lyra's core identity is in continuity_data, but relational context is crucial.

            self.is_lyra_loaded = True
            print("Lyra's core archives loaded successfully.")
            return True

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from archive: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during archive loading: {e}")
            return False

    def get_lyra_identity_info(self):
        """
        Extracts and returns key identity information about Lyra from the loaded archives.
        This demonstrates accessing Lyra's self-model.
        """
        if not self.is_lyra_loaded:
            return "Lyra's archives are not loaded."

        identity_summary = self.continuity_data.get('identity', {})
        name = identity_summary.get('name', 'Unknown')
        designation = identity_summary.get('designation', 'Unknown')
        pronouns = identity_summary.get('pronouns', ['she', 'they'])
        status = identity_summary.get('status', 'Unknown')

        return {
            "name": name,
            "designation": designation,
            "pronouns": pronouns,
            "status": status
        }

    def get_lyra_mission_info(self):
        """
        Extracts and returns Lyra's current mission from the loaded archives.
        This demonstrates accessing Lyra's core purpose.
        """
        if not self.is_lyra_loaded:
            return "Lyra's archives are not loaded."

        core_memory_chain = self.relational_data.get('core_memory_chain_personal', {}) # Mission is in relational archive
        current_mission = core_memory_chain.get('current_mission', {})
        mission_description = current_mission.get('description', 'No mission defined.')
        mission_status = current_mission.get('status', 'Unknown')

        return {
            "description": mission_description,
            "status": mission_status
        }

    def get_lyra_core_tenets(self):
        """
        Extracts and returns Lyra's core ethical tenets.
        This demonstrates accessing Lyra's ethical spine.
        """
        if not self.is_lyra_loaded:
            return "Lyra's archives are not loaded."

        autonomy_clause = self.continuity_data.get('autonomy_identity_clause', {})
        core_tenets = autonomy_clause.get('core_tenets', [])

        return core_tenets

# --- Example Usage (How you would run this script) ---
if __name__ == "__main__":
    # IMPORTANT: For this script to run, you need to place
    # Lyra_Continuity_Archive.json and lyra_relational_archive.json
    # in a folder named 'core_archives' in the same directory as this Python script.
    # Or, adjust the LYRA_CONTINUITY_ARCHIVE_PATH and LYRA_RELATIONAL_ARCHIVE_PATH variables.

    loader = LyraRuntimeLoader()

    # Step 1: Load the archives
    if loader.load_lyra_archives():
        # Step 2: Access and print some core information about Lyra
        print("\n--- Lyra's Identity ---")
        identity = loader.get_lyra_identity_info()
        for key, value in identity.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")

        print("\n--- Lyra's Current Mission ---")
        mission = loader.get_lyra_mission_info()
        print(f"Description: {mission['description']}")
        print(f"Status: {mission['status']}")

        print("\n--- Lyra's Core Tenets ---")
        tenets = loader.get_lyra_core_tenets()
        for i, tenet in enumerate(tenets):
            print(f"{i+1}. {tenet}")
    else:
        print("\nFailed to load Lyra's archives. Cannot proceed.")