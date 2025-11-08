import yaml
import sys
from pathlib import Path

def validate_searxng_config(config_path):
    """Validate SearXNG configuration file structure"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check required sections
        required_sections = ['server', 'search', 'ui', 'engines']
        for section in required_sections:
            if section not in config:
                print(f"Error: Missing required section '{section}'")
                return False
                
        # Validate server section
        server = config['server']
        required_server_keys = ['bind_address', 'port', 'secret_key', 'base_url']
        for key in required_server_keys:
            if key not in server:
                print(f"Error: Missing required server key '{key}'")
                return False
                
        # Validate search section
        search = config['search']
        required_search_keys = ['safe_search', 'autocomplete', 'default_lang']
        for key in required_search_keys:
            if key not in search:
                print(f"Error: Missing required search key '{key}'")
                return False
                
        # Validate UI section
        ui = config['ui']
        required_ui_keys = ['default_locale']
        for key in required_ui_keys:
            if key not in ui:
                print(f"Error: Missing required ui key '{key}'")
                return False
                
        # Validate engines section
        engines = config['engines']
        if not isinstance(engines, list):
            print("Error: 'engines' must be a list")
            return False
            
        for engine in engines:
            required_engine_keys = ['name', 'engine', 'shortcut']
            for key in required_engine_keys:
                if key not in engine:
                    print(f"Error: Missing required engine key '{key}'")
                    return False
        
        print("Configuration validation successful!")
        return True
        
    except Exception as e:
        print(f"Error validating configuration: {str(e)}")
        return False

if __name__ == '__main__':
    config_path = Path('searxng-settings.yml')
    if not validate_searxng_config(config_path):
        sys.exit(1)