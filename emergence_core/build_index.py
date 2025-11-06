import os
from lyra.librarian import LyraLibrarian

def main():
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    persist_dir = os.path.join(base_dir, 'model_cache', 'chroma_db')
    
    # Initialize Librarian
    librarian = LyraLibrarian(
        base_dir=base_dir,
        persist_dir=persist_dir
    )
    
    try:
        # Build the vector index
        librarian.build_index()
        print("\nIndex built successfully!")
        print(f"Vector store persisted at: {persist_dir}")
        
    except Exception as e:
        print(f"\nError building index: {str(e)}")
        raise

if __name__ == "__main__":
    main()