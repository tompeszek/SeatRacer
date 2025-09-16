from importlib import resources
from pathlib import Path
import pandas as pd
import seatracer
import os

class DataHandler:
    def __init__(self, folder_name='data'):
        # Get path to the data directory
        try:
            self.data_path = Path(resources.files('seatracer') / folder_name)
        except (AttributeError, ImportError):
            # Fallback for older Python versions or deployment issues
            self.data_path = Path(__file__).parent.parent / folder_name
    
    def get_available_datasets(self):
        """Return a list of available dataset files"""
        try:
            if self.data_path.exists():
                return [f.name for f in self.data_path.iterdir() if f.is_file() and f.suffix.lower() == '.csv']
            else:
                return []
        except Exception as e:
            print(f"Error accessing data directory: {e}")
            return []
    
    def load_dataset(self, filename):
        """Load a specific dataset"""
        try:
            # Convert filename to a Path object if it's a string
            if isinstance(filename, str):
                file_path = Path(filename)
            else:
                file_path = filename
                
            # Make sure we use the full path
            full_path = self.data_path / file_path.name
            
            # Add logic here to handle different file types
            if file_path.suffix.lower() == '.csv':
                print(f"Loading CSV from: {full_path}")
                return pd.read_csv(full_path)
            # Add other file type handlers as needed
        except Exception as e:
            print(f"Error loading dataset {filename}: {e}")
            return pd.DataFrame()
        
    def get_data_path(self):
        """Return the path to the data directory"""
        return self.data_path