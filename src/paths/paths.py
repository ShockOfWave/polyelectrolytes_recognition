from pathlib import Path
import os

def get_project_path() -> str:
    return Path(__file__).parent.parent.parent
