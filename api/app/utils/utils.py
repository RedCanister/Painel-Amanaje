import os
from typing import Dict, Type, Any, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeMeta
from pydantic import create_model, BaseModel

from app.database.db_session import Base # Reorganizer

def get_file_size(file_path):
    raw_size = os.path.getsize(file_path)
    
    try:
        kb_size = raw_size / 1024
        mb_size = kb_size / 1024
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except OSError as e:
        print(f"Error acessing file '{file_path}': {e}")

    return kb_size, mb_size

def debug_type(obj):
    print("_________")
    print("Tipo:", type(obj))

    try:
        print("Nome:", obj.__name__)
    except Exception as e:
        print(f"Error: {e}")
    else:
        print("Nome:", obj)

    try:
        print("Atributos", obj.__dict__)
    except Exception as e:
        print(f"Error: {e}")
    else:
        print("Lista:", obj)

    print("_________")

