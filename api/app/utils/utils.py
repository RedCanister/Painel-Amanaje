import os
import inspect
import aiofiles
import json

from typing import Dict, Type, Any, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeMeta
from pydantic import create_model, BaseModel

from app.database.db_session import Base # Reorganizer

# File functions
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

# CHANGED: Helper function to save file to disk
async def save_file_to_disk(file, dest_path: str) -> int:
    """Save file and return size in MB"""
    contents = await file.read()
    async with aiofiles.open(dest_path, 'wb') as out_f:
        await out_f.write(contents)
    return round(len(contents) / (1024 * 1024), 4)

def get_file_extension(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    return ext

def recover_model_params(file_path):
    ext = get_file_extension(file_path)
    params = {}

    try:
        # CHANGED: Support for JSON/YAML config files
        # Future extensions can add support for different model formats:
        # - Keras/TensorFlow: .h5, .keras models
        # - PyTorch: .pt, .pth model state dicts with optional config.json
        # - ONNX: .onnx models with input/output inspection
        if ext == ".json":
            with open(file_path) as f:
                params = json.load(f)
            params["framework"] = "config"
        else:
            params["error"] = f"Unsupported file type: {ext}"
    
    except Exception as e:
        params["error"] = str(e)
    
    return params


# Debug function
def _get_object_name(obj):
    """CHANGED: Helper to extract object name safely"""
    try:
        return getattr(obj, '__name__', None) or obj.__class__.__name__
    except Exception:
        return "Unknown"

def _get_object_repr(obj):
    """CHANGED: Helper to get object representation safely"""
    try:
        return repr(obj)
    except Exception:
        return "Error getting representation"

def _get_object_dict(obj):
    """CHANGED: Helper to get __dict__ safely"""
    try:
        if hasattr(obj, '__dict__'):
            return obj.__dict__
    except Exception:
        pass
    return None

def _get_pydantic_dump(obj):
    """CHANGED: Helper to dump Pydantic model safely"""
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump()
        except Exception:
            return None
    return None

def _get_object_attributes(obj):
    """CHANGED: Helper to get object attributes safely"""
    try:
        return dir(obj)
    except Exception:
        return []

def _get_function_signature(obj):
    """CHANGED: Helper to get function signature safely"""
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            return inspect.signature(obj)
        except Exception:
            pass
    return None

def _get_module_file(obj):
    """CHANGED: Helper to get module file safely"""
    if inspect.ismodule(obj):
        try:
            return obj.__file__
        except Exception:
            return "Built-in or no __file__"
    return None

def debug_type(obj):
    """
    CHANGED: Refactored debug function to reduce cognitive complexity
    by extracting helper functions for each inspection type
    
    Purpose: Print comprehensive debugging information about any Python object
    Displays:
    - Type and class name
    - String representation
    - Instance attributes (__dict__)
    - Pydantic model dump (if applicable)
    - Available attributes/methods
    - Function signature (if function/method)
    - Module information (if module)
    
    This is useful for development/debugging to understand object structure
    """
    print("\n" + "-" * 40)
    print("🔍 Debugging object")

    # Object type
    print("📦 Type:", type(obj))

    # Object name
    obj_name = _get_object_name(obj)
    print("🧩 Name:", obj_name, "\n")

    # Object representation
    obj_repr = _get_object_repr(obj)
    print("🪞 Representation:", obj_repr, "\n")

    # Object __dict__
    obj_dict = _get_object_dict(obj)
    if obj_dict:
        print("📚 __dict__:", obj_dict, "\n")

    # Pydantic dump
    pydantic_dump = _get_pydantic_dump(obj)
    if pydantic_dump:
        print("🧬 Pydantic model_dump:", pydantic_dump, "\n")

    # Available attributes
    attrs = _get_object_attributes(obj)
    if attrs:
        print("🔧 Attributes available:", len(attrs), "items")

    # Function signature
    sig = _get_function_signature(obj)
    if sig:
        print("📝 Signature:", sig)

    # Module file
    mod_file = _get_module_file(obj)
    if mod_file:
        print("📦 Module:", mod_file)

    print("-" * 40 + "\n")
