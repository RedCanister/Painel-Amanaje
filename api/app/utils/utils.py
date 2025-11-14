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

import inspect
from pydantic import BaseModel

def debug_type(obj):
    print("\n" + "-" * 40)
    print("🔍 Debugging object")

    # Tipo do objeto
    print("📦 Tipo:", type(obj))

    # Nome do objeto (se aplicável)
    try:
        obj_name = getattr(obj, '__name__', None) or obj.__class__.__name__
        print("🧩 Nome:", obj_name)
    except Exception as e:
        print("⚠️ Erro ao obter nome:", e)

    # Representação básica
    try:
        print("🪞 Representação:", repr(obj))
    except Exception as e:
        print("⚠️ Erro ao representar:", e)

    # Atributos __dict__ (se houver)
    try:
        if hasattr(obj, '__dict__'):
            print("📚 __dict__:", obj.__dict__)
    except Exception as e:
        print("⚠️ Erro ao acessar __dict__:", e)

    # Pydantic model_dump
    if isinstance(obj, BaseModel):
        try:
            print("🧬 Pydantic model_dump:", obj.model_dump())
        except Exception as e:
            print("⚠️ Erro ao usar model_dump:", e)

    # Listagem de atributos
    try:
        print("🔧 Atributos disponíveis:", dir(obj))
    except Exception as e:
        print("⚠️ Erro ao listar atributos:", e)

    # Inspeção de assinatura (se for função ou método)
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            sig = inspect.signature(obj)
            print("📝 Assinatura:", sig)
        except Exception as e:
            print("⚠️ Erro ao inspecionar assinatura:", e)

    # Inspeção de módulo
    if inspect.ismodule(obj):
        try:
            print("📦 Módulo:", obj.__file__)
        except Exception:
            print("📦 Módulo embutido ou sem __file__")

    print("-" * 40 + "\n")
