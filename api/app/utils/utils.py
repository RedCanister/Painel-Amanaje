import os
import inspect
import json

#import torch
#import tensorflow as tf

from typing import Dict, Type, Any, Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeMeta
from pydantic import create_model, BaseModel
from pydantic import BaseModel

import onnx

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

def get_file_extension(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    return ext

def recover_model_params(file_path):
    ext = get_file_extension(file_path)
    params = {}

    try:
        # # Keras/ Tensorflow
        # if ext in [".h5", ".keras"]:
        #     model = tf.keras.model.load_model(file_path)
        #     params["framework"] = "tensorflow/keras"
        #     params["layers"] = [layer.__class__.__name__ for layer in model.layers]
        #     params["input_shape"] = model.input_shape
        #     params["output_shape"] = model.output_shape

        # if ext in [".pt", ".pth"]:
        #     state_dict = torch.load(file_path, map_location="cpu")
        #     params["framework"] = "pytorch"
        #     params["keys"] = list(state_dict.keys())[:10]

        #     config_path = os.path.join(os.path.dirname(file_path), "config.json")
        #     if os.path.exists(config_path):
        #         with open(config_path) as f:
        #             params.update(json.load(f))
        
        # ONNX
        # elif ext == ".onnx":
        #     model = onnx.load(file_path)
        #     params["framework"] = "onnx"
        #     params["inputs"] = [inp.name for inp in model.graph.input]
        #     params["outputs"] = [out.name for out in model.graph.output]

        # JSON/YAML
        if ext == ".json":
            with open(file_path) as f:
                params = json.load(f)
            params["framework"] = "config"
        else:
            params["error"] = f"Unsupported file type: {ext}"
    
    except Exception as e:
        params["error"] = str(e)
    
    return params

def debug_type(obj):
    print("\n" + "-" * 40)
    print("🔍 Debugging object")

    # Tipo do objeto
    print("📦 Tipo:", type(obj))

    # Nome do objeto (se aplicável)
    try:
        obj_name = getattr(obj, '__name__', None) or obj.__class__.__name__
        print("🧩 Nome:", obj_name, "\n")
    except Exception as e:
        print("⚠️ Erro ao obter nome:", e)

    # Representação básica
    try:
        print("🪞 Representação:", repr(obj), "\n")
    except Exception as e:
        print("⚠️ Erro ao representar:", e)

    # Atributos __dict__ (se houver)
    try:
        if hasattr(obj, '__dict__'):
            print("📚 __dict__:", obj.__dict__, "\n")
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
