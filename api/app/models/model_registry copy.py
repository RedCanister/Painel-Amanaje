from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from typing import Dict, Any, Optional, Type, List
from pydantic import create_model, BaseModel
from datetime import datetime

#from app.models.model_objects import ObjectModel

from app.database.db_session import get_db, Base
from app.database.db_utils import create_entry, get_entry, get_all_entries, update_entry, delete_entry

from app.utils.utils import debug_type

SQLA_TYPE_MAP = {
        str: String,
        int: Integer,
        float: Float,
        bool: Boolean,
        dict: JSONB,
        list: JSONB,
        datetime: DateTime
        # Tipos de schema de outros modelos
}

# Conjunto de operações para o registro de modelos pydantic como objeto relacional mapeado no banco de dados postgres
# O registro é uma estrutura que aceita apenas as entradas em formato de dicionário para relacionar os modelos pydantic com os ORMs
class ModelRegistry:
    _registry: Dict[Type[BaseModel], Type[DeclarativeMeta]] = {}

    # Registrando e resgatando modelos de dados pydantic para ORM no banco de dados postgres com SQLAlchemy

    @classmethod
    def register_model(cls, pydantic_model: Type[BaseModel], orm_model: Type[DeclarativeMeta]):
        """Fazendo o registro de um modelo pydantic no formato de um Objeto Relacional Mapeado do Postgres 16"""

        cls._registry[pydantic_model] = orm_model

    @classmethod
    def get_orm(cls, pydantic_model: Type[BaseModel]) -> Type[DeclarativeMeta]:
        """Resgatando o registro de um modelo pydantic convertido para modelo ORM do Postgres 16"""

        orm = cls._registry.get(pydantic_model)

        if not orm:
            raise ValueError(f"No ORM model registered for {pydantic_model.__name__}")

        return orm
    
    @classmethod
    def register_orm_pair(
        cls,
        model: str,
        fields: Dict[str, Any],
        base_orm: Optional[Type[DeclarativeMeta]] = None,
        base_pydantic: Optional[Type[BaseModel]] = None,
        table_name: Optional[str] = None
    ):
        """Criando modelos pares ORM para pydantic com uma tabela schema personalizada com parâmetros no formato JSON"""

        # ORM
        base_orm = base_orm or Base
        base_pydantic = base_pydantic or BaseModel

        orm_attrs = {"__tablename__": table_name or f"{model.lower()}s"}
        orm_attrs["id"] = Column(Integer, primary_key=True, index=True)

        for fname, ftype in fields.items():
            sqlatype = SQLA_TYPE_MAP(ftype if not isinstance(ftype, tuple) else ftype[0], JSONB)
            orm_attrs[fname] = Column(sqlatype)
        
        orm_model = type(f"{model}ORM", (base_orm), orm_attrs)

        # Pydantic
        pydantic_fields = {}
        for fname, ftype in fields.items():
            if isinstance(ftype, tuple):
                pydantic_fields[fname] = ftype
            else:
                pydantic_fields[fname] = (ftype, ...)

        pydantic_model = create_model(model, **pydantic_fields, __base__=base_pydantic)

        # Register
        ModelRegistry.register(pydantic_model, orm_model)

        return pydantic_model, orm_model
    

    # Gerando rotas CRUD automaticamente a partir de schemas pares 
    # CHANGED: Auto-generated CRUD routes are now visualized via base_purple.html (/registry endpoint)
    # The base_purple.html template provides a complete user interface for Create, Read, Update, Delete operations
    # This eliminates the need for manual API testing tools; users can manage all registered models via the web UI
    @classmethod
    def _make_create(cls, orm_model: Type[DeclarativeMeta]):
        async def create_item(data: dict, db: AsyncSession = Depends(get_db)):
            obj = await create_entry(db, orm_model, data)
            return {"id": getattr(obj, "id", None), "message": f"{orm_model.__name__} created"}
        return create_item

    @classmethod
    def _make_read(cls, orm_model: Type[DeclarativeMeta]):
        async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
            obj = await get_entry(db, orm_model, item_id)
            if not obj:
                raise HTTPException(status_code=404, detail=f"{orm_model.__name__.lower()} not found")
            return obj.__dict__
        return read_item

    @classmethod
    def _make_list(cls, orm_model: Type[DeclarativeMeta]):
        async def list_items(db: AsyncSession = Depends(get_db)):
            objs = await get_all_entries(db, orm_model)
            return [o.__dict__ for o in objs]
        return list_items

    @classmethod
    def _make_update(cls, orm_model: Type[DeclarativeMeta]):
        async def update_item(item_id: int, data: dict, db: AsyncSession = Depends(get_db)):
            updated = await update_entry(db, orm_model, item_id, data)
            if not updated:
                raise HTTPException(status_code=404, detail=f"{orm_model.__name__.lower()} not found")
            return getattr(updated, "__dict__", updated)
        return update_item

    @classmethod
    def _make_delete(cls, orm_model: Type[DeclarativeMeta]):
        async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
            ok = await delete_entry(db, orm_model, item_id)
            if ok:
                return {"status": "deleted", "id": item_id}
            raise HTTPException(status_code=404, detail=f"{orm_model.__name__.lower()} not found")
        return delete_item

    @classmethod
    def generate_router(cls, pydantic_model: Type[BaseModel], prefix_: str = None) -> APIRouter:
        """Generate CRUD APIRouter for a registered pydantic/ORM pair."""

        orm_model = cls.get_orm(pydantic_model)
        name = prefix_ or pydantic_model.__name__.lower()
        router = APIRouter(prefix=f"/{name}", tags=[name.capitalize()])

        # Register minimal CRUD endpoints using factory helpers to keep complexity low
        router.post("/create", response_model=dict)(cls._make_create(orm_model))
        router.get("/get/{item_id}")(cls._make_read(orm_model))
        router.get("/list")(cls._make_list(orm_model))
        router.put("/update/{item_id}", response_model=dict)(cls._make_update(orm_model))
        router.delete("/delete/{item_id}", response_model=dict)(cls._make_delete(orm_model))

        return router
    
    

    @classmethod
    def generate_all_routers(cls) -> List[APIRouter]:
        """Gerando roteadores para todos os modelos registrados."""
        routers = []

        for model in cls._registry:
            router = cls.generate_router(model)
            routers.append(router)
        
        return routers
    
    # Gerando modelos pares novos automaticamente 

