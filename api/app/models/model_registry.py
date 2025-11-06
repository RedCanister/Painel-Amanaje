from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB

from typing import Dict, Any, Optional, Type, List
from pydantic import create_model, BaseModel
from datetime import datetime

from app.database.db_session import get_db, Base
from app.database.db_utils import create_entry, get_entry, get_all_entries, update_entry, delete_entry


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
    # TODO - generate_router() must have a base_purple.html simply for visualizing the objects in the 
    # database in a simple interface.
    @classmethod
    def generate_router(cls, pydantic_model: Type[BaseModel], prefix: str = None) -> APIRouter:
        """Traduzindo funções de operação no banco de dados para rotas do FastAPI"""

        orm_model = cls.get_orm(pydantic_model)
        name = prefix or pydantic_model.__name__.lower()
        router = APIRouter(prefix=f"{name}", tags=name.capitalize())

        # Criar
        @router.post("/", response_model=dict)
        async def create_item(data: BaseModel, db: AsyncSession = Depends(get_db)):
            """Rota para criação e inserção de objeto pydantic ao banco de dados postgres"""

            obj = await create_entry(db, orm_model, data)
            return {"id": obj.id, "message": "f{name} created"}
        
        # Ver uma
        @router.get("/{item_id}", response_model=dict)
        async def read_item(item_id: int, db: AsyncSession = Depends(get_db)):
            """Rota para Leitura e resgate de objeto pydantic no banco de dados postgres"""

            obj = await get_entry(db, orm_model, item_id)

            if not obj:
                raise HTTPException(status_code=404, detail=f"{name} not found")
            
            return obj.__dict__
        
        # Ver todas
        @router.get("/", response_model=dict)
        async def read_all_items(db: AsyncSession = Depends(get_db)):
            """Rota para leitura e resgate de todos os objetos pydantic de uma classe específicaa ao banco de dados postgres"""

            objs = await get_all_entries(db, orm_model)

            return [o.__dict__ for o in objs]

        # Atualizar
        @router.put("/{item_id}", response_model=dict)
        async def update_item(item_id: int, data: BaseModel, db: AsyncSession = Depends(get_db)):
            """Rota para atualização e inserção de objeto pydantic no banco de dados postgres"""

            updated = await update_entry(db, orm_model, item_id, data)

            if not updated:
                raise HTTPException(status_code=404, detail=f"{name} not found")
            
            return updated.__dict__

        # Deletar
        @router.delete("/{item_id}", response_model=dict)
        async def delete_item(item_id: int, db: AsyncSession = Depends(get_db)):
            """Rota para deleção e atualização de objeto pydantic do banco de dados postgres"""

            ok = await delete_entry(db, orm_model, item_id)
            if not ok:
                raise HTTPException(status_code=404, detail=f"{name} not found")
            return {"status": "orm_model:{orm_model}, id:{item_id} deleted"}
        
        # Retorno de generate_router()
        return router
    
    

    @classmethod
    def generate_all_routers(cls) -> List[APIRouter]:
        """Gerando roteadores para todos os modelos registrados."""
        routers = []

        for model in cls._registry_keys():
            router = cls.generate_router(model)
            routers.append(router)
        
        return routers
    
    # Gerando modelos pares novos automaticamente 
