from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB

from typing import Dict, Any, Optional, Type, List
from pydantic import create_model, BaseModel
from datetime import datetime

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
        name: str,
        fields: Dict[str, Any],
        base_orm: Optional[Type[DeclarativeMeta]] = None,
        base_pydantic: Optional[Type[BaseModel]] = None,
        table_name: Optional[str] = None
    ):
        from .model_objects import ObjectModel
        from .model_orm import ObjectORM

        """Criando modelos pares ORM para pydantic com uma tabela schema personalizada com parâmetros no formato JSON"""

        # ORM
        base_orm = base_orm or Base
        base_pydantic = base_pydantic or BaseModel

        orm_attrs = {"__tablename__": (table_name or name.lower()) + "s"}
        orm_attrs["id"] = Column(Integer, ForeignKey("objects.id"), primary_key=True)

        for fname, ftype in fields.items():
            if isinstance(ftype, str):
                ftype = str

            sqlatype = SQLA_TYPE_MAP.get(ftype if not isinstance(ftype, tuple) else ftype[0], JSONB)
            
            if sqlatype == JSONB:
                orm_attrs[fname] = Column(JSONB)
            else:
                orm_attrs[fname] = Column(sqlatype)
        
        orm_attrs["__mapper_args__"] = {"polymorphic_identity": name.lower()}

        orm_model = type(f"{name.capitalize()}ORM", (ObjectORM,), orm_attrs)

        # Pydantic
        pydantic_fields = {}
        for fname, ftype in fields.items():
            if isinstance(ftype, tuple):
                pydantic_fields[fname] = ftype
            elif isinstance(ftype, type):
                pydantic_fields[fname] = (ftype, None)
            else:
                pydantic_fields[fname] = (dict, None)
        
        pydantic_model = create_model(name + "Model",
                                      **pydantic_fields, 
                                      __base__=ObjectModel,
                                      __config__={"from_attributes": True}
                                    )

        # Register
        ModelRegistry.register_model(pydantic_model, orm_model)

        return pydantic_model, orm_model
    

    # Gerando rotas CRUD automaticamente a partir de schemas pares
    # CHANGED: `generate_router()` provides a standard set of CRUD endpoints for the
    # registered Pydantic/ORM pair. A lightweight management UI is available in
    # `api/templates/base_purple.html` and can be mounted by the application
    # (templates are served elsewhere in the app). This function intentionally
    # focuses on producing the API surface; the UI uses these routes to visualize
    # and manage objects.
    @classmethod
    def generate_router(cls, pydantic_model: Type[BaseModel], prefix_: str = None) -> APIRouter:
        """Traduzindo funções de operação no banco de dados para rotas do FastAPI"""

        orm_model = cls.get_orm(pydantic_model)
        name = prefix_ or pydantic_model.__name__.lower()
        router = APIRouter(prefix=f"/{name}", tags=[name.capitalize()])

        # TODO - This function is causing trouble within the app and needs to be revised
        async def _parse_body(request, model):
            """
            Parse JSON body and attempt to validate using the provided pydantic
            model when available. Always return a plain dict suitable for DB
            helpers; be tolerant of validation failures and return the raw
            payload in that case.
            """

            payload = {}

            try:
                payload = await request.json()
            except Exception:
                payload = {}

            try:
                # pydantic v2 API
                if hasattr(model, "model_validate") and isinstance(payload, dict):
                    validated = model.model_validate(payload)
                    try:
                        if hasattr(validated, "model_dump"):
                            payload = validated.model_dump()
                        elif hasattr(validated, "dict"):
                            payload = validated.dict()
                        else:
                            payload = dict(validated)
                    except Exception:
                        pass

                    #debug_type(payload)

                # pydantic v1 fallback
                elif hasattr(model, "parse_obj") and isinstance(payload, dict):
                    try:
                        validated = model.parse_obj(payload)
                        payload = validated.dict()
                    except Exception:
                        pass
            except Exception:
                # Keep payload as-is on any unexpected error
                pass

            return payload
        

        # Criar
        # TODO - This function is causing trouble within the app and needs to be revised
        @router.post("/create", response_model=dict)
        async def create_item(request: Request, 
                              #data: pydantic_model, 
                              db: AsyncSession = Depends(get_db)):
            """Rota para criação e inserção de objeto pydantic ao banco de dados postgres"""

            try:
                payload = await request.json()

                obj = await create_entry(db, orm_model, payload)

                print("Created!")

                return {"id": getattr(obj, "id", None), "message": f"{name} created"}
            except Exception as e:
                return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


        # Ver uma
        @router.get("/get/{item_id}", response_model=pydantic_model)
        async def read_item(item_id: int | str, db: AsyncSession = Depends(get_db)):
            """Rota para Leitura e resgate de objeto pydantic no banco de dados postgres"""

            print("Getting entry")
            debug_type(item_id)

            if isinstance(item_id, str):
                obj = await get_entry(db, orm_model, str(item_id))
            elif isinstance(item_id, int):
                obj = await get_entry(db, orm_model, int(item_id))

            debug_type(orm_model)

            if not obj:
                raise HTTPException(status_code=404, detail=f"{name} not found")

            return obj.__dict__
        
        # Ver todas
        @router.get("/list", response_model=List[pydantic_model])
        async def read_all_items(db: AsyncSession = Depends(get_db)):
            """Rota para leitura e resgate de todos os objetos pydantic de uma classe específica ao banco de dados postgres"""
            print("Objects are requested")
            try:
                
                objs = await get_all_entries(db, orm_model)
                print("Objects are listed", objs)
                obj_list = [o.__dict__ for o in objs]
            except Exception as e:
                return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

            return obj_list

        # Atualizar
        @router.put("/update/{item_id}", response_model=dict)
        async def update_item(item_id: int | str,
                              request: Request, 
                              #data: pydantic_model, 
                              db: AsyncSession = Depends(get_db)):
            """Rota para atualização e inserção de objeto pydantic no banco de dados postgres"""

            try:
                payload = await _parse_body(request, pydantic_model)
                updated = await update_entry(db, orm_model, item_id, payload)

                if not updated:
                    raise HTTPException(status_code=404, detail=f"{name} not found")
            except HTTPException:
                raise
            except Exception as e:
                return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
                
            return updated.__dict__

        # Deletar
        @router.delete("/delete/{item_id}", response_model=dict)
        async def delete_item(item_id: int | str, db: AsyncSession = Depends(get_db)):
            """Rota para deleção e atualização de objeto pydantic do banco de dados postgres"""

            ok = await delete_entry(db, orm_model, item_id)

            if ok:
                return {"status": "deleted", "orm_model": str(orm_model), "id": item_id}

            raise HTTPException(status_code=404, detail=f"{name} not found")
              
        
        # Retorno de generate_router()
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
