from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.inspection import inspect as sqlalchemy_inspect

from app.database.db_session import Base, get_db
from app.database.db_utils import create_entry, delete_entry, get_all_entries, get_entry, update_entry


SQLA_TYPE_MAP = {
    str: String,
    int: Integer,
    float: Float,
    bool: Boolean,
    dict: JSONB,
    list: JSONB,
    datetime: DateTime,
}


class ModelRegistry:
    _registry: Dict[Type[BaseModel], Type[DeclarativeMeta]] = {}

    @classmethod
    def register_model(cls, pydantic_model: Type[BaseModel], orm_model: Type[DeclarativeMeta]) -> None:
        """Register a Pydantic model to its SQLAlchemy ORM counterpart."""

        cls._registry[pydantic_model] = orm_model

    @classmethod
    def _ensure_defaults_registered(cls) -> None:
        """Populate the registry with the canonical model pairs defined in the app."""

        from .model_objects import CodeModel, DatasetModel, InferenceModel, LearningModel, ObjectModel, StudyModel
        from .model_orm import CodeORM, DatasetORM, InferenceORM, LearningORM, ObjectORM, StudyORM

        default_pairs = (
            (ObjectModel, ObjectORM),
            (DatasetModel, DatasetORM),
            (LearningModel, LearningORM),
            (InferenceModel, InferenceORM),
            (CodeModel, CodeORM),
            (StudyModel, StudyORM),
        )

        for pydantic_model, orm_model in default_pairs:
            if cls._registry.get(pydantic_model) is not orm_model:
                cls._registry[pydantic_model] = orm_model

    @classmethod
    def get_orm(cls, pydantic_model: Type[BaseModel]) -> Type[DeclarativeMeta]:
        """Return the ORM model registered for the provided Pydantic model."""

        cls._ensure_defaults_registered()
        orm = cls._registry.get(pydantic_model)

        if orm is None:
            raise ValueError(f"No ORM model registered for {pydantic_model.__name__}")

        return orm

    @staticmethod
    def _orm_to_dict(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}

        mapper = getattr(obj, "__mapper__", None)
        if mapper is not None:
            return {
                attr.key: getattr(obj, attr.key)
                for attr in sqlalchemy_inspect(obj).mapper.column_attrs
            }

        table = getattr(obj, "__table__", None)
        if table is None:
            return dict(getattr(obj, "__dict__", {}))

        return {column.name: getattr(obj, column.name) for column in table.columns}

    @staticmethod
    async def _parse_body(request: Request, model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Parse a request body into a plain dict.

        Validation is best-effort: if model validation fails, the raw JSON payload
        is returned unchanged so current API behavior is preserved.
        """

        try:
            payload = await request.json()
        except Exception:
            return {}

        if not isinstance(payload, dict):
            return {}

        try:
            if hasattr(model, "model_validate"):
                validated = model.model_validate(payload)
                return validated.model_dump()
            if hasattr(model, "parse_obj"):
                validated = model.parse_obj(payload)
                return validated.dict()
        except Exception:
            return payload

        return payload

    @classmethod
    def register_orm_pair(
        cls,
        name: str,
        fields: Dict[str, Any],
        base_orm: Optional[Type[DeclarativeMeta]] = None,
        base_pydantic: Optional[Type[BaseModel]] = None,
        table_name: Optional[str] = None,
    ):
        """Create and register a dynamic Pydantic/ORM pair."""

        from .model_objects import ObjectModel
        from .model_orm import ObjectORM

        base_orm = base_orm or ObjectORM
        base_pydantic = base_pydantic or ObjectModel

        orm_attrs = {"__tablename__": (table_name or name.lower()) + "s"}

        if issubclass(base_orm, ObjectORM):
            orm_attrs["id"] = Column(Integer, ForeignKey("objects.id"), primary_key=True)

        for field_name, field_type in fields.items():
            normalized_type = field_type[0] if isinstance(field_type, tuple) else field_type

            if isinstance(normalized_type, str):
                normalized_type = str

            sqlatype = SQLA_TYPE_MAP.get(normalized_type, JSONB)
            orm_attrs[field_name] = Column(JSONB if sqlatype == JSONB else sqlatype)

        orm_attrs["__mapper_args__"] = {"polymorphic_identity": name.lower()}
        orm_model = type(f"{name.capitalize()}ORM", (base_orm,), orm_attrs)

        pydantic_fields: Dict[str, Any] = {}
        for field_name, field_type in fields.items():
            if isinstance(field_type, tuple):
                pydantic_fields[field_name] = field_type
            elif isinstance(field_type, type):
                pydantic_fields[field_name] = (field_type, None)
            else:
                pydantic_fields[field_name] = (dict, None)

        pydantic_model = create_model(
            f"{name}Model",
            **pydantic_fields,
            __base__=base_pydantic,
            __config__={"from_attributes": True},
        )

        cls.register_model(pydantic_model, orm_model)
        return pydantic_model, orm_model

    @classmethod
    def generate_router(cls, pydantic_model: Type[BaseModel], prefix_: str = None) -> APIRouter:
        """Generate CRUD routes for a registered Pydantic/ORM pair."""

        orm_model = cls.get_orm(pydantic_model)
        name = prefix_ or pydantic_model.__name__.lower()
        router = APIRouter(prefix=f"/{name}", tags=[name.capitalize()])

        @router.post("/create", response_model=dict)
        async def create_item(request: Request, db: AsyncSession = Depends(get_db)):
            try:
                payload = await cls._parse_body(request, pydantic_model)
                obj = await create_entry(db, orm_model, payload)
                return {"id": getattr(obj, "id", None), "message": f"{name} created"}
            except Exception as exc:
                return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

        @router.get("/get/{item_id}", response_model=pydantic_model)
        async def read_item(item_id: int | str, db: AsyncSession = Depends(get_db)):
            obj = await get_entry(db, orm_model, item_id)

            if not obj:
                raise HTTPException(status_code=404, detail=f"{name} not found")

            try:
                if hasattr(pydantic_model, "model_validate"):
                    return pydantic_model.model_validate(obj, from_attributes=True)
                if hasattr(pydantic_model, "parse_obj"):
                    return pydantic_model.parse_obj(obj)
            except Exception:
                return cls._orm_to_dict(obj)

            return cls._orm_to_dict(obj)

        @router.get("/list", response_model=List[pydantic_model])
        async def read_all_items(db: AsyncSession = Depends(get_db)):
            try:
                objs = await get_all_entries(db, orm_model)
                items: List[Any] = []

                for obj in objs:
                    try:
                        if hasattr(pydantic_model, "model_validate"):
                            items.append(pydantic_model.model_validate(obj, from_attributes=True))
                        elif hasattr(pydantic_model, "parse_obj"):
                            items.append(pydantic_model.parse_obj(obj))
                        else:
                            items.append(cls._orm_to_dict(obj))
                    except Exception:
                        items.append(cls._orm_to_dict(obj))
            except Exception as exc:
                return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

            return items

        @router.put("/update/{item_id}", response_model=dict)
        async def update_item(item_id: int | str, request: Request, db: AsyncSession = Depends(get_db)):
            try:
                payload = await cls._parse_body(request, pydantic_model)
                updated = await update_entry(db, orm_model, item_id, payload)

                if not updated:
                    raise HTTPException(status_code=404, detail=f"{name} not found")
            except HTTPException:
                raise
            except Exception as exc:
                return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

            return cls._orm_to_dict(updated)

        @router.delete("/delete/{item_id}", response_model=dict)
        async def delete_item(item_id: int | str, db: AsyncSession = Depends(get_db)):
            ok = await delete_entry(db, orm_model, item_id)

            if ok:
                return {"status": "deleted", "orm_model": str(orm_model), "id": item_id}

            raise HTTPException(status_code=404, detail=f"{name} not found")

        return router

    @classmethod
    def generate_all_routers(cls) -> List[APIRouter]:
        """Generate routers for all registered models."""

        cls._ensure_defaults_registered()
        return [cls.generate_router(model) for model in cls._registry]
