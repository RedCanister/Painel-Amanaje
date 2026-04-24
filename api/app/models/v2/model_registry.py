from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeMeta

from app.database.db_session import get_db
from app.database.db_utils import create_entry, delete_entry, get_all_entries, get_entry, update_entry

from .model_orm import CodeORM, DatasetORM, LearningORM, ObjectORM, StudyORM
from .model_schemas import (
    CodeCreate,
    CodeRead,
    CodeUpdate,
    DatasetCreate,
    DatasetRead,
    DatasetUpdate,
    LearningCreate,
    LearningRead,
    LearningUpdate,
    ObjectCreate,
    ObjectRead,
    ObjectUpdate,
    StudyCreate,
    StudyRead,
    StudyUpdate,
)

LOGGER = logging.getLogger(__name__)


async def _get_db_dependency():
    async for db in get_db():
        yield db


@dataclass(frozen=True)
class ModelRegistration:
    create_schema: Type[BaseModel]
    read_schema: Type[BaseModel]
    update_schema: Type[BaseModel]
    orm_model: Type[DeclarativeMeta]
    prefix: str
    tag: str


class ModelRegistry:
    _registry: Dict[Type[BaseModel], ModelRegistration] = {}

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def register(
        cls,
        *,
        create_schema: Type[BaseModel],
        read_schema: Type[BaseModel],
        update_schema: Type[BaseModel],
        orm_model: Type[DeclarativeMeta],
        prefix: str,
        tag: Optional[str] = None,
    ) -> None:
        registration = ModelRegistration(
            create_schema=create_schema,
            read_schema=read_schema,
            update_schema=update_schema,
            orm_model=orm_model,
            prefix=prefix,
            tag=tag or prefix.capitalize(),
        )
        cls._registry[read_schema] = registration

    @classmethod
    def get_registration(cls, read_schema: Type[BaseModel]) -> ModelRegistration:
        registration = cls._registry.get(read_schema)
        if registration is None:
            raise KeyError(f"{read_schema.__name__} is not registered")
        return registration

    @staticmethod
    def _orm_to_dict(obj: Any) -> Dict[str, Any]:
        table = getattr(obj, "__table__", None)
        if table is None:
            raise TypeError("Expected a SQLAlchemy ORM object with a __table__ attribute")
        return {column.name: getattr(obj, column.name) for column in table.columns}

    @staticmethod
    async def _read_json_payload(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be valid JSON",
            ) from exc

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be a JSON object",
            )

        return payload

    @classmethod
    def _validate_payload(cls, schema: Type[BaseModel], payload: Dict[str, Any], *, partial: bool) -> Dict[str, Any]:
        try:
            validated = schema.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.errors(),
            ) from exc

        return validated.model_dump(exclude_unset=partial)

    @classmethod
    def generate_router(cls, read_schema: Type[BaseModel]) -> APIRouter:
        registration = cls.get_registration(read_schema)
        router = APIRouter(prefix=f"/{registration.prefix}", tags=[registration.tag])

        @router.post("/create", response_model=registration.read_schema, status_code=status.HTTP_201_CREATED)
        async def create_item(request: Request, db: AsyncSession = Depends(_get_db_dependency)):
            payload = await cls._read_json_payload(request)
            validated_payload = cls._validate_payload(registration.create_schema, payload, partial=False)

            try:
                created = await create_entry(db, registration.orm_model, validated_payload)
            except TypeError as exc:
                LOGGER.exception("Failed to create %s with payload %s", registration.prefix, validated_payload)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except Exception as exc:
                LOGGER.exception("Unexpected create failure for %s", registration.prefix)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Create failed") from exc

            return registration.read_schema.model_validate(created, from_attributes=True)

        @router.get("/get/{item_id}", response_model=registration.read_schema)
        async def read_item(item_id: int | str, db: AsyncSession = Depends(_get_db_dependency)):
            try:
                obj = await get_entry(db, registration.orm_model, item_id)
            except Exception as exc:
                LOGGER.exception("Unexpected read failure for %s with id %s", registration.prefix, item_id)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Read failed") from exc

            if obj is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{registration.tag} not found")

            return registration.read_schema.model_validate(obj, from_attributes=True)

        @router.get("/list", response_model=List[registration.read_schema])
        async def list_items(db: AsyncSession = Depends(_get_db_dependency)):
            try:
                objects = await get_all_entries(db, registration.orm_model)
            except Exception as exc:
                LOGGER.exception("Unexpected list failure for %s", registration.prefix)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="List failed") from exc

            return [
                registration.read_schema.model_validate(obj, from_attributes=True)
                for obj in objects
            ]

        @router.put("/update/{item_id}", response_model=registration.read_schema)
        async def update_item(item_id: int | str, request: Request, db: AsyncSession = Depends(_get_db_dependency)):
            payload = await cls._read_json_payload(request)
            validated_payload = cls._validate_payload(registration.update_schema, payload, partial=True)

            try:
                updated = await update_entry(db, registration.orm_model, item_id, validated_payload)
            except TypeError as exc:
                LOGGER.exception("Failed to update %s with id %s", registration.prefix, item_id)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except Exception as exc:
                LOGGER.exception("Unexpected update failure for %s with id %s", registration.prefix, item_id)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Update failed") from exc

            if updated is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{registration.tag} not found")

            return registration.read_schema.model_validate(updated, from_attributes=True)

        @router.delete("/delete/{item_id}", response_model=Dict[str, Any])
        async def delete_item(item_id: int | str, db: AsyncSession = Depends(_get_db_dependency)):
            try:
                deleted = await delete_entry(db, registration.orm_model, item_id)
            except Exception as exc:
                LOGGER.exception("Unexpected delete failure for %s with id %s", registration.prefix, item_id)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Delete failed") from exc

            if not deleted:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{registration.tag} not found")

            return {"status": "deleted", "id": item_id, "resource": registration.prefix}

        return router

    @classmethod
    def generate_all_routers(cls) -> List[APIRouter]:
        return [cls.generate_router(read_schema) for read_schema in cls._registry]


def register_default_models() -> None:
    ModelRegistry.clear()
    ModelRegistry.register(
        create_schema=ObjectCreate,
        read_schema=ObjectRead,
        update_schema=ObjectUpdate,
        orm_model=ObjectORM,
        prefix="objects-v2",
        tag="ObjectsV2",
    )
    ModelRegistry.register(
        create_schema=DatasetCreate,
        read_schema=DatasetRead,
        update_schema=DatasetUpdate,
        orm_model=DatasetORM,
        prefix="datasets-v2",
        tag="DatasetsV2",
    )
    ModelRegistry.register(
        create_schema=LearningCreate,
        read_schema=LearningRead,
        update_schema=LearningUpdate,
        orm_model=LearningORM,
        prefix="learning-models-v2",
        tag="LearningModelsV2",
    )
    ModelRegistry.register(
        create_schema=CodeCreate,
        read_schema=CodeRead,
        update_schema=CodeUpdate,
        orm_model=CodeORM,
        prefix="code-models-v2",
        tag="CodeModelsV2",
    )
    ModelRegistry.register(
        create_schema=StudyCreate,
        read_schema=StudyRead,
        update_schema=StudyUpdate,
        orm_model=StudyORM,
        prefix="study-models-v2",
        tag="StudyModelsV2",
    )


def include_registered_routers(app: FastAPI) -> None:
    for router in ModelRegistry.generate_all_routers():
        app.include_router(router)
