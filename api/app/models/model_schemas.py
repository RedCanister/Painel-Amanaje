from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from .model_registry import ModelRegistry
from app.database.db_utils import create_entry, get_entry, get_all_entries, update_entry, delete_entry

class AsyncCRUDMixin(BaseModel):
    """CRUD assíncrono universal com resolução de ORM automática"""

    @classmethod
    async def create(cls, db:AsyncSession, data: dict):
        orm_model = ModelRegistry.get_orm(cls)
        return await create_entry(db, orm_model, cls(**data))
    
    @classmethod
    async def read(cls, db:AsyncSession, entry_id: int | str):
        orm_model = ModelRegistry.get_orm(cls)
        return await get_entry(db, orm_model, entry_id)
    
    @classmethod
    async def read_all(cls, db: AsyncSession, ):
        orm_model = ModelRegistry.get_orm(cls)
        return await get_all_entries(db, orm_model,)
    
    @classmethod
    async def update(cls, db:AsyncSession, entry_id: int | str, data: dict):
        orm_model = ModelRegistry.get_orm(cls)
        return await update_entry(db, orm_model, entry_id, cls(**data))
    
    @classmethod
    async def delete(cls, db:AsyncSession, entry_id: int | str):
        orm_model = ModelRegistry.get_orm(cls)
        return await delete_entry(db, orm_model, entry_id)
    
    model_config = ConfigDict(from_attributes=True)
