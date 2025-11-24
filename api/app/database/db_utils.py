from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Type, List, Optional
from pydantic import BaseModel

from app.models.model_orm import ObjectORM

from app.database.db_session import Base
from app.utils.utils import debug_type

# Result.scalars - return _asdict(), _fields(), _mapping(), 

# Criando uma entrada de um modelo orm com classe pydantic associada e seus dados correspondentes
async def create_entry(db: AsyncSession, orm_model: ObjectORM, data: BaseModel | dict):
    """Create an ORM entry from a Pydantic model or plain dict.

    Accept both a BaseModel instance (with .dict()) or a plain dict so caller
    logic can construct payloads that map to ORM column names.
    """

    if hasattr(data, "dict"):
        payload = data.model_dump(exclude_unset=True)
    elif isinstance(data, dict):
        payload = data
    else:
        raise TypeError("data must be a pydantic BaseModel or a dict")

    obj = orm_model(**payload)
    db.add(obj)

    debug_type(obj)

    await db.commit()
    await db.refresh(obj)

    return obj

# Resgatando uma entrada de um modelo orm com classe pydantic
async def get_entry(db: AsyncSession, orm_model:Type, entry_id: int):
    "docstring"
    
    result = await db.execute(select(orm_model).where(orm_model.id == entry_id))
    first_entry = result.scalars().first()
    
    # debug_type(result)
    debug_type(first_entry)

    return first_entry

# Resgatando todas as entradas de um modelo orm específico
async def get_all_entries(db: AsyncSession, orm_model: Type) -> List:
    "docstring"
    
    result = await db.execute(select(orm_model))

    all_entries = result.scalars().all()

    debug_type(all_entries)

    return all_entries

# Atualizando todas as entradas de um modelo orm específico
async def update_entry(db: AsyncSession, orm_model: ObjectORM, entry_id: int, data: BaseModel | dict):
    """Update an ORM entry by id with data from a BaseModel or dict."""

    #debug_type(data)

    obj = await get_entry(db, orm_model, entry_id)
    
    #debug_type(obj)

    if not obj:
        return None
    
    if hasattr(data, "dict"):
        updates = data.model_dump(exclude_unset=True)
    elif isinstance(data, dict):
        updates = data
    else:
        raise TypeError("data must be a pydantic BaseModel or a dict")

    debug_type(updates)

    for field, value in updates.items():
        setattr(obj, field, value)


    await db.commit()
    await db.refresh(obj)

    return obj


async def delete_entry(db: AsyncSession, orm_model:Type, entry_id: int) -> bool:
    "docstring"
    
    obj = await get_entry(db, orm_model, entry_id)

    if not obj:
        return False

    await db.delete(obj)
    await db.commit()

    return True
