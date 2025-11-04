from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Type, List, Optional
from pydantic import BaseModel
from app.database.db_session import Base
from app.utils.utils import debug_type

# Result.scalars - return _asdict(), _fields(), _mapping(), 

# Criando uma entrada de um modelo orm com classe pydantic associada e seus dados correspondentes
async def create_entry(db: AsyncSession, orm_model: Type, data: BaseModel):
    "docstring"
    
    obj = orm_model(**data.dict(exclude_unset=True))
    db.add(obj)

    debug_type(obj)
    
    await db.commit()
    await db.refresh(obj)
    
    return obj

# Resgatando uma entrada de um modelo orm com classe pydantic
async def get_entry(db: AsyncSession, orm_model:Type, entry_id: int):
    "docstring"
    
    result = await db.execute(select(orm_model).where(orm_model.id == entry_id))
    first_entry = result.scalars.first()
    
    debug_type(result)
    debug_type(first_entry)

    return first_entry

# Resgatando todas as entradas de um modelo orm específico
async def get_all_entries(db: AsyncSession, orm_model: Type) -> List:
    "docstring"
    
    result = await db.execute(select(orm_model))
    all_entries = result.scalars().all()

    debug_type(result)
    debug_type(all_entries)

    return all_entries

# Atualizando todas as entradas de um modelo orm específico
async def update_entry(db: AsyncSession, orm_model: Type, entry_id: int, data: BaseModel):
    "docstring"
    
    obj = await get_entry(db, orm_model, entry_id)

    if not obj:
        return None
    
    for field, value in data.dict(exclude_unset=True).items():
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
