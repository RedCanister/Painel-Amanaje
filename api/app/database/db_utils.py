from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Type, List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.models.model_orm import ObjectORM

from app.database.db_session import Base
from app.utils.utils import debug_type

# Result.scalars - return _asdict(), _fields(), _mapping()


def _normalize_datetime_value(value):
    """Convert supported datetime payloads to datetime objects."""
    if not isinstance(value, str):
        return value

    normalized_value = value.strip()
    if normalized_value.endswith("Z"):
        normalized_value = normalized_value[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized_value)
    except (ValueError, TypeError):
        return value

# Database utilities: create/get/update/delete helpers for ORM models
async def create_entry(db: AsyncSession, orm_model: ObjectORM, data: BaseModel | dict):
    """Create an ORM entry from a Pydantic model or plain dict.

    Accept both a BaseModel instance (with .dict()) or a plain dict so caller
    logic can construct payloads that map to ORM column names.
    """

    # Normalize input to a plain dict. Support Pydantic v2 (.model_dump) and v1 (.dict).
    payload = None

    if isinstance(data, BaseModel) or hasattr(data, "model_dump") or hasattr(data, "dict"):
        try:
            if hasattr(data, "model_dump"):
                payload = data.model_dump()
            elif hasattr(data, "dict"):
                # pydantic v1 dict supports exclude_unset
                payload = data.dict(exclude_unset=True)
            else:
                payload = dict(data)
            print("Has dump or dict.")
        except Exception:
            # fallback to a best-effort conversion
            
            payload = dict(data)
            
            print("Not in format.")
        
    elif isinstance(data, dict):
        payload = data
        print("utils.py:create_entry 45:23 Data is a dict")
    else:
        raise TypeError("data must be a pydantic BaseModel or a dict")

    if not isinstance(payload, dict):
        raise TypeError("normalized payload must be a dict")

    # Coerce string ISO datetime values to datetime objects for DateTime columns
    # This handles payloads with ISO-formatted date strings from the frontend
    if 'date' in payload.keys() and isinstance(payload['date'], str):
        print("Date is not datetime.datetime...")
        payload['date'] = _normalize_datetime_value(payload['date'])
        print("Date is normalized")

    try:
        obj = orm_model(**payload)
    except TypeError as e:
        print(f"Failed to instantiate {orm_model} with filtered payload: {e}")
        raise

    print("Obj", obj)

    db.add(obj)

    debug_type(obj)
    try:
        print("Creation successful")
        await db.commit()
        await db.refresh(obj)
    except Exception as e:
        print(f"DB commit/refresh failed for {orm_model} with payload {payload}: {e}")
        await db.rollback()
        raise

    return obj

# Resgatando uma entrada de um modelo orm por id ou nome
async def get_entry(db: AsyncSession, orm_model: Type, entry_id: int | str) -> Optional[object]:
    # Try integer id lookup first
    entry = None

    print(f"Getting entry from {orm_model.__name__}...")
    if isinstance(entry_id, int):
        print("{orm_model.__name__} item_id == (int)")
        result = await db.execute(select(orm_model).where(orm_model.id == entry_id))
        entry = result.scalars().first()
        debug_type(entry)
        return entry

    # If a string was provided, try to look up by a 'name' column if present,
    # otherwise attempt to coerce to int and lookup by id.
    if isinstance(entry_id, str):
        attr = getattr(orm_model, "name", None)
        print(f"{orm_model.__name__} item_id == (str)")
        if attr is not None:
            result = await db.execute(select(orm_model).where(attr == str(entry_id)))
            entry = result.scalars().first()
            debug_type(entry)
            print("Returned entry.")
            return entry

        # Fallback: try numeric string -> id
        try:
            iid = int(entry_id)
            result = await db.execute(select(orm_model).where(orm_model.id == iid))
            entry = result.scalars().first()
            debug_type(entry)
            print("Returned entry.")
            return entry
        
        except Exception:
            return None

    return None


# Resgatando todas as entradas de um modelo orm específico
async def get_all_entries(db: AsyncSession, orm_model: Type) -> List:
    "docstring"

    print("List got called", orm_model)
    result = await db.execute(select(orm_model))
    print("Result is printed,", result)
    all_entries = result.scalars().all()

    debug_type(all_entries)

    return all_entries


# Atualizando todas as entradas de um modelo orm específico
async def update_entry(db: AsyncSession, orm_model: ObjectORM, entry_id: int | str, data: BaseModel | dict):
    """Update an ORM entry by id with data from a BaseModel or dict."""

    #debug_type(data)

    obj = await get_entry(db, orm_model, entry_id)

    if not obj:
        return None
    
    if hasattr(data, "dict"):
        # pydantic v2 preferred, fall back to v1
        try:
            updates = data.model_dump(exclude_unset=True)
        except Exception:
            updates = data.dict(exclude_unset=True)
    elif isinstance(data, dict):
        updates = data
    else:
        raise TypeError("data must be a pydantic BaseModel or a dict")

    debug_type(updates)

    # Coerce string ISO datetime values to datetime objects for DateTime columns
    if 'date' in updates and isinstance(updates['date'], str):
        updates['date'] = _normalize_datetime_value(updates['date'])

    for field, value in updates.items():
        setattr(obj, field, value)

    await db.commit()
    await db.refresh(obj)

    return obj


async def delete_entry(db: AsyncSession, orm_model:Type, entry_id: int | str) -> bool:
    "docstring"
    
    obj = await get_entry(db, orm_model, entry_id)

    if not obj:
        return False

    await db.delete(obj)
    await db.commit()

    return True
