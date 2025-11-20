from fastapi import HTTPException
import asyncpg
from datetime import datetime

async def check_if_name_already_added(name: str, db_pool):
    query = "select monolith_name from monotomicro"
    async with db_pool.acquire() as conn:
        result = await conn.fetch(query)
        names = [record[0] for record in result]
        print (names)
        return name in names

async def get_all(db_pool: asyncpg.Pool):
    query = "select monolith_name, monolith_url, language, date_time, status, action from monotomicro"
    async with db_pool.acquire() as conn:
        result = await conn.fetch(query)
        return result

async def get_data_by_name(name: str, db_pool: asyncpg.Pool):
    name_present = await check_if_name_already_added(name, db_pool)
    if not name_present:
        raise HTTPException(
            status_code=400, detail="Given name not present in the database."
        )
    query = "select monolith_name, monolith_url, language, date_time, status, action from monotomicro where monolith_name = $1"
    async with db_pool.acquire() as conn:
            result = await conn.fetchrow(query, name)
            return result

async def insert_db_record(
    monolith_name : str,
    monolith_url : str,
    language: str,
    db_pool: asyncpg.Pool,
) -> str:
    query = """
        INSERT INTO monotomicro (monolith_name, monolith_url, monolith_analysis, microservice_suggestion, date_time, status, language, action)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute(query, monolith_name, monolith_url, "", "", datetime.now(), "uploaded", language, "embedding")
            return "monolith files added!"
    except Exception as e:
        print(f"Error inserting repo data: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during inserting monolith data"
        )