import os
import asyncpg
from typing import Optional

conn_pool: Optional[asyncpg.Pool] = None


import asyncpg

# SQL for creating sequence and tables
CREATE_DB_SCHEMA_SQL = """
DROP TABLE IF EXISTS public.monotomicro CASCADE;
DROP TABLE IF EXISTS public.monotomicro_summary CASCADE;
DROP SEQUENCE IF EXISTS monotomicro_id_seq CASCADE;

CREATE SEQUENCE monotomicro_id_seq;

CREATE TABLE IF NOT EXISTS public.monotomicro (
    id INTEGER NOT NULL DEFAULT nextval('monotomicro_id_seq'::regclass),
    monolith_name VARCHAR NOT NULL,
    monolith_url VARCHAR NOT NULL,
    monolith_analysis VARCHAR,
    microservice_suggestion VARCHAR,
    date_time TIMESTAMP,
    status VARCHAR NOT NULL,
    language VARCHAR(10),
    action VARCHAR(100),
    zip_file_name VARCHAR,
    zip_file_path VARCHAR,

    CONSTRAINT monotomicro_pkey PRIMARY KEY (id)
);


CREATE TABLE IF NOT EXISTS public.monotomicro_summary (
    id SERIAL PRIMARY KEY,
    monolith_name VARCHAR NOT NULL,
    monolith_analysis VARCHAR
);
"""

# Global connection pool
conn_pool = None

async def init_postgres() -> None:
    """
    Initialize PostgreSQL connection pool and set up database schema.
    """
    global conn_pool
    try:
        print("Initializing PostgreSQL connection pool...")

        conn_pool = await asyncpg.create_pool(
            dsn="postgres://user:usertcs@localhost:5432/mydb",
            min_size=1,
            max_size=10
        )

        print("PostgreSQL connection pool created successfully.")

        # Create the database schema
        async with conn_pool.acquire() as connection:
            #await connection.execute(CREATE_DB_SCHEMA_SQL)
            print("Database schema initialized successfully.")

    except Exception as e:
        print(f"Error initializing PostgreSQL connection pool or schema: {e}")
        raise



async def get_postgres() -> asyncpg.Pool:
    """
    Return the PostgreSQL connection pool.

    This function returns the connection pool object, from which individual
    connections can be acquired as needed for database operations. The caller
    is responsible for acquiring and releasing connections from the pool.

    Returns
    -------
    asyncpg.Pool
        The connection pool object to the PostgreSQL database.

    Raises
    ------
    ConnectionError
        Raised if the connection pool is not initialized.
    """
    global conn_pool
    if conn_pool is None:
        print("Connection pool is not initialized.")
        raise ConnectionError("PostgreSQL connection pool is not initialized.")
    try:
        return conn_pool
    except Exception as e:
        print(f"Failed to return PostgreSQL connection pool: {e}")
        raise


async def close_postgres() -> None:
    """
    Close the PostgreSQL connection pool.

    This function should be called during the shutdown of the FastAPI app
    to properly close all connections in the pool and release resources.
    """
    global conn_pool
    if conn_pool is not None:
        try:
            print("Closing PostgreSQL connection pool...")
            await conn_pool.close()
            print("PostgreSQL connection pool closed successfully.")
        except Exception as e:
            print(f"Error closing PostgreSQL connection pool: {e}")
            raise
    else:
        print("PostgreSQL connection pool was not initialized.")