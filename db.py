import os
import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=os.getenv("DATABASE_URL") + "?sslmode=require",
            min_size=2,
            max_size=10,
            timeout=30,
        )
    return _pool

def get_db():
    return get_pool().getconn()

def release_db(conn):
    get_pool().putconn(conn)
