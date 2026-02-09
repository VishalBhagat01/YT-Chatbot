import os
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
except Exception:
    st = None

_pool = None


def _get_database_url():

    url = os.getenv("DATABASE_URL")
    if not url and st:
        url = st.secrets.get("DATABASE_URL", None)

    if not url:
        raise RuntimeError("DATABASE_URL not configured")

    if "sslmode=" not in url:
        url = url + "?sslmode=require"

    return url


def get_pool():
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_get_database_url(),
            min_size=2,
            max_size=10,
            timeout=30,
        )
    return _pool


def get_db():
    return get_pool().getconn()


def release_db(conn):
    get_pool().putconn(conn)
