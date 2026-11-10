import logging
import mysql.connector
from mysql.connector import pooling
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

logger = logging.getLogger(__name__)

_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        try:
            _pool = pooling.MySQLConnectionPool(
                pool_name="deepfake_db_pool",
                pool_size=8,
                pool_reset_session=True,
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME
            )
        except Exception as exc:
            logger.warning("Could not initialize MySQL connection pool: %s. Using direct connections.", exc)
            _pool = None
    return _pool


def get_db_connection():
    pool = _get_pool()
    if pool is not None:
        try:
            return pool.get_connection()
        except Exception as exc:
            logger.warning("Pool exhausted or connection error: %s. Falling back to direct connection.", exc)
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
