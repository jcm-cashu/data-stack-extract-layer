import os
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import duckdb
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
try:  # SQLAlchemy >=2.0
    from sqlalchemy.engine import Engine
except ImportError:  # pragma: no cover - fallback for older versions
    from sqlalchemy.engine.base import Engine
from sqlalchemy.sql.elements import TextClause


# ---------------------------------------------------------------------------
# DuckDB Configuration and Helpers
# ---------------------------------------------------------------------------

load_dotenv("/home/joao/cashu/.env")
load_dotenv()

DEFAULT_DUCKDB_PATH = os.getenv("DUCKDB_PATH", "coletas.duckdb")

# Type mapping from pandas dtypes to DuckDB SQL types
DUCKDB_TYPE_MAP = {
    "object": "VARCHAR",
    "string": "VARCHAR",
    "float64": "DOUBLE",
    "float32": "FLOAT",
    "int64": "BIGINT",
    "Int64": "BIGINT",
    "int32": "INTEGER",
    "Int32": "INTEGER",
    "int16": "SMALLINT",
    "Int16": "SMALLINT",
    "int8": "TINYINT",
    "Int8": "TINYINT",
    "uint64": "UBIGINT",
    "uint32": "UINTEGER",
    "uint16": "USMALLINT",
    "uint8": "UTINYINT",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
    "datetime64[ns, UTC]": "TIMESTAMPTZ",
    "timedelta64[ns]": "INTERVAL",
    "category": "VARCHAR",
}


def generate_duckdb_ddl(
    df,
    table_name: str,
    *,
    type_overrides: Optional[Dict[str, str]] = None,
    include_drop: bool = False,
) -> str:
    """Generate CREATE TABLE DDL statement from a pandas DataFrame.

    Useful for creating bronze/raw layer tables when API documentation is stale.

    Args:
        df: pandas DataFrame to infer schema from.
        table_name: Name of the table to create.
        type_overrides: Optional dict mapping column names to specific SQL types.
        include_drop: If True, prepend DROP TABLE IF EXISTS statement.

    Returns:
        DDL string for creating the table.

    Example:
        >>> ddl = generate_duckdb_ddl(df, "raw_holdings", type_overrides={"amount": "DECIMAL(18,2)"})
        >>> print(ddl)
    """
    overrides = type_overrides or {}

    columns = []
    for column, dtype in df.dtypes.items():
        column_key = str(column)
        sql_type = overrides.get(column_key)
        if not sql_type:
            sql_type = DUCKDB_TYPE_MAP.get(str(dtype), "VARCHAR")
        columns.append(f'    {column_key} {sql_type}')

    cols_sql = ",\n".join(columns)

    ddl_parts = []
    if include_drop:
        ddl_parts.append(f"DROP TABLE IF EXISTS {table_name};")

    ddl_parts.append(f"CREATE TABLE {table_name} (\n{cols_sql}\n);")

    return "\n\n".join(ddl_parts)


def print_schema_from_df(df, table_name: str = "table_name") -> None:
    """Print DDL and sample data info for a DataFrame.

    Useful for debugging and understanding raw API responses.

    Args:
        df: pandas DataFrame to analyze.
        table_name: Name to use in the DDL output.
    """
    print(f"\n{'='*60}")
    print(f"Schema for: {table_name}")
    print(f"{'='*60}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"\n-- Column Types --")
    for col, dtype in df.dtypes.items():
        null_count = df[col].isna().sum()
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
        sample_str = str(sample)[:50] + "..." if len(str(sample)) > 50 else str(sample)
        print(f"  {col}: {dtype} (nulls: {null_count}, sample: {sample_str})")
    print(f"\n-- DDL --")
    print(generate_duckdb_ddl(df, table_name))
    print(f"{'='*60}\n")


def get_duckdb_connection(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection.

    Args:
        db_path: Path to the DuckDB database file. If None, uses DEFAULT_DUCKDB_PATH.
                 Use `:memory:` for an in-memory database.

    Returns:
        DuckDB connection object.
    """
    path = db_path or DEFAULT_DUCKDB_PATH
    return duckdb.connect(path)


def ensure_duckdb_schema(conn: duckdb.DuckDBPyConnection, schema: str) -> None:
    """Ensure a schema exists in DuckDB."""
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")


def load_to_duckdb(
    df,
    table_name: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[str] = None,
    if_exists: str = "append",
) -> None:
    """Load a DataFrame into a DuckDB table.

    Args:
        df: pandas DataFrame to load.
        table_name: Target table name (can include schema as 'schema.table').
        conn: Optional existing DuckDB connection. If None, creates a new one.
        db_path: Path to DuckDB file (used only if conn is None).
        if_exists: How to handle existing data - 'append', 'replace', or 'fail'.
    """
    should_close = conn is None
    conn = conn or get_duckdb_connection(db_path)

    try:
        # Handle schema.table format
        if "." in table_name:
            schema, tbl = table_name.split(".", 1)
            ensure_duckdb_schema(conn, schema)
        else:
            schema = None
            tbl = table_name

        # Use unquoted names for DuckDB (simpler identifiers work without quotes)
        full_table_name = table_name

        # Check if table exists (include schema in check)
        if schema:
            table_exists = conn.execute(
                """SELECT COUNT(*) FROM information_schema.tables 
                   WHERE table_schema = ? AND table_name = ?""",
                [schema, tbl],
            ).fetchone()[0] > 0
        else:
            table_exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [tbl],
            ).fetchone()[0] > 0

        if if_exists == "replace" or not table_exists:
            # Create or replace table from DataFrame
            conn.execute(f"CREATE OR REPLACE TABLE {full_table_name} AS SELECT * FROM df")
        elif if_exists == "append":
            # Insert into existing table
            conn.execute(f"INSERT INTO {full_table_name} SELECT * FROM df")
        elif if_exists == "fail" and table_exists:
            raise ValueError(f"Table {full_table_name} already exists")

        row_count = len(df)
        print(f"Successfully loaded {row_count} rows to {full_table_name} (DuckDB)")

    finally:
        if should_close:
            conn.close()


def execute_duckdb_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[str] = None,
    as_dataframe: bool = False,
):
    """Execute a query on DuckDB and return results.

    Args:
        query: SQL query string.
        params: Optional dictionary of parameters for the query.
        conn: Optional existing DuckDB connection.
        db_path: Path to DuckDB file (used only if conn is None).
        as_dataframe: If True, return results as a pandas DataFrame.

    Returns:
        Query results as DataFrame, list of tuples, or row count for non-SELECT queries.
    """
    should_close = conn is None
    conn = conn or get_duckdb_connection(db_path)

    try:
        if params:
            # DuckDB uses $name for named parameters
            for key, value in params.items():
                query = query.replace(f":{key}", f"${key}")
            result = conn.execute(query, params)
        else:
            result = conn.execute(query)

        # Check if query returns rows
        try:
            if as_dataframe:
                return result.df()
            return result.fetchall()
        except RuntimeError:
            # Query doesn't return rows (INSERT, UPDATE, DELETE, etc.)
            return None

    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# PostgreSQL Configuration and Helpers (Legacy)
# ---------------------------------------------------------------------------

def create_engine_from_env(env_var: str = "cashu_production") -> Engine:
    load_dotenv()
    dsn = os.getenv(env_var)
    if not dsn:
        raise ValueError(f"Environment variable '{env_var}' is not set or empty")
    return create_engine(dsn)


@lru_cache(maxsize=1)
def get_engine(env_var: str = "cashu_production") -> Engine:
    """Return a cached SQLAlchemy engine built from the environment."""

    return create_engine_from_env(env_var)

def fast_copy_to_postgres(df, table_name, engine, schema=None):
    """Fast bulk insert using COPY command with schema support"""
    raw_conn = engine.raw_connection()
    cursor = raw_conn.cursor()

    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, sep='\t', na_rep='\\N')
    buffer.seek(0)

    if schema:
        full_table_name = f'"{schema}"."{table_name}"'
    elif '.' in table_name:
        parts = table_name.split('.')
        full_table_name = f'"{parts[0]}"."{parts[1]}"'
    else:
        full_table_name = f'"{table_name}"'

    columns = ', '.join([f'"{col}"' for col in df.columns])

    try:
        copy_sql = (
            f"COPY {full_table_name} ({columns}) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', NULL '\\N')"
        )
        cursor.copy_expert(sql=copy_sql, file=buffer)
        raw_conn.commit()
        print(f"Successfully copied {len(df)} rows to {full_table_name}")
    except Exception as e:
        raw_conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        cursor.close()
        raw_conn.close()



def execute_query(
    query: Union[str, TextClause],
    params: Optional[Dict[str, Any]] = None,
    engine: Optional[Engine] = None,
    fetch: Optional[str] = "all",
    as_dataframe: bool = False,
):
    """Execute ``query`` using ``engine`` and return rows or rowcount."""

    if fetch not in {None, "all", "one"}:
        raise ValueError("fetch must be one of {'all', 'one', None}")

    engine = engine or get_engine()
    statement = query if isinstance(query, TextClause) else text(query)

    with engine.begin() as connection:
        result = connection.execute(statement, params or {})

        if not result.returns_rows:
            return result.rowcount

        if fetch is None:
            return None

        if fetch == "one":
            row = result.fetchone()
            if as_dataframe:
                import pandas as pd

                if row is None:
                    return pd.DataFrame(columns=result.keys())
                return pd.DataFrame([dict(row._mapping)], columns=result.keys())
            return row

        rows = result.fetchall()
        if as_dataframe:
            import pandas as pd

            data = [dict(row._mapping) for row in rows]
            return pd.DataFrame(data, columns=result.keys())

        return rows
