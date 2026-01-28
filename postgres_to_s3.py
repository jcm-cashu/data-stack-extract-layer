"""
Simple EL (Extract-Load) pipeline: PostgreSQL -> Parquet on S3.

Zero-copy extraction using DuckDB's postgres_scanner and direct S3 write.
"""

import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

import duckdb
from dotenv import load_dotenv

load_dotenv("/home/joao/cashu/.env")
load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_PARTITION_DIVISOR = 100_000  # rows per partition chunk


def generate_execution_id() -> str:
    """Generate a unique execution ID for tracking."""
    return str(uuid4())


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create and configure DuckDB connection with required extensions.
    
    Loads postgres_scanner for PostgreSQL access and httpfs for S3 access.
    Configures AWS credentials from environment variables.
    
    Returns:
        Configured DuckDB connection.
    """
    conn = duckdb.connect()
    
    # Install and load required extensions
    conn.execute("INSTALL postgres_scanner; LOAD postgres_scanner;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    
    # Configure S3 credentials
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
    
    conn.execute(f"""
        SET s3_access_key_id = '{aws_access_key}';
        SET s3_secret_access_key = '{aws_secret_key}';
        SET s3_region = '{aws_region}';
    """)
    
    # Performance settings
    conn.execute("""
        SET preserve_insertion_order = false;
        SET threads = 4;
        SET memory_limit = '40GB';
    """)
    
    return conn


def get_postgres_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables.
    
    Uses standard PostgreSQL environment variables:
    - PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
    
    Returns:
        PostgreSQL connection string for DuckDB postgres_scanner.
    """
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    user = os.getenv("PG_USER", "")
    password = os.getenv("PG_PASS", "")
    database = os.getenv("PG_DB", "")
    
    return f"dbname={database} user={user} password={password} host={host} port={port}"


def coleta_postgres_to_s3(
    schema: str,
    table: str,
    s3_path: str,
    *,
    partition_divisor: int = DEFAULT_PARTITION_DIVISOR,
    execution_id: Optional[str] = None,
    use_row_number: bool = False,
) -> None:
    """Zero-copy PostgreSQL to S3 Parquet using DuckDB.
    
    Extracts data directly from PostgreSQL and writes partitioned Parquet
    files to S3 without loading data into Python memory.
    
    Args:
        schema: PostgreSQL schema name (e.g., 'public').
        table: PostgreSQL table name.
        s3_path: S3 destination path (e.g., 's3://bucket/path/table').
        partition_divisor: Number of rows per partition (default 100,000).
        execution_id: UUID for tracking (generated if not provided).
        use_row_number: If True, use ROW_NUMBER() for chunk_id instead of id column.
    """
    exec_id = execution_id or generate_execution_id()
    pg_conn = get_postgres_connection_string()
    source_table = f"{schema}.{table}"
    
    print("=" * 60)
    print("Pipeline Extract & Load (DuckDB Zero-Copy)")
    print(f"Execution ID: {exec_id}")
    print(f"Source: {source_table}")
    print(f"Destination: {s3_path}")
    print(f"Partition size: {partition_divisor:,} rows")
    print("=" * 60)
    
    conn = get_duckdb_connection()
    
    # Build chunk_id expression based on whether table has id column
    if use_row_number:
        chunk_expr = f"(ROW_NUMBER() OVER () / {partition_divisor})::int"
    else:
        chunk_expr = f"(id / {partition_divisor})::int"
    
    # Execute zero-copy transfer with metadata columns
    query = f"""
        COPY (
            SELECT
                {chunk_expr} AS chunk_id,
                '{exec_id}' AS _execution_id,
                '{source_table}' AS _source_table,
                current_timestamp AS _loaded_at,
                *
            FROM postgres_scan(
                '{pg_conn}',
                '{schema}',
                '{table}'
            )
        ) TO '{s3_path}' (FORMAT PARQUET, PARTITION_BY (chunk_id), OVERWRITE_OR_IGNORE);
    """
    
    print("Executing zero-copy transfer...")
    start_time = datetime.now()
    
    conn.execute(query)
    
    elapsed = datetime.now() - start_time
    print(f"Transfer completed in {elapsed.total_seconds():.2f} seconds")
    print("=" * 60)
    
    conn.close()


def execute_pipeline(parameters: dict) -> None:
    """Execute extract and load pipeline: PostgreSQL -> S3.
    
    Args:
        parameters: Dict with keys:
            - schema: PostgreSQL schema name (default: 'public')
            - table: Source table name (required)
            - s3_path: Full S3 path (required, e.g., 's3://bucket/path/table')
            - partition_divisor: Rows per partition (default 100,000)
            - use_row_number: If True, use ROW_NUMBER() instead of id column
    """
    execution_id = generate_execution_id()
    
    schema = parameters.get("schema", "public")
    table = parameters.get("table")
    s3_path = parameters.get("s3_path")
    partition_divisor = parameters.get("partition_divisor", DEFAULT_PARTITION_DIVISOR)
    use_row_number = parameters.get("use_row_number", False)
    
    if not table:
        raise ValueError("Parameter 'table' is required.")
    if not s3_path:
        raise ValueError("Parameter 's3_path' is required.")
    
    coleta_postgres_to_s3(
        schema=schema,
        table=table,
        s3_path=s3_path,
        partition_divisor=partition_divisor,
        execution_id=execution_id,
        use_row_number=use_row_number,
    )
    
    print("Pipeline completed.")


if __name__ == "__main__":
    # S3 bucket base path
    S3_BASE = "s3://cashu-data-stack/el/cashu_postgres"
    
    # Tables to extract (schema, table, use_row_number)
    tables = [
        # Small/medium tables with id column
        #("data_science", "cadastro_calendario_anbima", True),
        #("public", "customers", True),
        #("public", "corporates", True),
        #("public", "businesses", True),
        #("public", "orders", True),
        #("public", "invoice_financing_items", True),
        #("public", "order_installments", True),
        ("public", "invoice_financings", True),
        #("public", "integration_receivable", True),  # uses ROW_NUMBER
        #("public", "bank_billets", True),
        #("public", "recommendations", True),
        #("public", "invoices", True),
    ]
    
    # Execute pipeline for each table
    for schema, table, use_row_number in tables:
        try:
            execute_pipeline(
                parameters={
                    "schema": schema,
                    "table": table,
                    "s3_path": f"{S3_BASE}/{table}",
                    "partition_divisor": 1_250_000,
                    "use_row_number": use_row_number,
                }
            )
            print(f"Completed: {schema}.{table}\n")
        except Exception as e:
            print(f"Error processing {schema}.{table}: {e}\n")
