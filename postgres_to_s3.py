"""
Simple EL (Extract-Load) pipeline: PostgreSQL -> Parquet/JSON on S3.

Zero-copy extraction using DuckDB's postgres_scanner and direct S3 write.
Supports Parquet (default) and JSON (NDJSON) for tables with JSONB columns.
"""

import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

import duckdb
import snowflake.connector
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


def get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Create Snowflake connection from environment variables.
    
    Uses Snowflake environment variables:
    - SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
    - SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
    
    Returns:
        Snowflake connection.
    """
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
        user=os.getenv("SNOWFLAKE_USER", ""),
        #password=os.getenv("SNOWFLAKE_PASSWORD", ""),
        private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE", ""),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "DBT_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", ""),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "bronze"),
    )


def execute_snowflake_task(task_name: str) -> None:
    """Execute Snowflake task to load data from S3 stage.
    
    Args:
        task_name: Fully qualified Snowflake task name.
    """
    print(f"Executing Snowflake task: {task_name}")
    
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        # Execute the task
        cursor.execute(f"EXECUTE TASK {task_name};")
        
        print(f"Snowflake task {task_name} executed successfully.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error executing Snowflake task {task_name}: {e}")
        raise


def coleta_postgres_to_s3(
    schema: str,
    table: str,
    s3_path: str,
    *,
    partition_divisor: int = DEFAULT_PARTITION_DIVISOR,
    execution_id: Optional[str] = None,
    use_row_number: bool = False,
    execute_task: bool = True,
    task_name: Optional[str] = None,
    output_format: str = "parquet",
) -> None:
    """Zero-copy PostgreSQL to S3 using DuckDB.
    
    Extracts data directly from PostgreSQL and writes partitioned files
    to S3 without loading data into Python memory.
    
    Args:
        schema: PostgreSQL schema name (e.g., 'public').
        table: PostgreSQL table name.
        s3_path: S3 destination path (e.g., 's3://bucket/path/table').
        partition_divisor: Number of rows per partition (default 100,000).
        execution_id: UUID for tracking (generated if not provided).
        use_row_number: If True, use ROW_NUMBER() for chunk_id instead of id column.
        execute_task: If True, executes Snowflake task after upload.
        task_name: Fully qualified task name to execute. If not provided,
            defaults to bronze.raw_cashu_app__{table}.
        output_format: 'parquet' (default) or 'json' (NDJSON, for tables with JSONB columns).
    """
    exec_id = execution_id or generate_execution_id()
    pg_conn = get_postgres_connection_string()
    source_table = f"{schema}.{table}"
    
    print("=" * 60)
    print("Pipeline Extract & Load (DuckDB Zero-Copy)")
    print(f"Execution ID: {exec_id}")
    print(f"Source: {source_table}")
    print(f"Destination: {s3_path}")
    print(f"Format: {output_format.upper()}")
    print(f"Partition size: {partition_divisor:,} rows")
    print("=" * 60)
    
    conn = get_duckdb_connection()
    
    if use_row_number:
        chunk_expr = f"(ROW_NUMBER() OVER () / {partition_divisor})::int"
    else:
        chunk_expr = f"(id / {partition_divisor})::int"
    
    print("Executing zero-copy transfer...")
    start_time = datetime.now()
    
    if output_format == "json":
        conn.execute(f"""
            CREATE TEMP TABLE _export AS
            SELECT
                {chunk_expr} AS _chunk_id,
                '{exec_id}' AS _execution_id,
                '{source_table}' AS _source_table,
                current_timestamp AS _loaded_at,
                *
            FROM postgres_scan('{pg_conn}', '{schema}', '{table}')
        """)
        
        chunks = conn.execute(
            "SELECT DISTINCT _chunk_id FROM _export ORDER BY _chunk_id"
        ).fetchall()
        
        print(f"Writing {len(chunks)} JSON chunk(s)...")
        for (chunk_id,) in chunks:
            dest = f"{s3_path}/chunk_{chunk_id}.json"
            conn.execute(f"""
                COPY (
                    SELECT * EXCLUDE (_chunk_id) FROM _export
                    WHERE _chunk_id = {chunk_id}
                ) TO '{dest}' (FORMAT JSON);
            """)
            print(f"  chunk {chunk_id} -> {dest}")
        
        conn.execute("DROP TABLE _export")
    else:
        query = f"""
            COPY (
                SELECT
                    {chunk_expr} AS chunk_id,
                    '{exec_id}' AS _execution_id,
                    '{source_table}' AS _source_table,
                    current_timestamp AS _loaded_at,
                    *
                FROM postgres_scan('{pg_conn}', '{schema}', '{table}')
            ) TO '{s3_path}' (FORMAT PARQUET, PARTITION_BY (chunk_id), OVERWRITE_OR_IGNORE);
        """
        conn.execute(query)
    
    elapsed = datetime.now() - start_time
    print(f"Transfer completed in {elapsed.total_seconds():.2f} seconds")
    print("=" * 60)
    
    conn.close()
    
    if execute_task:
        resolved_task_name = task_name or f"bronze.raw_cashu_app__{table}"
        execute_snowflake_task(resolved_task_name)


def execute_pipeline(parameters: dict) -> None:
    """Execute extract and load pipeline: PostgreSQL -> S3.
    
    Args:
        parameters: Dict with keys:
            - schema: PostgreSQL schema name (default: 'public')
            - table: Source table name (required)
            - s3_path: Full S3 path (required, e.g., 's3://bucket/path/table')
            - partition_divisor: Rows per partition (default 100,000)
            - use_row_number: If True, use ROW_NUMBER() instead of id column
            - execute_task: If True, executes Snowflake task after upload
            - task_name: Fully qualified Snowflake task name to execute
            - output_format: 'parquet' (default) or 'json'
    """
    execution_id = generate_execution_id()
    
    schema = parameters.get("schema", "public")
    table = parameters.get("table")
    s3_path = parameters.get("s3_path")
    partition_divisor = parameters.get("partition_divisor", DEFAULT_PARTITION_DIVISOR)
    use_row_number = parameters.get("use_row_number", False)
    execute_task = parameters.get("execute_task", True)
    task_name = parameters.get("task_name")
    output_format = parameters.get("output_format", "parquet")
    
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
        execute_task=execute_task,
        task_name=task_name,
        output_format=output_format,
    )
    
    print("Pipeline completed.")


if __name__ == "__main__":
    S3_BASE = "s3://cashu-data-stack/el/cashu_postgres"
    
    # (schema, table, use_row_number, output_format)
    tables = [
        #("data_science", "cadastro_calendario_anbima", True, "parquet", 1_250_000),
        #("data_science", "cadastro_modelos_experimentos", True, "parquet", 1_250_000),
        ("data_science", "movimentacao_modelos_experimentos", True, "parquet", 1_250_000),
        ("public", "cnabs", True, "parquet", 1_250_000),
        ("public", "cnab_operations", True, "parquet", 1_250_000),
        ("public", "customers", True, "parquet", 1_250_000),
        ("public", "corporates", True, "parquet", 1_250_000),
        ("public", "businesses", True, "parquet", 1_250_000),
        ("public", "orders", True, "parquet", 1_250_000),
        ("public", "order_installment_charge_backs", True, "parquet", 1_250_000),
        ("public", "invoice_financing_items", True, "parquet", 1_250_000),
        ("public", "order_installments", True, "parquet",1_250_000),
        ("public", "invoice_financings", True, "parquet", 1_250_000),
        ("public", "bank_billets", True, "parquet", 1_250_000),
        ("public", "invoices", True, "parquet", 1_250_000),
        ("public", "invoice_receivables", True, "parquet", 1_250_000),
        ("public", "recommendations", True, "parquet", 1_250_000),
        ("public", "integration_receivable", True, "parquet", 1_250_000),
        ("public", "core_data_scrs", True, "json", 25_000),
    ]
    
    for schema, table, use_row_number, output_format, partition_divisor in tables:
        try:
            execute_pipeline(
                parameters={
                    "schema": schema,
                    "table": table,
                    "s3_path": f"{S3_BASE}/{table}",
                    "partition_divisor": partition_divisor,
                    "use_row_number": use_row_number,
                    "execute_task": True,
                    "task_name": f"bronze.raw_cashu_app__{table}",
                    "output_format": output_format,
                }
            )
            print(f"Completed: {schema}.{table}\n")
        except Exception as e:
            print(f"Error processing {schema}.{table}: {e}\n")