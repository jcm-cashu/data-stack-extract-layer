"""
Simple EL (Extract-Load) pipeline: CSV -> CSV on S3.

Reads a CSV file from a local folder, standardizes it, writes it to S3 as UTF-8
CSV with comma separator, and optionally executes a Snowflake task after the upload.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import duckdb
import snowflake.connector
from dotenv import load_dotenv

load_dotenv("/Users/joaomagalhaes/cashu/.env")
#load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_OUTPUT_DELIMITER = ","
DEFAULT_OUTPUT_ENCODING = "utf-8"


def generate_execution_id() -> str:
    """Generate a unique execution ID for tracking."""
    return str(uuid4())


def sql_literal(value: str) -> str:
    """Escape string values for SQL literals."""
    return value.replace("'", "''")


def bool_to_sql(value: bool) -> str:
    """Convert Python booleans to DuckDB SQL booleans."""
    return "true" if value else "false"


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create and configure DuckDB connection with required extensions."""
    conn = duckdb.connect()

    conn.execute("INSTALL httpfs; LOAD httpfs;")

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)

    conn.execute(
        f"""
        SET s3_access_key_id = '{sql_literal(aws_access_key)}';
        SET s3_secret_access_key = '{sql_literal(aws_secret_key)}';
        SET s3_region = '{sql_literal(aws_region)}';
    """
    )

    conn.execute(
        """
        SET preserve_insertion_order = false;
        SET threads = 4;
        SET memory_limit = '40GB';
    """
    )

    return conn


print(os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE"))
def get_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Create Snowflake connection from environment variables."""
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
    """Execute a Snowflake task after the S3 upload."""
    print(f"Executing Snowflake task: {task_name}")

    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(f"EXECUTE TASK {task_name};")
        print(f"Snowflake task {task_name} executed successfully.")
        cursor.close()
        conn.close()
    except Exception as exc:
        print(f"Error executing Snowflake task {task_name}: {exc}")
        raise


def build_source_file_path(source_folder: str, file_name: str) -> Path:
    """Build and validate the local CSV source file path."""
    source_path = Path(source_folder).expanduser() / file_name

    if not source_path.exists():
        raise FileNotFoundError(f"CSV file not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Source path is not a file: {source_path}")

    return source_path.resolve()


def print_inferred_csv_ddl(
    conn: duckdb.DuckDBPyConnection,
    source_file: Path,
    delimiter: str,
    header: bool,
    encoding: str,
    ignore_errors: bool,
) -> None:
    """Print the inferred DuckDB schema for the CSV file."""
    escaped_source_file = sql_literal(str(source_file))
    escaped_delimiter = sql_literal(delimiter)
    escaped_encoding = sql_literal(encoding)

    describe_query = f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{escaped_source_file}',
            delim = '{escaped_delimiter}',
            header = {bool_to_sql(header)},
            encoding = '{escaped_encoding}',
            ignore_errors = {bool_to_sql(ignore_errors)}
        );
    """

    schema_rows = conn.execute(describe_query).fetchall()
    ddl_lines = ['CREATE TABLE inferred_csv_schema (']

    for index, row in enumerate(schema_rows):
        column_name = row[0]
        column_type = row[1]
        suffix = "," if index < len(schema_rows) - 1 else ""
        ddl_lines.append(f'  "{column_name}" {column_type}{suffix}')

    ddl_lines.append(");")

    print("Inferred CSV schema:")
    print("\n".join(ddl_lines))
    print("=" * 60)


def coleta_csv_to_s3(
    source_folder: str,
    file_name: str,
    s3_path: str,
    *,
    delimiter: str = ",",
    header: bool = True,
    encoding: str = "utf-8",
    ignore_errors: bool = False,
    print_ddl: bool = False,
    execute_task: bool = True,
    task_name: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> None:
    """Read a CSV file from disk and write it to S3 as normalized CSV."""
    exec_id = execution_id or generate_execution_id()
    source_file = build_source_file_path(source_folder, file_name)

    print("=" * 60)
    print("Pipeline Extract & Load (DuckDB CSV -> S3)")
    print(f"Execution ID: {exec_id}")
    print(f"Source file: {source_file}")
    print(f"Destination: {s3_path}")
    print(f"Header: {header}")
    print(f"Input delimiter: {delimiter}")
    print(f"Input encoding: {encoding}")
    print(f"Output delimiter: {DEFAULT_OUTPUT_DELIMITER}")
    print(f"Output encoding: {DEFAULT_OUTPUT_ENCODING}")
    print(f"Ignore errors: {ignore_errors}")
    print(f"Print DDL: {print_ddl}")
    print("=" * 60)

    conn = get_duckdb_connection()

    escaped_source_file = sql_literal(str(source_file))
    escaped_s3_path = sql_literal(s3_path)
    escaped_exec_id = sql_literal(exec_id)
    escaped_delimiter = sql_literal(delimiter)
    escaped_encoding = sql_literal(encoding)

    if print_ddl:
        print_inferred_csv_ddl(
            conn=conn,
            source_file=source_file,
            delimiter=delimiter,
            header=header,
            encoding=encoding,
            ignore_errors=ignore_errors,
        )

    query = f"""
        COPY (
            SELECT
                '{escaped_exec_id}' AS _execution_id,
                current_timestamp AS _loaded_at,
                *
            FROM read_csv_auto(
                '{escaped_source_file}',
                delim = '{escaped_delimiter}',
                header = {bool_to_sql(header)},
                encoding = '{escaped_encoding}',
                ignore_errors = {bool_to_sql(ignore_errors)}
            )
        ) TO '{escaped_s3_path}' (
            FORMAT CSV,
            HEADER,
            DELIMITER '{DEFAULT_OUTPUT_DELIMITER}',
            OVERWRITE_OR_IGNORE
        );
    """

    print("Executing transfer to S3...")
    start_time = datetime.now()

    conn.execute(query)

    elapsed = datetime.now() - start_time
    print(f"Transfer completed in {elapsed.total_seconds():.2f} seconds")
    print("=" * 60)

    conn.close()

    if execute_task:
        if not task_name:
            raise ValueError("Parameter 'task_name' is required when execute_task=True.")
        execute_snowflake_task(task_name)


def execute_pipeline(parameters: dict) -> None:
    """Execute extract and load pipeline: CSV -> normalized CSV on S3."""
    execution_id = generate_execution_id()

    source_folder = parameters.get("source_folder")
    file_name = parameters.get("file_name")
    s3_path = parameters.get("s3_path")
    delimiter = parameters.get("delimiter", ",")
    header = parameters.get("header", True)
    encoding = parameters.get("encoding", "utf-8")
    ignore_errors = parameters.get("ignore_errors", False)
    print_ddl = parameters.get("print_ddl", False)
    execute_task = parameters.get("execute_task", True)
    task_name = parameters.get("task_name")

    if not source_folder:
        raise ValueError("Parameter 'source_folder' is required.")
    if not file_name:
        raise ValueError("Parameter 'file_name' is required.")
    if not s3_path:
        raise ValueError("Parameter 's3_path' is required.")

    coleta_csv_to_s3(
        source_folder=source_folder,
        file_name=file_name,
        s3_path=s3_path,
        delimiter=delimiter,
        header=header,
        encoding=encoding,
        ignore_errors=ignore_errors,
        print_ddl=print_ddl,
        execute_task=execute_task,
        task_name=task_name,
        execution_id=execution_id,
    )

    print("Pipeline completed.")


if __name__ == "__main__":
    file_name = "estoque_20260304.csv"
    source_folder = f"/Users/joaomagalhaes/Downloads"
    execute_pipeline(
        parameters={
            "source_folder": source_folder,
            "file_name": file_name,
            "s3_path": f"s3://cashu-data-stack/el/fromtis/estoque/{file_name}",
            "delimiter": ";",
            "header": True,
            "encoding": "latin-1",
            "ignore_errors": False,
            "print_ddl": False,
            "execute_task": True,
            "task_name": "bronze.raw_fromtis__estoque",
        }
    )
