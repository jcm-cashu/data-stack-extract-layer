import os
from datetime import datetime
from io import BytesIO
from typing import Optional
from uuid import uuid4

import boto3
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

from source.kanastra_api import APIConfig, extract_dataset


DEFAULT_BUCKET = "cashu-data-stack"
DEFAULT_HOLDINGS_KEY = "el/kanastra/estoque"
DEFAULT_ACQUISITIONS_KEY = "el/kanastra/aquisicoes"
DEFAULT_LIQUIDATIONS_KEY = "el/kanastra/liquidacoes"
DEFAULT_SLUG = "fidc-inova-credtech-iii"
DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_COMPRESSION = "snappy"

load_dotenv("/home/joao/cashu/.env")
load_dotenv()


def generate_execution_id() -> str:
    """Generate a unique execution ID for tracking parallel extractions."""
    return str(uuid4())


def print_schema_from_df(df: pd.DataFrame, name: str) -> None:
    """Print inferred schema from DataFrame.

    Args:
        df: DataFrame to inspect.
        name: Name to display in the output.
    """
    print(f"\n-- Schema for: {name}")
    print(f"-- Rows: {len(df):,}")
    print("-" * 50)
    for col in df.columns:
        dtype = df[col].dtype
        print(f"  {col}: {dtype}")
    print("-" * 50)


def get_s3_client(region: Optional[str] = None):
    """Cria cliente S3 usando credenciais do ambiente."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region or os.getenv("AWS_REGION", DEFAULT_AWS_REGION),
    )


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
        password=os.getenv("SNOWFLAKE_PASSWORD", ""),
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


def load_s3_parquet(
    df: pd.DataFrame,
    bucket: str,
    key: str,
    *,
    region: Optional[str] = None,
    compression: str = DEFAULT_COMPRESSION,
) -> str:
    """Carrega DataFrame no S3 como Parquet.

    Args:
        df: DataFrame para upload.
        bucket: Nome do bucket S3.
        key: Caminho do objeto no bucket.
        region: Região AWS.
        compression: Algoritmo de compressão ('snappy', 'gzip', 'zstd', None).

    Returns:
        URI S3 do arquivo carregado.
    """
    s3_client = get_s3_client(region)

    print(f"Carregando dados no S3...")
    print(f"  Bucket: {bucket}")
    print(f"  Key: {key}")

    buffer = BytesIO()
    df.to_parquet(buffer, index=False, compression=compression, engine="pyarrow")
    buffer.seek(0)

    s3_client.upload_fileobj(buffer, bucket, key)

    s3_uri = f"s3://{bucket}/{key}"
    print(f"  URI: {s3_uri}")

    return s3_uri


def coleta_holdings_kanastra(
    reference_date: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    key_prefix: str = DEFAULT_HOLDINGS_KEY,
    slug: Optional[str] = None,
    fmt: str = "CSV",
    region: Optional[str] = None,
    print_ddl: bool = False,
    execution_id: Optional[str] = None,
    execute_task: bool = False,
    task_name: Optional[str] = None,
) -> pd.DataFrame:
    """Extrai holdings via API v3 e carrega como Parquet no S3.

    Args:
        reference_date: Data de referência (YYYY-MM-DD).
        bucket: Nome do bucket S3.
        key_prefix: Prefixo do caminho no S3.
        slug: Slug da operação.
        fmt: Formato de retorno da API (CSV ou JSON).
        region: Região AWS.
        print_ddl: Se True, imprime o schema e não faz upload.
        execution_id: UUID único para rastrear execuções paralelas.
        execute_task: If True, executes Snowflake task after upload.
        task_name: Fully qualified task name to execute.
    """
    config = APIConfig(
        base_url=os.getenv("KANASTRA_BASE_URL"),
        client_id=os.getenv("KANASTRA_CLIENT_ID"),
        client_secret=os.getenv("KANASTRA_CLIENT_SECRET"),
    )

    effective_slug = slug or DEFAULT_SLUG
    holdings = extract_dataset(
        "holdings_v3",
        config=config,
        reference_date=reference_date,
        slug=effective_slug,
        fmt=fmt,
    )

    df = holdings if isinstance(holdings, pd.DataFrame) else pd.DataFrame(holdings)

    if df.empty:
        print(f"Nenhum dado de holdings retornado para slug '{effective_slug}' na referência {reference_date}.")
        return df

    # Add metadata columns
    df["_execution_id"] = execution_id or generate_execution_id()
    df["_slug"] = effective_slug
    df["_loaded_at"] = datetime.now().isoformat()

    if print_ddl:
        key = f"{key_prefix}/{reference_date}.parquet"
        print_schema_from_df(df, f"s3://{bucket}/{key}")
        return df

    # Save to S3 with date-based filename
    key = f"{key_prefix}/{reference_date}.parquet"
    load_s3_parquet(df, bucket, key, region=region)
    print(f"Holdings carregado: {len(df)} linhas -> s3://{bucket}/{key}")
    
    if execute_task:
        if not task_name:
            raise ValueError("Parâmetro 'task_name' é obrigatório quando execute_task=True.")
        execute_snowflake_task(task_name)

    return df


def coleta_aquisicoes_kanastra(
    start_reference_date: str,
    end_reference_date: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    key_prefix: str = DEFAULT_ACQUISITIONS_KEY,
    slug: Optional[str] = None,
    page_size: int = 500,
    region: Optional[str] = None,
    print_ddl: bool = False,
    execution_id: Optional[str] = None,
    execute_task: bool = False,
    task_name: Optional[str] = None,
) -> pd.DataFrame:
    """Extrai aquisições via API e carrega como Parquet no S3.

    Args:
        start_reference_date: Data inicial (YYYY-MM-DD).
        end_reference_date: Data final (YYYY-MM-DD).
        bucket: Nome do bucket S3.
        key_prefix: Prefixo do caminho no S3.
        slug: Slug da operação.
        page_size: Tamanho da página para paginação da API.
        region: Região AWS.
        print_ddl: Se True, imprime o schema e não faz upload.
        execution_id: UUID único para rastrear execuções paralelas.
        execute_task: If True, executes Snowflake task after upload.
        task_name: Fully qualified task name to execute.
    """
    config = APIConfig(
        base_url=os.getenv("KANASTRA_BASE_URL"),
        client_id=os.getenv("KANASTRA_CLIENT_ID"),
        client_secret=os.getenv("KANASTRA_CLIENT_SECRET"),
    )

    effective_slug = slug or DEFAULT_SLUG
    payload = extract_dataset(
        "acquisitions",
        config=config,
        start_reference_date=start_reference_date,
        end_reference_date=end_reference_date,
        slug=effective_slug,
        page_size=page_size,
    )

    df = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    if df.empty:
        print(f"Nenhuma aquisição retornada para {start_reference_date} a {end_reference_date} (slug {effective_slug}).")
        return df

    # Add metadata columns
    df["_execution_id"] = execution_id or generate_execution_id()
    df["_slug"] = effective_slug
    df["_loaded_at"] = datetime.now().isoformat()

    if print_ddl:
        key = f"{key_prefix}/{end_reference_date}.parquet"
        print_schema_from_df(df, f"s3://{bucket}/{key}")
        return df

    # Save to S3 with date-based filename (using end_reference_date)
    key = f"{key_prefix}/{end_reference_date}.parquet"
    load_s3_parquet(df, bucket, key, region=region)
    print(f"Aquisições carregadas: {len(df)} linhas -> s3://{bucket}/{key}")
    
    if execute_task:
        if not task_name:
            raise ValueError("Parâmetro 'task_name' é obrigatório quando execute_task=True.")
        execute_snowflake_task(task_name)

    return df


def coleta_liquidacoes_kanastra(
    start_reference_date: str,
    end_reference_date: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    key_prefix: str = DEFAULT_LIQUIDATIONS_KEY,
    slug: Optional[str] = None,
    page_size: int = 500,
    region: Optional[str] = None,
    print_ddl: bool = False,
    execution_id: Optional[str] = None,
    execute_task: bool = False,
    task_name: Optional[str] = None,
) -> pd.DataFrame:
    """Extrai liquidações via API e carrega como Parquet no S3.

    Args:
        start_reference_date: Data inicial (YYYY-MM-DD).
        end_reference_date: Data final (YYYY-MM-DD).
        bucket: Nome do bucket S3.
        key_prefix: Prefixo do caminho no S3.
        slug: Slug da operação.
        page_size: Tamanho da página para paginação da API.
        region: Região AWS.
        print_ddl: Se True, imprime o schema e não faz upload.
        execution_id: UUID único para rastrear execuções paralelas.
        execute_task: If True, executes Snowflake task after upload.
        task_name: Fully qualified task name to execute.
    """
    config = APIConfig(
        base_url=os.getenv("KANASTRA_BASE_URL"),
        client_id=os.getenv("KANASTRA_CLIENT_ID"),
        client_secret=os.getenv("KANASTRA_CLIENT_SECRET"),
    )

    effective_slug = slug or DEFAULT_SLUG
    payload = extract_dataset(
        "liquidations",
        config=config,
        start_reference_date=start_reference_date,
        end_reference_date=end_reference_date,
        slug=effective_slug,
        page_size=page_size,
    )

    df = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    if df.empty:
        print(f"Nenhuma liquidação retornada para {start_reference_date} a {end_reference_date} (slug {effective_slug}).")
        return df

    # Add metadata columns
    df["_execution_id"] = execution_id or generate_execution_id()
    df["_slug"] = effective_slug
    df["_loaded_at"] = datetime.now().isoformat()

    if print_ddl:
        key = f"{key_prefix}/{end_reference_date}.parquet"
        print_schema_from_df(df, f"s3://{bucket}/{key}")
        return df

    # Save to S3 with date-based filename (using end_reference_date)
    key = f"{key_prefix}/{end_reference_date}.parquet"
    load_s3_parquet(df, bucket, key, region=region)
    print(f"Liquidações carregadas: {len(df)} linhas -> s3://{bucket}/{key}")
    
    if execute_task:
        if not task_name:
            raise ValueError("Parâmetro 'task_name' é obrigatório quando execute_task=True.")
        execute_snowflake_task(task_name)

    return df


def execute_pipeline(parameters: dict) -> None:
    """Execute extract and load pipeline to S3.

    Args:
        parameters: Dict with keys:
            - opt: "1" (holdings), "2" (aquisicoes), "3" (liquidacoes)
            - data_referencia: Required for holdings (YYYY-MM-DD)
            - data_inicio/data_fim: Required for aquisicoes/liquidacoes
            - slug: Optional operation slug
            - page_size: Optional pagination size (default 500)
            - bucket: S3 bucket name (default: cashu-data-stack)
            - region: AWS region (default: us-east-1)
            - print_ddl: If True, print schema instead of uploading
            - execute_task: If True, executes Snowflake task after upload
            - task_name: Fully qualified Snowflake task name to execute
    """
    # Generate unique execution ID for this pipeline run
    execution_id = generate_execution_id()

    reference_date = parameters.get("data_referencia")
    start_reference_date = parameters.get("data_inicio")
    end_reference_date = parameters.get("data_fim")
    slug = parameters.get("slug") or DEFAULT_SLUG
    fmt = parameters.get("formato", "CSV")
    page_size = parameters.get("page_size", 500)
    bucket = parameters.get("bucket", DEFAULT_BUCKET)
    region = parameters.get("region")
    print_ddl = parameters.get("print_ddl", False)
    execute_task = parameters.get("execute_task", False)
    task_name = parameters.get("task_name")

    options = {
        "1": coleta_holdings_kanastra,
        "2": coleta_aquisicoes_kanastra,
        "3": coleta_liquidacoes_kanastra,
    }

    opt = parameters.get("opt", "1")
    if opt not in options:
        raise ValueError(f"Opção inválida: {opt}. Use 1=holdings, 2=aquisicoes, 3=liquidacoes")

    print("--------------------------------")
    print("Pipeline Extract & Load (S3)")
    print(f"Execution ID: {execution_id}")
    if print_ddl:
        print("Mode: Schema/DDL Generation (no upload)")
    else:
        print(f"Bucket: {bucket}")
    print("--------------------------------")

    fn = options[opt]
    if fn is coleta_holdings_kanastra:
        if not reference_date:
            raise ValueError("Parâmetro 'data_referencia' é obrigatório.")
        fn(
            reference_date,
            bucket=bucket,
            slug=slug,
            fmt=fmt,
            region=region,
            print_ddl=print_ddl,
            execution_id=execution_id,
            execute_task=execute_task,
            task_name=task_name,
        )
    else:
        if not start_reference_date or not end_reference_date:
            raise ValueError("Parâmetros 'data_inicio' e 'data_fim' são obrigatórios.")
        fn(
            start_reference_date,
            end_reference_date,
            bucket=bucket,
            slug=slug,
            page_size=page_size,
            region=region,
            print_ddl=print_ddl,
            execution_id=execution_id,
            execute_task=execute_task,
            task_name=task_name,
        )

    print("--------------------------------")
    print("Pipeline concluído.")
    print("--------------------------------")


if __name__ == "__main__":
    execute_pipeline(
        parameters={
            "opt": "3",
            #"data_referencia": "2026-01-20",
            "data_inicio": "2025-02-01",
            "data_fim": "2026-02-09",
            "slug": DEFAULT_SLUG,
            "page_size": 1000,
            "execute_task": True,
            "task_name": "bronze.raw_kanastra__aquisicoes",
            #"task_name": "bronze.raw_kanastra__liquidacoes",
            #"print_ddl": True,
        }
    )