import io
import json
import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy.engine import Engine

from source.db import create_engine_from_env, execute_query, fast_copy_to_postgres

DEFAULT_CDI_TABLE = "data_science.staging_mercado_financeiro_cdi"
DEFAULT_CDI_FINAL_TABLE = "data_science.mercado_financeiro_cdi"
DEFAULT_TYPE_OVERRIDES: dict[str, str] = {}
SQL_TYPE_MAP = {
    "object": "VARCHAR",
    "string": "VARCHAR",
    "float64": "DOUBLE PRECISION",
    "float32": "DOUBLE PRECISION",
    "int64": "BIGINT",
    "Int64": "BIGINT",
    "int32": "BIGINT",
    "Int32": "BIGINT",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
    "datetime64[ns, tz]": "TIMESTAMP WITH TIME ZONE",
}

load_dotenv()
load_dotenv('/home/joao/cashu/.env')


def generate_create_table_statement(
    df: pd.DataFrame,
    table_name: str,
    *,
    type_overrides: Optional[dict[str, str]] = None,
) -> str:
    overrides = {**DEFAULT_TYPE_OVERRIDES, **(type_overrides or {})}

    columns = []
    for column, dtype in df.dtypes.items():
        column_key = str(column)
        sql_type = overrides.get(column_key)
        if not sql_type:
            sql_type = SQL_TYPE_MAP.get(str(dtype))
        if not sql_type:
            sql_type = "VARCHAR"
        columns.append(f'"{column_key}" {sql_type}')

    cols_sql = ",\n    ".join(columns)
    return f"CREATE TABLE {table_name} (\n    {cols_sql}\n);"


def coleta_cdi(
    start_reference_date: str,
    end_reference_date: str,
    *,
    engine: Engine,
    table_name: str = DEFAULT_CDI_TABLE,
    print_ddl: bool = False,
) -> pd.DataFrame:
    """
    Extrai dados do CDI (Taxa Selic Apurada) via BCB e carrega no staging.
    
    Args:
        start_reference_date: Data inicial formato YYYY-MM-DD.
        end_reference_date: Data final formato YYYY-MM-DD.
    """
    
    # Conversão de datas para formato do BCB (dd/mm/YYYY)
    dt_inicio = datetime.strptime(start_reference_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    dt_fim = datetime.strptime(end_reference_date, "%Y-%m-%d").strftime("%d/%m/%Y")

    url = 'https://www3.bcb.gov.br/novoselic/rest/taxaSelicApurada/pub/exportarCsv'
    
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'accept-language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        'content-type': 'application/x-www-form-urlencoded',
        'origin': 'https://www3.bcb.gov.br',
        'referer': 'https://www3.bcb.gov.br/novoselic/pesquisa-taxa-apurada.jsp',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
    }

    filtro = {
        "dataInicial": dt_inicio,
        "dataFinal": dt_fim
    }

    data = {
        'filtro': json.dumps(filtro),
        'parametrosOrdenacao': '[]',
    }

    print(f"Solicitando dados CDI (Selic) de {dt_inicio} a {dt_fim}...")
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()

    csv_content = response.content.decode('latin1')
    
    try:
        df = pd.read_csv(io.StringIO(csv_content), sep=';', decimal=',', skiprows=1)
    except Exception:
        df = pd.read_csv(io.StringIO(csv_content), sep=';', decimal=',', skiprows=1)

    if df.empty:
        print(f"Nenhum dado retornado para o período {start_reference_date} a {end_reference_date}.")
        return df

    # Padronização de colunas
    # Exemplo de colunas: "Data", "Taxa (% a.a.)", "Fator Diário" ...
    # Vamos normalizar para snake_case
    df = df.dropna(axis=1, how='all')
    
    df.columns = [
        'data',
        'taxa_aa',
        'fator_diario',
        'financeiro',
        'operacoes',
        'taxa_media',
        'taxa_mediana',
        'taxa_modal',
        'desvio_padrao',
        'kurtose'
    ]
    
    # Tratamento de data
    if 'data_apuracao' in df.columns:
        df['data_referencia'] = pd.to_datetime(df['data_apuracao'], format='%d/%m/%Y', errors='coerce')
    elif 'data' in df.columns:
        df['data_referencia'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
    
    df = df.copy()
    df["data_atualizacao"] = datetime.now()
    df["replication_key"] = str(uuid4())

    if print_ddl:
        ddl = generate_create_table_statement(df, table_name)
        print("\n-- DDL Sugerido para staging (CDI) --")
        print(ddl)
        return df

    fast_copy_to_postgres(df, table_name, engine)
    print(f"Dados CDI carregados com sucesso ({len(df)} linhas).")
    
    return df


def ingestao_cdi(
    *,
    engine: Engine,
    start_reference_date: Optional[str] = None,
    end_reference_date: Optional[str] = None,
    target_table: str = DEFAULT_CDI_FINAL_TABLE,
    staging_table: str = DEFAULT_CDI_TABLE
) -> None:
    """
    Ingestão dos dados de CDI do staging para tabela final.
    Idempotente baseada na data de referência.
    """
    
    if not start_reference_date or not end_reference_date:
        print("Datas de inicio e fim necessárias para limpeza do destino.")
        return

    params = {
        "start_date": start_reference_date,
        "end_date": end_reference_date
    }

    print(f"Executando ingestão final de CDI para período {start_reference_date} a {end_reference_date}...")

    
    # 2. Deletar período existente no destino (Idempotência)
    delete_query = f"""
        delete from data_science.mercado_financeiro_cdi
            using (
                select
                    distinct
                    data_referencia
                from data_science.staging_mercado_financeiro_cdi
            ) vw
            where data_science.mercado_financeiro_cdi.data_referencia = vw.data_referencia
    """
    execute_query(engine=engine, query=delete_query, params=params)
    
    insert_query = f"""
        insert into data_science.mercado_financeiro_cdi (
        select
            *
        from data_science.vw_ingestao_mercado_financeiro_cdi
        )
    """
    execute_query(engine=engine, query=insert_query, params=params)
    print("Ingestão de CDI concluída com sucesso.")


def execute_pipeline(parameters: dict) -> None:
    engine = create_engine_from_env()
    
    start_date = parameters.get("data_inicio")
    end_date = parameters.get("data_fim")
    print_ddl = parameters.get("print_ddl", False)
    
    # Opções de pipeline
    # 1: Coleta + Ingestão
    # 2: Apenas Coleta
    # 3: Apenas Ingestão
    
    opt = parameters.get("opt", "1")
    
    if not start_date or not end_date:
        raise ValueError("Parâmetros 'data_inicio' e 'data_fim' são obrigatórios.")

    print("--------------------------------")
    print("Iniciando Pipeline CDI (Mercado Financeiro)...")
    print(f"Período: {start_date} a {end_date}")
    print("--------------------------------")

    if opt in ["1", "2"]:
        coleta_cdi(
            start_reference_date=start_date,
            end_reference_date=end_date,
            engine=engine,
            print_ddl=print_ddl
        )

    if opt in ["1", "3"]:
        ingestao_cdi(
            engine=engine,
            start_reference_date=start_date,
            end_reference_date=end_date
        )

    print("--------------------------------")
    print("Pipeline CDI concluído.")
    print("--------------------------------")


if __name__ == "__main__":
    # Exemplo de uso local
    execute_pipeline({
        "opt": "3",
        "data_inicio": "2015-12-01",
        "data_fim": "2025-11-23",
        #"print_ddl": True
    })

