import os
import boto3
from typing import List, Dict

REDSHIFT_SCHEMA = os.getenv("REDSHIFT_SCHEMA", 'public')
AWS_REGION = os.getenv("AWS_REGION")
REDSHIFT_WORKGROUP_NAME = os.getenv("REDSHIFT_WORKGROUP_NAME")
REDSHIFT_DATABASE = os.getenv("REDSHIFT_DATABASE")
REDSHIFT_AWSCATALOG_DATABASE=os.getenv("REDSHIFT_AWSCATALOG_DATABASE", '')
DB_PLATFORM = "AWS Redshift"
DB_SPECIFICS = ""

redshift_client = boto3.client("redshift-data", region_name=AWS_REGION)

def get_columns():
    """
    Returns the result of get_native_columns concatenated with get_external_columns.
    Signature matches get_native_columns.
    """
    native = get_native_columns()
    external = get_external_columns()
    return native + external

def get_tables():
    """
    Returns the result of get_native_tables concatenated with get_external_tables.
    Signature matches get_native_tables.
    """
    native = get_native_tables()
    external = get_external_tables()
    return native + external

def execute_redshift_query(sql: str) -> List[Dict[str, str]]:
    """
    Helper to execute a SQL query against Redshift and return results as a list of dicts.
    """
    try:
        res = redshift_client.execute_statement(
            WorkgroupName=REDSHIFT_WORKGROUP_NAME,
            Database=REDSHIFT_DATABASE,
            Sql=sql
        )
        query_id = res["Id"]
        while True:
            status = redshift_client.describe_statement(Id=query_id)
            if status["Status"] in ["FINISHED", "FAILED", "ABORTED"]:
                break
        if status["Status"] != "FINISHED":
            return []
        result = redshift_client.get_statement_result(Id=query_id)
        columns = [col["name"] for col in result["ColumnMetadata"]]
        rows = [
            dict(zip(columns, [v.get("stringValue", "") for v in row]))
            for row in result["Records"]
        ]
        return rows
    except Exception:
        return []

def get_schema_comment() -> str:
    """
    Returns the comment for the given schema, or an empty string if none exists.
    """
    schema = REDSHIFT_SCHEMA
    sql = f"""
SELECT
    d.description AS schema_comment
FROM pg_catalog.pg_namespace n
LEFT JOIN pg_catalog.pg_description d
    ON n.oid = d.objoid
WHERE n.nspname = '{schema}'
"""
    rows = execute_redshift_query(sql)
    if rows and "schema_comment" in rows[0]:
        return rows[0]["schema_comment"]
    return ""
    
def get_native_tables() -> List[Dict[str, str]]:
    """
    Returns a string with one line for each database table: table_name -- table_comment
    Only includes tables with a non-null comment.
    """
    schema = REDSHIFT_SCHEMA
    sql = f"""
SELECT
    CONCAT('{schema}.', c.relname) AS table_name,
    obj_description(c.oid) AS table_comment
FROM pg_namespace n
JOIN pg_class c ON c.relnamespace = n.oid
WHERE c.relkind = 'r'
    AND n.nspname = '{schema}'
    AND (table_comment IS NULL OR table_comment != 'hidden');
"""
    return execute_redshift_query(sql)
    
def get_native_columns() -> List[Dict[str, str]]:
    """
    Fetch table and column info from Redshift and return as a list of dicts.
    Each dict contains table_name, column_name, and data_type.
    """
    schema = REDSHIFT_SCHEMA
    sql = f"""
SELECT
    CONCAT('{schema}.', c.table_name) as table_name,
    c.column_name,
    c.data_type,
    d.description AS column_comment
FROM information_schema.columns c
JOIN pg_catalog.pg_class cls
    ON cls.relname = c.table_name
JOIN pg_catalog.pg_namespace ns
    ON ns.nspname = c.table_schema
    AND ns.oid = cls.relnamespace
LEFT JOIN pg_catalog.pg_description d
    ON d.objoid = cls.oid
    AND d.objsubid = c.ordinal_position
WHERE c.table_schema = '{schema}'
    AND column_comment IS NULL or column_comment != 'hidden'
ORDER BY c.table_name, c.ordinal_position;
"""
    return execute_redshift_query(sql)

def filter_columns(columns: List[Dict[str, str]], table_names: List[str]) -> List[Dict[str, str]]:
    """
    Filters the columns list to only rows where table_name is in table_names.
    Returns a list of dicts for those tables.
    """
    return [row for row in columns if row.get("table_name") in table_names]

def query_database(sql: str) -> str:
    """
    Run a SQL query against AWS Redshift Serverless using the Data API and return results as a string.
    """
    try:
        res = redshift_client.execute_statement(
            WorkgroupName=REDSHIFT_WORKGROUP_NAME,
            Database=REDSHIFT_DATABASE,
            Sql=sql
        )
        query_id = res["Id"]
        while True:
            status = redshift_client.describe_statement(Id=query_id)
            if status["Status"] in ["FINISHED", "FAILED", "ABORTED"]:
                break
        if status["Status"] != "FINISHED":
            return f"Query failed: {status.get('Error', 'Unknown error')}"
        result = redshift_client.get_statement_result(Id=query_id)
        columns = [col["name"] for col in result["ColumnMetadata"]]
        def extract_value(v):
            # Redshift Data API returns one of these keys depending on type
            for key in ("stringValue", "longValue", "doubleValue", "booleanValue", "arrayValue"): 
                if key in v:
                    return v[key]
            return ""
        rows = [
            dict(zip(columns, [extract_value(v) for v in row]))
            for row in result["Records"]
        ]
        return str(rows)
    except Exception as e:
        return f"Error running query: {str(e)}"

def get_external_tables() -> List[Dict[str, str]]:
    """
    Returns a list of dicts with table names and their description from AWS Glue Data Catalog for the given database.
    Each dict contains 'table_name' and 'table_description'. If no description exists, the list will be empty.
    """
    database_name = REDSHIFT_AWSCATALOG_DATABASE
    region_name = AWS_REGION
    if not database_name or database_name == '':
        return []
    glue = boto3.client('glue', region_name=region_name)
    tables = []
    paginator = glue.get_paginator('get_tables')
    for page in paginator.paginate(DatabaseName=database_name):
        tables.extend(page['TableList'])
    result = []
    for table in tables:
        name = table.get('Name', '')
        desc = table.get('Description', '')
        result.append({'table_name': 'awsdatacatalog."' + database_name + '".' + name, 'table_comment': desc})
    return result

def get_external_columns() -> List[Dict[str, str]]:
    """
    Returns a list of dicts with all columns of AWS Glue Data Catalog for the given database.
    Each dict contains: table_name, column_name, data_type, column_comment.
    """
    database_name = REDSHIFT_AWSCATALOG_DATABASE
    region_name = AWS_REGION 
    if not database_name or database_name == '':
        return []   
    glue = boto3.client('glue', region_name=region_name)
    columns = []
    paginator = glue.get_paginator('get_tables')
    for page in paginator.paginate(DatabaseName=database_name):
        for table in page['TableList']:
            table_name = table.get('Name', '')
            for col in table.get('StorageDescriptor', {}).get('Columns', []):
                col_name = col.get('Name', '')
                data_type = col.get('Type', '')
                col_comment = col.get('Comment', '')
                columns.append({
                    'table_name': table_name,
                    'column_name': col_name,
                    'data_type': data_type,
                    'column_comment': col_comment
                })
    return columns
