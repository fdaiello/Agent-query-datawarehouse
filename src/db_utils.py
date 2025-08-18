import os
import boto3
from typing import List, Dict

REDSHIFT_SCHEMA = os.getenv("REDSHIFT_SCHEMA")
AWS_REGION = os.getenv("AWS_REGION")
REDSHIFT_WORKGROUP_NAME = os.getenv("REDSHIFT_WORKGROUP_NAME")
REDSHIFT_DATABASE = os.getenv("REDSHIFT_DATABASE")

redshift_client = boto3.client("redshift-data", region_name=AWS_REGION)

def get_schema_comment(schema: str) -> str:
    """
    Returns the comment for the given schema, or an empty string if none exists.
    """
    sql = f"""
    SELECT
        d.description AS schema_comment
    FROM pg_catalog.pg_namespace n
    LEFT JOIN pg_catalog.pg_description d
        ON n.oid = d.objoid
    WHERE n.nspname = '{schema}'
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
            return ""
        result = redshift_client.get_statement_result(Id=query_id)
        if result["Records"]:
            return result["Records"][0][0].get("stringValue", "")
        return ""
    except Exception:
        return ""
    
def get_tables(schema: str) -> List[Dict[str, str]]:
    """
    Returns a string with one line for each database table: table_name -- table_comment
    Only includes tables with a non-null comment.
    """
    sql = f"""
    SELECT
        c.relname AS table_name,
        obj_description(c.oid) AS table_comment
    FROM pg_namespace n
    JOIN pg_class c ON c.relnamespace = n.oid
    WHERE c.relkind = 'r'
    AND n.nspname = '{schema}'
    AND (table_comment IS NULL OR table_comment != 'hidden');
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
    except Exception as e:
        return []
    
def get_columns(schema: str) -> List[Dict[str, str]]:
    """
    Fetch table and column info from Redshift and return as a list of dicts.
    Each dict contains table_name, column_name, and data_type.
    """
    sql = f"""
        SELECT
            c.table_name,
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
        ORDER BY c.table_name, c.ordinal_position;
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

def filter_columns(columns: List[Dict[str, str]], table_names: List[str]) -> List[Dict[str, str]]:
    """
    Filters the columns list to only rows where table_name is in table_names.
    Returns a list of dicts for those tables.
    """
    return [row for row in columns if row.get("table_name") in table_names]

def query_redshift(sql: str) -> str:
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
