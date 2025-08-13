import os
import boto3
from typing import List, Dict

REDSHIFT_SCHEMA = os.getenv("REDSHIFT_SCHEMA", "dg1")
AWS_REGION = os.getenv("AWS_REGION")
REDSHIFT_WORKGROUP_NAME = os.getenv("REDSHIFT_WORKGROUP_NAME")
REDSHIFT_DATABASE = os.getenv("REDSHIFT_DATABASE")

redshift_client = boto3.client("redshift-data", region_name=AWS_REGION)

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
    AND obj_description(c.oid) is not null
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
    
def get_redshift_schema_info(schema: str) -> str:
    """
    Fetch table and column info from Redshift and format as a string for prompt context.
    """
    sql = f"""
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = '{schema}'
    ORDER BY table_name, ordinal_position
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
            return "(Could not fetch schema info)"
        result = redshift_client.get_statement_result(Id=query_id)
        columns = [col["name"] for col in result["ColumnMetadata"]]
        rows = [
            dict(zip(columns, [v.get("stringValue", "") for v in row]))
            for row in result["Records"]
        ]
        table_map = {}
        for row in rows:
            t = row["table_name"]
            c = row["column_name"]
            d = row["data_type"]
            table_map.setdefault(t, []).append(f"{c} ({d})")
        lines = []
        for t, cols in table_map.items():
            lines.append(f"Table: {t}\n  " + ", ".join(cols))
        return "\n".join(lines)
    except Exception as e:
        return f"(Error fetching schema info: {str(e)})"

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
        rows = [
            dict(zip(columns, [v.get("stringValue", "") for v in row]))
            for row in result["Records"]
        ]
        return str(rows)
    except Exception as e:
        return f"Error running query: {str(e)}"
