import os
import pyodbc
from typing import List, Dict

# Azure SQL connection settings from environment variables
AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER")
AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE")
AZURE_SQL_USERNAME = os.getenv("AZURE_SQL_USERNAME")
AZURE_SQL_PASSWORD = os.getenv("AZURE_SQL_PASSWORD")
AZURE_SQL_DRIVER = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")
DB_SCHEMA = os.getenv("AZURE_SQL_SCHEMA", 'dbo')
DB_PLATFORM = "Azure SQL Server"
DB_SPECIFICS = "Never use LIMIT — use TOP (n)."

def get_connection():
	conn_str = (
		f"DRIVER={{{AZURE_SQL_DRIVER}}};"
		f"SERVER={AZURE_SQL_SERVER};"
		f"DATABASE={AZURE_SQL_DATABASE};"
		f"UID={AZURE_SQL_USERNAME};"
		f"PWD={AZURE_SQL_PASSWORD}"
	)
	return pyodbc.connect(conn_str)

def get_schema_comment(schema: str) -> str:
	"""
	Returns the comment for the given schema, or an empty string if none exists.
	"""
	sql = """
	SELECT CAST(ep.value AS NVARCHAR(MAX)) AS schema_comment
	FROM sys.schemas s
	LEFT JOIN sys.extended_properties ep
		ON ep.major_id = s.schema_id AND ep.minor_id = 0 AND ep.name = 'MS_Description'
	WHERE s.name = ?
	"""
	try:
		with get_connection() as conn:
			cursor = conn.cursor()
			cursor.execute(sql, (schema,))
			row = cursor.fetchone()
			return row[0] if row and row[0] else ""
	except Exception as e:
		print("Exception occurred:", e)
		return ""

def get_tables(schema: str) -> List[Dict[str, str]]:
    """
    Returns a list of tables and their comments in the given schema.
    """
    sql = """
    SELECT t.name AS table_name,
           CAST(ep.value AS NVARCHAR(MAX)) AS table_comment
    FROM sys.tables t
    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
    LEFT JOIN sys.extended_properties ep
        ON ep.major_id = t.object_id AND ep.minor_id = 0 AND ep.name = 'MS_Description'
    WHERE s.name = ?
    ORDER BY t.name;
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (schema,))
            rows = cursor.fetchall()
            return [
                {"table_name": row[0], "table_comment": row[1] if row[1] else ""}
                for row in rows
            ]
    except Exception as e:
        print("Exception occurred:", e)
        return []

def get_columns(schema: str) -> List[Dict[str, str]]:
	"""
	Returns a list of columns and their comments for all tables in the given schema.
	"""
	sql = """
	SELECT t.name AS table_name,
		   c.name AS column_name,
		   ty.name AS data_type,
		   CAST(ep.value AS NVARCHAR(MAX)) AS column_comment
	FROM sys.tables t
	INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
	INNER JOIN sys.columns c ON t.object_id = c.object_id
	INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id
	LEFT JOIN sys.extended_properties ep
		ON ep.major_id = t.object_id AND ep.minor_id = c.column_id AND ep.name = 'MS_Description'
	WHERE s.name = ?
	ORDER BY t.name, c.column_id;
	"""
	try:
		with get_connection() as conn:
			cursor = conn.cursor()
			cursor.execute(sql, (schema,))
			rows = cursor.fetchall()
			return [
				{
					"table_name": row[0],
					"column_name": row[1],
					"data_type": row[2],
					"column_comment": row[3] if row[3] else ""
				}
				for row in rows
			]
	except Exception as e:
		print("Exception occurred:", e)
		return []

def filter_columns(columns: List[Dict[str, str]], table_names: List[str]) -> List[Dict[str, str]]:
	"""
	Filters the columns list to only rows where table_name is in table_names.
	Returns a list of dicts for those tables.
	"""
	return [row for row in columns if row.get("table_name") in table_names]

def query_database(sql: str, params=None) -> str:
	"""
	Run a SQL query against Azure SQL Server and return results as a string.
	"""
	try:
		with get_connection() as conn:
			cursor = conn.cursor()
			cursor.execute(sql, params or ())
			columns = [column[0] for column in cursor.description]
			rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
			return str(rows)
	except Exception as e:
		print("Exception occurred:", e)
		return f"Error running query: {str(e)}"
