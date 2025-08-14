from typing import List, Dict

def format_schema_description(tables: List[Dict[str, str]], columns: List[Dict[str, str]]) -> str:
    """
    Returns a formatted schema description string for the given tables and columns.
    Format:
    Table: table_name
    Description: table_comments
    Columns:
    - column_name (data_type)
    ...
    """
    lines = []
    for table in tables:
        table_name = table.get("table_name", "")
        table_comment = table.get("table_comment", "")
        lines.append(f"Table: {table_name}")
        lines.append(f"Description: {table_comment}")
        lines.append("Columns:")
        # Find columns for this table
        table_columns = [col for col in columns if col.get("table_name") == table_name]
        for col in table_columns:
            col_name = col.get("column_name", "")
            col_comment = col.get("column_comment", "")
            line = f"- {col_name}"
            if col_comment and col_comment.strip():
                line += f" / {col_comment.strip()}"
            lines.append(line)
        lines.append("")  # Blank line between tables
    return "\n".join(lines)
