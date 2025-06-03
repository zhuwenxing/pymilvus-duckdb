import json
import os
import uuid
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from typing import List, Union
from pymilvus import DataType

from .base_backend import BaseBackend
from ..logger_config import logger

BASE_DIR = "./data/duckdb"


class DuckDBBackend(BaseBackend):
    """DuckDB backend implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.duck_conn = None
        
    def connect(self, **kwargs) -> None:
        """Establish connection to DuckDB."""
        duckdb_dir = kwargs.get("duckdb_dir", BASE_DIR)
        host = kwargs.get("host", "localhost")
        duckdb_path = f"{duckdb_dir}/{host}/{uuid.uuid4()}.db"
        
        logger.info(f"Connecting to DuckDB at: '{duckdb_path}'")
        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Handle WAL file corruption
        wal_path = duckdb_path + ".wal"
        if os.path.exists(wal_path):
            try:
                test_conn = duckdb.connect(duckdb_path, read_only=True)
                test_conn.close()
            except Exception as e:
                logger.warning(f"WAL file corrupted, deleting: {wal_path}, error: {e}")
                os.remove(wal_path)
        
        self.duck_conn = duckdb.connect(duckdb_path)
        logger.info("DuckDB connection established successfully")
    
    def close(self) -> None:
        """Close DuckDB connection."""
        try:
            if self.duck_conn:
                self.duck_conn.close()
        except Exception as e:
            logger.error(f"Failed to close DuckDB connection: {e}")
    
    def create_table(self, table_name: str, schema_fields: List[str]) -> None:
        """Create a DuckDB table."""
        fields_sql = ", ".join(schema_fields)
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({fields_sql});"
        logger.debug(f"Create table SQL: {create_sql}")
        
        self.duck_conn.execute("BEGIN TRANSACTION;")
        try:
            self.duck_conn.execute(create_sql)
            self.duck_conn.execute("COMMIT;")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise e
    
    def drop_table(self, table_name: str) -> None:
        """Drop a DuckDB table."""
        self.duck_conn.execute("BEGIN TRANSACTION;")
        try:
            self.duck_conn.execute(f"DROP TABLE IF EXISTS {table_name};")
            self.duck_conn.execute("COMMIT;")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise e
    
    def insert_data(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data into DuckDB table."""
        # Process JSON fields
        for field in self.json_fields:
            if field in data.columns:
                data[field] = data[field].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
        
        self.duck_conn.register("df", data)
        self.duck_conn.execute("BEGIN TRANSACTION;")
        try:
            self.duck_conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            self.duck_conn.execute("COMMIT;")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise e
    
    def delete_data(self, table_name: str, primary_key_values: List[Union[int, str]]) -> pd.DataFrame:
        """Delete data from DuckDB table and return deleted data for rollback."""
        # First, backup the data that will be deleted
        pk_list = ','.join(str(pk) for pk in primary_key_values)
        backup_query = f"SELECT * FROM {table_name} WHERE {self.primary_field} IN ({pk_list})"
        backup_data = self.duck_conn.execute(backup_query).fetchdf()
        
        self.duck_conn.execute("BEGIN TRANSACTION;")
        try:
            delete_sql = f"DELETE FROM {table_name} WHERE {self.primary_field} IN ({pk_list})"
            self.duck_conn.execute(delete_sql)
            self.duck_conn.execute("COMMIT;")
            return backup_data
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise e
    
    def upsert_data(self, table_name: str, data: pd.DataFrame, primary_key: str) -> pd.DataFrame:
        """Upsert data into DuckDB table and return original data for rollback."""
        # Process JSON fields
        for field in self.json_fields:
            if field in data.columns:
                data[field] = data[field].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
        
        # Backup existing data for rollback
        pks = data[primary_key].tolist()
        pk_list = ','.join(str(pk) for pk in pks)
        backup_query = f"SELECT * FROM {table_name} WHERE {primary_key} IN ({pk_list})"
        try:
            backup_data = self.duck_conn.execute(backup_query).fetchdf()
        except:
            backup_data = pd.DataFrame()
        
        self.duck_conn.register("df", data)
        self.duck_conn.execute("BEGIN TRANSACTION;")
        try:
            self.duck_conn.execute(f"""INSERT INTO {table_name}
                                    SELECT * FROM df
                                    ON CONFLICT ({primary_key}) DO UPDATE SET
                                    {", ".join([f"{col} = EXCLUDED.{col}" for col in self.fields_name_list])}""")
            self.duck_conn.execute("COMMIT;")
            return backup_data
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise e
    
    def query_data(self, table_name: str, filter_expr: str, output_fields: List[str]) -> pd.DataFrame:
        """Query data from DuckDB table."""
        fields_str = ', '.join(output_fields)
        sql_filter = self.convert_milvus_filter_to_sql(filter_expr)
        query_sql = f"SELECT {fields_str} FROM {table_name} WHERE {sql_filter}"
        return self.duck_conn.execute(query_sql).fetchdf()
    
    def count_records(self, table_name: str) -> int:
        """Count records in DuckDB table."""
        try:
            # Check if table exists
            table_exists = self.duck_conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()[0] > 0
            
            if not table_exists:
                logger.error(f"DuckDB table '{table_name}' does not exist.")
                return 0
            
            count_result = self.duck_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return count_result[0] if count_result else 0
        except Exception as e:
            logger.error(f"Failed to count records in DuckDB table '{table_name}': {e}")
            return 0
    
    def export_data(self, table_name: str) -> pd.DataFrame:
        """Export all data from DuckDB table."""
        return self.duck_conn.execute(f"SELECT * FROM {table_name}").fetchdf()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if DuckDB table exists."""
        try:
            result = self.duck_conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()
            return result[0] > 0
        except Exception:
            return False
    
    def sample_data(self, table_name: str, num_samples: int) -> pd.DataFrame:
        """Sample data from DuckDB table."""
        query = f"SELECT * FROM {table_name} USING SAMPLE {num_samples} ROWS"
        return self.duck_conn.execute(query).fetchdf()
    
    def convert_milvus_filter_to_sql(self, filter_expr: str) -> str:
        """Convert Milvus filter expression to DuckDB SQL."""
        import re

        if not filter_expr or filter_expr.strip() == "":
            return "1=1"

        expr = filter_expr

        # 1. Replace logical operators to uppercase
        expr = re.sub(r"\\b(and)\\b", "AND", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\\b(or)\\b", "OR", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\\b(not)\\b", "NOT", expr, flags=re.IGNORECASE)

        # 2. Replace comparison operators == -> =
        expr = re.sub(r"(?<![!<>])==", "=", expr)

        # 3. Replace IN: field in [..] -> field IN (..)
        def in_repl(match):
            field = match.group(1)
            values = match.group(2)
            pylist = eval(values)
            sql_list = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in pylist])
            return f"{field} IN ({sql_list})"

        expr = re.sub(r"(\w+)\s+in\s+(\[[^\]]*\])", in_repl, expr, flags=re.IGNORECASE)

        # 4. LIKE: keep as is, but ensure string is single quoted
        expr = re.sub(r'LIKE\s+"([^"]*)"', lambda m: f"LIKE '{m.group(1)}'", expr)

        # 5. IS NULL / IS NOT NULL
        expr = re.sub(r"is\s+null", "IS NULL", expr, flags=re.IGNORECASE)
        expr = re.sub(r"is\s+not\s+null", "IS NOT NULL", expr, flags=re.IGNORECASE)

        # 6. JSON key access: field["key"] -> field->>'key'
        expr = re.sub(r"(\w+)\[\"([\w_]+)\"\]", r"\1->>'\2'", expr)

        # 7. String constants: "abc" -> 'abc'
        expr = re.sub(r'"([^"]*)"', lambda m: f"'{m.group(1)}'", expr)

        # 8. Remove redundant spaces
        expr = re.sub(r"\s+", " ", expr).strip()

        return expr
    
    def get_backend_type(self) -> str:
        """Return backend type."""
        return "duckdb"
    
    def rollback_insert(self, table_name: str, primary_key_values: List[Union[int, str]]) -> None:
        """Rollback insert operation by deleting the inserted data."""
        try:
            pk_list = ','.join(str(pk) for pk in primary_key_values)
            delete_sql = f"DELETE FROM {table_name} WHERE {self.primary_field} IN ({pk_list})"
            self.duck_conn.execute("BEGIN TRANSACTION;")
            self.duck_conn.execute(delete_sql)
            self.duck_conn.execute("COMMIT;")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            logger.error(f"Failed to rollback insert: {e}")
    
    def rollback_delete(self, table_name: str, backup_data: pd.DataFrame) -> None:
        """Rollback delete operation by re-inserting the data."""
        if backup_data is not None and not backup_data.empty:
            try:
                self.duck_conn.register("backup_df", backup_data)
                self.duck_conn.execute("BEGIN TRANSACTION;")
                self.duck_conn.execute(f"INSERT INTO {table_name} SELECT * FROM backup_df")
                self.duck_conn.execute("COMMIT;")
            except Exception as e:
                self.duck_conn.execute("ROLLBACK;")
                logger.error(f"Failed to rollback delete: {e}")
    
    def rollback_upsert(self, table_name: str, primary_key_values: List[Union[int, str]], backup_data: pd.DataFrame) -> None:
        """Rollback upsert operation by restoring original data."""
        try:
            pk_list = ','.join(str(pk) for pk in primary_key_values)
            self.duck_conn.execute("BEGIN TRANSACTION;")
            
            # Delete the new data
            self.duck_conn.execute(f"DELETE FROM {table_name} WHERE {self.primary_field} IN ({pk_list})")
            
            # Restore backup data if it existed
            if backup_data is not None and not backup_data.empty:
                self.duck_conn.register("backup_df", backup_data)
                self.duck_conn.execute(f"INSERT INTO {table_name} SELECT * FROM backup_df")
            
            self.duck_conn.execute("COMMIT;")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            logger.error(f"Failed to rollback upsert: {e}")
    
    def milvus_dtype_to_backend_dtype(self, milvus_type: DataType) -> str:
        """Convert Milvus data type to DuckDB data type."""
        type_mapping = {
            DataType.BOOL: "BOOLEAN",
            DataType.INT8: "INT8",
            DataType.INT16: "INT16",
            DataType.INT32: "INT32",
            DataType.INT64: "INT64",
            DataType.FLOAT: "FLOAT",
            DataType.DOUBLE: "DOUBLE",
            DataType.VARCHAR: "VARCHAR",
            DataType.ARRAY: "INT64[]",
            DataType.JSON: "JSON",
            DataType.FLOAT_VECTOR: "FLOAT[]",
        }
        
        if milvus_type in type_mapping:
            return type_mapping[milvus_type]
        else:
            raise ValueError(f"Unsupported Milvus data type: {milvus_type}")
    
    def start_transaction(self) -> None:
        """Start a transaction."""
        self.duck_conn.execute("BEGIN TRANSACTION;")
    
    def commit_transaction(self) -> None:
        """Commit a transaction."""
        self.duck_conn.execute("COMMIT;")
    
    def rollback_transaction(self) -> None:
        """Rollback a transaction."""
        self.duck_conn.execute("ROLLBACK;") 