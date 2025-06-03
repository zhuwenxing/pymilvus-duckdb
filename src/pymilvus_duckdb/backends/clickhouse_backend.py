import json
import clickhouse_connect
import pandas as pd
import numpy as np
from typing import List, Union
from pymilvus import DataType

from .base_backend import BaseBackend
from ..logger_config import logger


class ClickHouseBackend(BaseBackend):
    """ClickHouse backend implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ch_client = None
        
    def connect(self, **kwargs) -> None:
        """Establish connection to ClickHouse."""
        clickhouse_host = kwargs.get("clickhouse_host", "localhost")
        clickhouse_port = kwargs.get("clickhouse_port", 8123)
        clickhouse_username = kwargs.get("clickhouse_username", "default")
        clickhouse_password = kwargs.get("clickhouse_password", "")
        clickhouse_database = kwargs.get("clickhouse_database", "default")
        
        logger.info(f"Connecting to ClickHouse at: '{clickhouse_host}:{clickhouse_port}'")
        
        try:
            self.ch_client = clickhouse_connect.get_client(
                host=clickhouse_host,
                port=clickhouse_port,
                username=clickhouse_username,
                password=clickhouse_password,
                database=clickhouse_database
            )
            # Test connection
            self.ch_client.command("SELECT 1")
            logger.info("ClickHouse connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise
    
    def close(self) -> None:
        """Close ClickHouse connection."""
        try:
            if self.ch_client:
                self.ch_client.close()
        except Exception as e:
            logger.error(f"Failed to close ClickHouse connection: {e}")
    
    def create_table(self, table_name: str, schema_fields: List[str]) -> None:
        """Create a ClickHouse table."""
        fields_sql = ", ".join(schema_fields)
        # ClickHouse uses MergeTree engine and requires ORDER BY clause
        create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({fields_sql}) ENGINE = MergeTree() ORDER BY `{self.primary_field}`"
        logger.debug(f"Create table SQL: {create_sql}")
        self.ch_client.command(create_sql)
    
    def drop_table(self, table_name: str) -> None:
        """Drop a ClickHouse table."""
        self.ch_client.command(f"DROP TABLE IF EXISTS `{table_name}`")
    
    def insert_data(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data into ClickHouse table."""
        # Process JSON fields
        for field in self.json_fields:
            if field in data.columns:
                data[field] = data[field].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
        
        self.ch_client.insert_df(f"`{table_name}`", data)
    
    def delete_data(self, table_name: str, primary_key_values: List[Union[int, str]]) -> pd.DataFrame:
        """Delete data from ClickHouse table and return deleted data for rollback."""
        # First, backup the data that will be deleted
        pk_list = "', '".join(str(pk) for pk in primary_key_values)
        backup_query = f"SELECT * FROM `{table_name}` WHERE `{self.primary_field}` IN ('{pk_list}')"
        backup_data = self.ch_client.query_df(backup_query)
        
        # Delete from ClickHouse using ALTER TABLE DELETE
        delete_sql = f"ALTER TABLE `{table_name}` DELETE WHERE `{self.primary_field}` IN ('{pk_list}')"
        self.ch_client.command(delete_sql)
        return backup_data
    
    def upsert_data(self, table_name: str, data: pd.DataFrame, primary_key: str) -> pd.DataFrame:
        """Upsert data into ClickHouse table and return original data for rollback."""
        # Process JSON fields
        for field in self.json_fields:
            if field in data.columns:
                data[field] = data[field].apply(lambda x: json.dumps(x) if not isinstance(x, str) else x)
        
        # Backup existing data for rollback
        pks = data[primary_key].tolist()
        pk_list = "', '".join(str(pk) for pk in pks)
        backup_query = f"SELECT * FROM `{table_name}` WHERE `{primary_key}` IN ('{pk_list}')"
        try:
            backup_data = self.ch_client.query_df(backup_query)
        except:
            backup_data = pd.DataFrame()
        
        # ClickHouse doesn't have native UPSERT, so we delete then insert
        if not backup_data.empty:
            delete_sql = f"ALTER TABLE `{table_name}` DELETE WHERE `{primary_key}` IN ('{pk_list}')"
            self.ch_client.command(delete_sql)
        
        # Insert new data
        self.ch_client.insert_df(f"`{table_name}`", data)
        return backup_data
    
    def query_data(self, table_name: str, filter_expr: str, output_fields: List[str]) -> pd.DataFrame:
        """Query data from ClickHouse table."""
        fields_str = ", ".join([f"`{field}`" if field != "*" else "*" for field in output_fields])
        sql_filter = self.convert_milvus_filter_to_sql(filter_expr)
        query_sql = f"SELECT {fields_str} FROM `{table_name}` WHERE {sql_filter}"
        return self.ch_client.query_df(query_sql)
    
    def count_records(self, table_name: str) -> int:
        """Count records in ClickHouse table."""
        try:
            table_exists_query = f"EXISTS TABLE `{table_name}`"
            table_exists = self.ch_client.command(table_exists_query)
            
            if not table_exists:
                logger.error(f"ClickHouse table '{table_name}' does not exist.")
                return 0
            
            count_result = self.ch_client.query(f"SELECT COUNT(*) FROM `{table_name}`")
            if count_result is None or not count_result.result_rows:
                logger.error(f"ClickHouse count query returned None for table '{table_name}'.")
                return 0
            
            return count_result.result_rows[0][0]
        except Exception as e:
            logger.error(f"Failed to count records in ClickHouse table '{table_name}': {e}")
            return 0
    
    def export_data(self, table_name: str) -> pd.DataFrame:
        """Export all data from ClickHouse table."""
        return self.ch_client.query_df(f"SELECT * FROM `{table_name}`")
    
    def table_exists(self, table_name: str) -> bool:
        """Check if ClickHouse table exists."""
        try:
            table_exists_query = f"EXISTS TABLE `{table_name}`"
            return self.ch_client.command(table_exists_query)
        except Exception:
            return False
    
    def sample_data(self, table_name: str, num_samples: int) -> pd.DataFrame:
        """Sample data from ClickHouse table."""
        # ClickHouse: use ORDER BY rand() LIMIT as alternative to SAMPLE
        query = f"SELECT * FROM `{table_name}` ORDER BY rand() LIMIT {num_samples}"
        return self.ch_client.query_df(query)
    
    def convert_milvus_filter_to_sql(self, filter_expr: str) -> str:
        """Convert Milvus filter expression to ClickHouse SQL."""
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
            return f"`{field}` IN ({sql_list})"

        expr = re.sub(r"(\w+)\s+in\s+(\[[^\]]*\])", in_repl, expr, flags=re.IGNORECASE)

        # 4. LIKE: keep as is, but ensure string is single quoted
        expr = re.sub(r'LIKE\s+"([^"]*)"', lambda m: f"LIKE '{m.group(1)}'", expr)

        # 5. IS NULL / IS NOT NULL
        expr = re.sub(r"is\s+null", "IS NULL", expr, flags=re.IGNORECASE)
        expr = re.sub(r"is\s+not\s+null", "IS NOT NULL", expr, flags=re.IGNORECASE)

        # 6. JSON key access: field["key"] -> JSONExtractString(field, 'key')
        expr = re.sub(r"(\w+)\[\"([\w_]+)\"\]", r"JSONExtractString(`\1`, '\2')", expr)

        # 7. String constants: "abc" -> 'abc'
        expr = re.sub(r'"([^"]*)"', lambda m: f"'{m.group(1)}'", expr)

        # 8. Add backticks around field names that don't have them
        expr = re.sub(r"\\b(\w+)\\b(?=\s*[=!<>])", r"`\1`", expr)

        # 9. Remove redundant spaces
        expr = re.sub(r"\s+", " ", expr).strip()

        return expr
    
    def get_backend_type(self) -> str:
        """Return backend type."""
        return "clickhouse"
    
    def rollback_insert(self, table_name: str, primary_key_values: List[Union[int, str]]) -> None:
        """Rollback insert operation by deleting the inserted data."""
        try:
            pk_list = "', '".join(str(pk) for pk in primary_key_values)
            delete_sql = f"ALTER TABLE `{table_name}` DELETE WHERE `{self.primary_field}` IN ('{pk_list}')"
            self.ch_client.command(delete_sql)
        except Exception as e:
            logger.error(f"Failed to rollback insert: {e}")
    
    def rollback_delete(self, table_name: str, backup_data: pd.DataFrame) -> None:
        """Rollback delete operation by re-inserting the data."""
        if backup_data is not None and not backup_data.empty:
            try:
                self.ch_client.insert_df(f"`{table_name}`", backup_data)
            except Exception as e:
                logger.error(f"Failed to rollback delete: {e}")
    
    def rollback_upsert(self, table_name: str, primary_key_values: List[Union[int, str]], backup_data: pd.DataFrame) -> None:
        """Rollback upsert operation by restoring original data."""
        try:
            pk_list = "', '".join(str(pk) for pk in primary_key_values)
            
            # Delete the new data first
            delete_sql = f"ALTER TABLE `{table_name}` DELETE WHERE `{self.primary_field}` IN ('{pk_list}')"
            self.ch_client.command(delete_sql)
            
            # Restore backup data if it existed
            if backup_data is not None and not backup_data.empty:
                self.ch_client.insert_df(f"`{table_name}`", backup_data)
        except Exception as e:
            logger.error(f"Failed to rollback upsert: {e}")
    
    def milvus_dtype_to_backend_dtype(self, milvus_type: DataType) -> str:
        """Convert Milvus data type to ClickHouse data type."""
        type_mapping = {
            DataType.BOOL: "Bool",
            DataType.INT8: "Int8",
            DataType.INT16: "Int16", 
            DataType.INT32: "Int32",
            DataType.INT64: "Int64",
            DataType.FLOAT: "Float32",
            DataType.DOUBLE: "Float64",
            DataType.VARCHAR: "String",
            DataType.ARRAY: "Array(Int64)",
            DataType.JSON: "String",  # ClickHouse can store JSON as String
            DataType.FLOAT_VECTOR: "Array(Float32)",
        }
        
        if milvus_type in type_mapping:
            return type_mapping[milvus_type]
        else:
            raise ValueError(f"Unsupported Milvus data type: {milvus_type}") 