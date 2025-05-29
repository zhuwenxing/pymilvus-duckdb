from pymilvus import MilvusClient, CollectionSchema
from pymilvus import DataType
import duckdb
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict

BASE_DIR = "./data/duckdb"

# forbid auto id
# milvus field schema to pyarrow schema then to duckdb schema

class MilvusDuckDBClient(MilvusClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uri = kwargs.get("uri", "")
        host = uri.split("://")[1].split(":")[0]
        duckdb_dir = kwargs.get("duckdb_dir", BASE_DIR)
        duckdb_path = f"{duckdb_dir}/{host}.db"
        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        self.duck_conn = duckdb.connect(duckdb_path)
        self.fields = []
        self.primary_field = ""
        self.fields_name_list = []
        self.json_fields = []
    
    def __del__(self):
        self.duck_conn.close()

    def create_collection(self, collection_name: str, schema: CollectionSchema=None, **kwargs):
        """
        Create a collection: first create table in DuckDB (in transaction), then create collection in Milvus.
        Rollback if any error happens to guarantee atomicity.
        """
        # Step 1: Prepare DuckDB table schema
        self.fields = []
        for field in schema.fields:
            if field.auto_id:
                raise ValueError("Auto ID is not supported for MilvusDuckDBClient")
            name = field.name
            duckdb_type = self._milvus_dtype_to_duckdb(field.dtype)
            if field.is_primary:
                self.primary_field = field.name
                duckdb_type = f"{duckdb_type} PRIMARY KEY"
            if field.dtype == DataType.FLOAT_VECTOR:
                dim = field.params.get("dim", 0)
                duckdb_type = f"FLOAT[{dim}]"
            if field.dtype == DataType.ARRAY:
                element_type = self._milvus_dtype_to_duckdb(field.element_type)
                duckdb_type = f"{element_type}[]"
            self.fields.append(f"{name} {duckdb_type}")
            self.fields_name_list.append(name)
            if field.dtype == DataType.JSON:
                self.json_fields.append(name)

        fields_sql = ", ".join(self.fields)
        create_sql = f"CREATE TABLE IF NOT EXISTS {collection_name} ({fields_sql});"
        print(create_sql)
        # Step 2: Start DuckDB transaction
        try:
            self.duck_conn.execute("BEGIN TRANSACTION;")
            # Step 3: Create table in DuckDB
            self.duck_conn.execute(create_sql)
            try:
                # Step 4: Create collection in Milvus
                result = super().create_collection(collection_name, schema=schema, **kwargs)
            except Exception as milvus_exc:
                # Step 5: If Milvus fails, rollback DuckDB and drop table if created
                self.duck_conn.execute(f"DROP TABLE IF EXISTS {collection_name};")
                self.duck_conn.execute("ROLLBACK;")
                raise milvus_exc
            # Step 6: Commit DuckDB transaction if all succeed
            self.duck_conn.execute("COMMIT;")
            return result
        except Exception as duckdb_exc:
            # Rollback if DuckDB fails at any step
            self.duck_conn.execute("ROLLBACK;")
            raise duckdb_exc


    def insert(self, collection_name: str, data: List[Dict], **kwargs):
        """
        Insert data with transaction logic: write to DuckDB first, if success then write to Milvus.
        If Milvus fails, rollback DuckDB transaction. Ensure data consistency.
        """
        df = pd.DataFrame(data)
        for field in self.json_fields:
            df[field] = df[field].apply(lambda x: json.dumps(x))
        self.duck_conn.register("df", df)
        try:
            self.duck_conn.execute("BEGIN TRANSACTION;")
            # Insert into DuckDB
            # data: [{"id": 1, "name": "test", "embedding": [0.1, 0.2, 0.3]}, ...]
            self.duck_conn.execute(f"INSERT INTO {collection_name} SELECT * FROM df")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            # DuckDB write failed, return failure
            raise RuntimeError(f"DuckDB insert failed: {e}")
        try:
            # Insert into Milvus
            result = super().insert(collection_name, data, **kwargs)
            self.duck_conn.execute("COMMIT;")
            return result
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            # Milvus write failed, rollback DuckDB
            raise RuntimeError(f"Milvus insert failed, DuckDB rolled back: {e}")

    def delete(self, collection_name: str, ids: List[int|str], **kwargs):
        """
        Delete data with transaction logic: write to DuckDB first, if success then write to Milvus.
        If Milvus fails, rollback DuckDB transaction. Ensure data consistency.
        """
        try:
            self.duck_conn.execute("BEGIN TRANSACTION;")
            # Parse ids and delete from DuckDB
            # ids: [1,2,3]
            delete_sql = f"DELETE FROM {collection_name} WHERE {self.primary_field} IN ({','.join(map(str, ids))})"
            self.duck_conn.execute(delete_sql)
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise RuntimeError(f"DuckDB delete failed: {e}")
        try:
            result = super().delete(collection_name, ids=ids, **kwargs)
            self.duck_conn.execute("COMMIT;")
            return result
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise RuntimeError(f"Milvus delete failed, DuckDB rolled back: {e}")

    def upsert(self, collection_name: str, data: List[Dict], **kwargs):
        """
        Upsert data with transaction logic: write to DuckDB first, if success then write to Milvus.
        If Milvus fails, rollback DuckDB transaction. Ensure data consistency.
        """
        df = pd.DataFrame(data)
        for field in self.json_fields:
            df[field] = df[field].apply(lambda x: json.dumps(x))
        self.duck_conn.register("df", df)
        try:
            self.duck_conn.execute("BEGIN TRANSACTION;")
            self.duck_conn.execute(f"""INSERT INTO {collection_name}
                                        SELECT * FROM df
                                        ON CONFLICT ({self.primary_field}) DO UPDATE SET
                                        {', '.join([f'{col} = EXCLUDED.{col}' for col in self.fields_name_list])}""")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise RuntimeError(f"DuckDB upsert failed: {e}")
        try:
            result = super().upsert(collection_name, data, **kwargs)
            self.duck_conn.execute("COMMIT;")
            return result
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise RuntimeError(f"Milvus upsert failed, DuckDB rolled back: {e}")

    def export(self, collection_name: str):
        return self.duck_conn.execute(f"SELECT * FROM {collection_name}").fetchdf()


    def _milvus_dtype_to_duckdb(self, milvus_type):
        MilvusDataTypeToDuckDBType= {
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
        if milvus_type in MilvusDataTypeToDuckDBType:
            return MilvusDataTypeToDuckDBType[milvus_type]
        else:
            raise ValueError(f"Unsupported data type: {milvus_type} when converting Milvus data type to DuckDB data type") 

            
