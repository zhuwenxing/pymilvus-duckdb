from pymilvus import MilvusClient, CollectionSchema
from pymilvus import DataType
import duckdb
import pandas as pd
import json
from pathlib import Path
from deepdiff import DeepDiff
from typing import List, Dict
from .logger_config import logger

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
        logger.info(f"Initializing MilvusDuckDBClient with Milvus URI: '{uri}', DuckDB path: '{duckdb_path}'")
        Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)
        self.duck_conn = duckdb.connect(duckdb_path)
        self.fields = []
        self.primary_field = ""
        self.fields_name_list = []
        self.json_fields = []
        self.array_fields = []
        self.float_vector_fields = []
    
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
            if field.dtype == DataType.ARRAY:
                self.array_fields.append(name)
            if field.dtype == DataType.FLOAT_VECTOR:
                self.float_vector_fields.append(name)
        fields_sql = ", ".join(self.fields)
        create_sql = f"CREATE TABLE IF NOT EXISTS {collection_name} ({fields_sql});"
        logger.debug(f"Create table SQL: {create_sql}")
        # Step 2: Start DuckDB transaction
        try:
            self.duck_conn.execute("BEGIN TRANSACTION;")
            # Step 3: Create table in DuckDB
            self.duck_conn.execute(create_sql)
            try:
                # Step 4: Create collection in Milvus
                result = super().create_collection(collection_name, schema=schema, consistency_level="Strong", **kwargs)
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
        # only for duckdb, milvus does not support export
        return self.duck_conn.execute(f"SELECT * FROM {collection_name}").fetchdf()

    def count(self, collection_name: str):
        milvus_count = super().query(collection_name, filter="", output_fields=["count(*)"])
        duckdb_count = self.duck_conn.execute(f"SELECT COUNT(*) FROM {collection_name}").fetchone()
        res = {
            "milvus_count": milvus_count[0]["count(*)"],
            "duckdb_count": duckdb_count[0]
        }
        return res

    def compare(self, collection_name: str):
        count_res = self.count(collection_name)
        if count_res["milvus_count"] != count_res["duckdb_count"]:
            return False
        duckdb_pks_df = self.duck_conn.execute(f"SELECT {self.primary_field} FROM {collection_name}").fetchdf()
        logger.debug(f"DuckDB PKs List:\n{duckdb_pks_df}")
        for i in range(len(duckdb_pks_df)):
            pk = duckdb_pks_df[self.primary_field].iloc[i]
            milvus_data = super().query(collection_name, filter=f"{self.primary_field} == {pk}", output_fields=["*"])
            milvus_data_df = pd.DataFrame(milvus_data)
            current_row_duckdb_df = self.duck_conn.execute(f"SELECT * FROM {collection_name} WHERE {self.primary_field} = {pk}").fetchdf()
            
            for field in self.json_fields:
                # milvus_data_df[field] = milvus_data_df[field].apply(lambda x: json.loads(x))
                current_row_duckdb_df[field] = current_row_duckdb_df[field].apply(lambda x: json.loads(x))
            for field in self.array_fields:
                milvus_data_df[field] = milvus_data_df[field].apply(lambda x: list(x))
                current_row_duckdb_df[field] = current_row_duckdb_df[field].apply(lambda x: list(x))
            for field in self.float_vector_fields:
                milvus_data_df[field] = milvus_data_df[field].apply(lambda x: list(x))
                current_row_duckdb_df[field] = current_row_duckdb_df[field].apply(lambda x: list(x))
            milvus_data_dict = milvus_data_df.to_dict(orient="records")[0]
            current_row_duckdb_dict = current_row_duckdb_df.to_dict(orient="records")[0]
            logger.debug(f"Milvus data for pk={pk}:\n{milvus_data_dict}")

            logger.debug(f"DuckDB data for pk={pk}:\n{current_row_duckdb_dict}")
            # compare
            diff = DeepDiff(milvus_data_dict, current_row_duckdb_dict, ignore_order=True)
            if diff:
                logger.error(f"diff: {diff}")
                logger.error(f"Data for pk={pk} does not match between Milvus and DuckDB\nMilvus: {milvus_data_dict}\nDuckDB: {current_row_duckdb_dict}")
                return False


        return True
    
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

            
