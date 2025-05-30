import json
import random
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from deepdiff import DeepDiff
from pymilvus import Collection, CollectionSchema, DataType, MilvusClient, connections

from .logger_config import logger

BASE_DIR = "./data/duckdb"

# Auto ID is not supported for MilvusDuckDBClient
# Milvus field schema is converted to DuckDB schema for data consistency


class MilvusDuckDBClient(MilvusClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uri = kwargs.get("uri", "")
        token = kwargs.get("token", "")
        connections.connect(uri=uri, token=token)
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
        self.varchar_fields = []
        self.float_vector_fields = []

    def __del__(self):
        self.duck_conn.close()

    def _get_schema(self, collection_name: str):
        c = Collection(collection_name)
        schema = c.schema
        for field in schema.fields:
            if field.is_primary:
                self.primary_field = field.name
            if field.dtype == DataType.FLOAT_VECTOR:
                self.float_vector_fields.append(field.name)
            if field.dtype == DataType.ARRAY:
                self.array_fields.append(field.name)
            if field.dtype == DataType.JSON:
                self.json_fields.append(field.name)
            if field.dtype == DataType.VARCHAR:
                self.varchar_fields.append(field.name)
        return schema

    def create_collection(self, collection_name: str, schema: CollectionSchema = None, **kwargs):
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
            if field.dtype == DataType.VARCHAR:
                self.varchar_fields.append(name)
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

    def drop_collection(self, collection_name: str):
        """
        Drop collection in both Milvus and DuckDB atomically. Use transaction to ensure consistency.
        """
        try:
            self.duck_conn.execute("BEGIN TRANSACTION;")  # Start DuckDB transaction
            try:
                self.duck_conn.execute(f"DROP TABLE IF EXISTS {collection_name};")
                super().drop_collection(collection_name)
                self.duck_conn.execute("COMMIT;")
            except Exception as milvus_exc:
                self.duck_conn.execute("ROLLBACK;")
                logger.error(f"Failed to drop collection '{collection_name}' in Milvus or DuckDB: {milvus_exc}")
                raise milvus_exc
        except Exception as duckdb_exc:
            logger.error(f"Failed to start DuckDB transaction or drop table: {duckdb_exc}")
            raise duckdb_exc

    def insert(self, collection_name: str, data: list[dict], **kwargs):
        """
        Insert data with transaction logic: write to DuckDB first, if success then write to Milvus.
        If Milvus fails, rollback DuckDB transaction. Ensure data consistency.
        """
        self._get_schema(collection_name)
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
            # Raise with exception context for better debugging
            raise RuntimeError(f"DuckDB insert failed: {e}") from e
        try:
            # Insert into Milvus
            result = super().insert(collection_name, data, **kwargs)
            self.duck_conn.execute("COMMIT;")
            return result
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            # Milvus write failed, rollback DuckDB
            # Raise with exception context for better debugging
            raise RuntimeError(f"Milvus insert failed, DuckDB rolled back: {e}") from e

    def delete(self, collection_name: str, ids: list[int | str], **kwargs):
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
            # Raise with exception context for better debugging
            raise RuntimeError(f"DuckDB delete failed: {e}") from e
        try:
            result = super().delete(collection_name, ids=ids, **kwargs)
            self.duck_conn.execute("COMMIT;")
            return result
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            # Raise with exception context for better debugging
            raise RuntimeError(f"Milvus delete failed, DuckDB rolled back: {e}") from e

    def upsert(self, collection_name: str, data: list[dict], **kwargs):
        """
        Upsert data with transaction logic: write to DuckDB first, if success then write to Milvus.
        If Milvus fails, rollback DuckDB transaction. Ensure data consistency.
        """
        self._get_schema(collection_name)
        df = pd.DataFrame(data)
        for field in self.json_fields:
            df[field] = df[field].apply(lambda x: json.dumps(x))
        self.duck_conn.register("df", df)
        try:
            self.duck_conn.execute("BEGIN TRANSACTION;")
            self.duck_conn.execute(f"""INSERT INTO {collection_name}
                                        SELECT * FROM df
                                        ON CONFLICT ({self.primary_field}) DO UPDATE SET
                                        {", ".join([f"{col} = EXCLUDED.{col}" for col in self.fields_name_list])}""")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            # Raise with exception context for better debugging
            raise RuntimeError(f"DuckDB upsert failed: {e}") from e
        try:
            result = super().upsert(collection_name, data, **kwargs)
            self.duck_conn.execute("COMMIT;")
            return result
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            # Raise with exception context for better debugging
            raise RuntimeError(f"Milvus upsert failed, DuckDB rolled back: {e}") from e

    def _milvus_filter_to_sql(self, filter: str):
        """
        Convert a Milvus filter expression to a SQL WHERE expression for DuckDB.
        Supports: ==, !=, >, <, >=, <=, IN, LIKE, IS NULL, IS NOT NULL, AND, OR, NOT, \
        JSON/ARRAY key/index access, arithmetic ops.
        """
        import re

        if not filter or filter.strip() == "":
            return "1=1"  # No filter, always true

        expr = filter

        # 1. Replace logical operators to uppercase (case-insensitive)
        expr = re.sub(r"\\b(and)\\b", "AND", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\\b(or)\\b", "OR", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\\b(not)\\b", "NOT", expr, flags=re.IGNORECASE)

        # 2. Replace comparison operators == -> =, != -> !=
        expr = re.sub(r"(?<![!<>])==", "=", expr)
        expr = re.sub(r"!=", "!=", expr)
        # >=, <=, >, < remain unchanged

        # 3. Replace IN: field in [..] -> field IN (..)
        def in_repl(match):
            field = match.group(1)
            values = match.group(2)
            # Convert [a, b, c] or ["a", "b"] to ('a','b')
            pylist = eval(values)
            sql_list = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in pylist])
            return f"{field} IN ({sql_list})"

        expr = re.sub(r"(\w+)\s+in\s+(\[[^\]]*\])", in_repl, expr, flags=re.IGNORECASE)

        # 4. LIKE: keep as is, but ensure string is single quoted
        expr = re.sub(r'LIKE\s+"([^"]*)"', lambda m: f"LIKE '{m.group(1)}'", expr)

        # 5. IS NULL / IS NOT NULL: keep as is, ensure uppercase
        expr = re.sub(r"is\s+null", "IS NULL", expr, flags=re.IGNORECASE)
        expr = re.sub(r"is\s+not\s+null", "IS NOT NULL", expr, flags=re.IGNORECASE)

        # 6. JSON key access: field["key"] -> field->>'key'
        expr = re.sub(r"(\w+)\[\"([\w_]+)\"\]", r"\1->>'\2'", expr)
        # ARRAY index: field[0] -> field[0] (DuckDB supports this syntax)

        # 7. String constants: "abc" -> 'abc'
        expr = re.sub(r'"([^"]*)"', lambda m: f"'{m.group(1)}'", expr)

        # 8. Remove redundant spaces
        expr = re.sub(r"\s+", " ", expr).strip()

        return expr

    def query(self, collection_name: str, filter: str = "", output_fields: list[str] = None):
        # Avoid mutable default argument
        if output_fields is None:
            output_fields = ["*"]
        milvus_res = super().query(collection_name, filter=filter, output_fields=output_fields)
        milvus_res = pd.DataFrame(milvus_res)
        sql_filter = self._milvus_filter_to_sql(filter)
        duckdb_res = self.duck_conn.execute(
            f"SELECT {', '.join(output_fields)} FROM {collection_name} WHERE {sql_filter}"
        ).fetchdf()
        milvus_res_df, duckdb_res_df = self._align_df(milvus_res, duckdb_res)

        return milvus_res_df, duckdb_res_df

    def export(self, collection_name: str):
        # only for duckdb, milvus does not support export
        return self.duck_conn.execute(f"SELECT * FROM {collection_name}").fetchdf()

    def count(self, collection_name: str):
        milvus_count = super().query(collection_name, filter="", output_fields=["count(*)"])
        duckdb_count = self.duck_conn.execute(f"SELECT COUNT(*) FROM {collection_name}").fetchone()
        res = {
            "milvus_count": milvus_count[0]["count(*)"],
            "duckdb_count": duckdb_count[0],
        }
        return res

    def _align_df(self, milvus_df: pd.DataFrame, duckdb_df: pd.DataFrame):
        # If primary_field is not already index, set it as index
        if self.primary_field not in milvus_df.index.names and self.primary_field in milvus_df.columns:
            milvus_df.set_index(self.primary_field, inplace=True)
        if self.primary_field not in duckdb_df.index.names and self.primary_field in duckdb_df.columns:
            duckdb_df.set_index(self.primary_field, inplace=True)
        # Only keep columns that exist in both DataFrames, and align column order
        common_cols = [col for col in milvus_df.columns if col in duckdb_df.columns]
        milvus_df = milvus_df[common_cols]
        duckdb_df = duckdb_df[common_cols]

        # Standardize types for json/array/vector fields
        for field in self.json_fields:
            if field in duckdb_df.columns:
                duckdb_df[field] = duckdb_df[field].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x and x[0] in ["{", "[", '"'] else x
                )
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(
                    lambda x: x
                    if isinstance(x, dict)
                    else json.loads(x)
                    if isinstance(x, str) and x and x[0] in ["{", "[", '"']
                    else x
                )
        for field in self.array_fields + self.float_vector_fields:
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(
                    lambda x: list(x) if not isinstance(x, list) and x is not None else x
                )
            if field in duckdb_df.columns:
                duckdb_df[field] = duckdb_df[field].apply(
                    lambda x: list(x) if not isinstance(x, list) and x is not None else x
                )

        # Align indexes in batch, only compare primary keys that exist in both sides
        shared_index = milvus_df.index.intersection(duckdb_df.index)
        milvus_aligned = milvus_df.loc[shared_index].sort_index()
        duckdb_aligned = duckdb_df.loc[shared_index].sort_index()

        # Strictly align the order of columns
        milvus_aligned = milvus_aligned.reindex(columns=duckdb_aligned.columns)
        duckdb_aligned = duckdb_aligned.reindex(columns=milvus_aligned.columns)
        return milvus_aligned, duckdb_aligned

    def _compare_df(self, milvus_df: pd.DataFrame, duckdb_df: pd.DataFrame):
        milvus_df, duckdb_df = self._align_df(milvus_df, duckdb_df)
        milvus_aligned_dict = milvus_df.to_dict(orient="records")
        duckdb_aligned_dict = duckdb_df.to_dict(orient="records")
        diff = DeepDiff(milvus_aligned_dict, duckdb_aligned_dict)
        if diff:
            logger.error(f"Data mismatch between Milvus and DuckDB:\n{diff}")
            return diff
        return diff

    def query_result_compare(self, collection_name: str, filter: str = "", output_fields: list[str] = None):
        # Avoid mutable default argument
        if output_fields is None:
            output_fields = ["*"]
        milvus_res, duckdb_res = self.query(collection_name, filter=filter, output_fields=output_fields)
        milvus_res_df = pd.DataFrame(milvus_res)
        duckdb_res_df = pd.DataFrame(duckdb_res)
        logger.info(f"Milvus query result:\n{milvus_res_df}")
        logger.info(f"DuckDB query result:\n{duckdb_res_df}")
        diff = self._compare_df(milvus_res_df, duckdb_res_df)
        if diff:
            logger.error(
                f"Query result mismatch for collection '{collection_name}' with filter "
                f"'{filter}' and output fields '{output_fields}'."
            )
            return diff
        else:
            logger.info(
                f"Query result match for collection '{collection_name}' with filter "
                f"'{filter}' and output fields '{output_fields}'."
            )
        return diff

    def entity_compare(self, collection_name: str, batch_size: int = 1000):
        self._get_schema(collection_name)
        count_res = self.count(collection_name)
        if count_res["milvus_count"] != count_res["duckdb_count"]:
            logger.error(
                f"Count mismatch for collection '{collection_name}': Milvus "
                f"({count_res['milvus_count']}) vs DuckDB ({count_res['duckdb_count']}). "
                f"Comparison aborted."
            )
            return False
        duckdb_pks_df = self.duck_conn.execute(f"SELECT {self.primary_field} FROM {collection_name}").fetchdf()
        total_pks = len(duckdb_pks_df)
        logger.debug(f"DuckDB PKs List:\n{duckdb_pks_df}")
        logger.info(f"Starting comparison for {total_pks} entries in collection '{collection_name}'.")
        # Define logging milestones
        milestones_to_log_at = set()
        if total_pks > 0:
            # Add 25%, 50%, 75% and 100% milestones
            milestones_to_log_at.add(max(1, total_pks // 4))
            milestones_to_log_at.add(max(1, total_pks // 2))
            milestones_to_log_at.add(max(1, (total_pks * 3) // 4))
            milestones_to_log_at.add(total_pks)  # Ensure the last item completion is logged
        # Batch fetch all primary keys
        pks = duckdb_pks_df[self.primary_field].tolist()
        if not pks:
            logger.info(f"No primary keys found in DuckDB for collection '{collection_name}'.")
            return True
        compared_count = 0
        for batch_start in range(0, len(pks), batch_size):
            batch_pks = pks[batch_start : batch_start + batch_size]
            # Query Milvus in batches
            milvus_filter = f"{self.primary_field} in {list(batch_pks)}"
            milvus_data = super().query(collection_name, filter=milvus_filter, output_fields=["*"])
            milvus_data_df = pd.DataFrame(milvus_data)
            # Query DuckDB in batches
            duckdb_data_df = self.duck_conn.execute(
                f"SELECT * FROM {collection_name} WHERE {self.primary_field} IN {tuple(batch_pks)}"
            ).fetchdf()

            diff = self._compare_df(milvus_data_df, duckdb_data_df)
            if diff:
                logger.error(f"Found difference(s) between Milvus and DuckDB for batch PKs:\n{diff}")

            # Check for primary keys that exist in DuckDB but not in Milvus
            only_in_duckdb = duckdb_data_df.index.difference(milvus_data_df.index)
            if len(only_in_duckdb) > 0:
                logger.error(f"PK(s) only in DuckDB, not found in Milvus: {list(only_in_duckdb)}")

            # Check for primary keys that exist in Milvus but not in DuckDB
            only_in_milvus = milvus_data_df.index.difference(duckdb_data_df.index)
            if len(only_in_milvus) > 0:
                logger.error(f"PK(s) only in Milvus, not found in DuckDB: {list(only_in_milvus)}")

            compared_count += len(batch_pks)
            if compared_count in milestones_to_log_at:
                percentage = (compared_count * 100) // total_pks if total_pks > 0 else 100
                logger.info(
                    f"Comparison progress for '{collection_name}': {compared_count}/{total_pks} "
                    f"({percentage}%) checked."
                )

        logger.info(f"Successfully completed comparison for {total_pks} entries in collection '{collection_name}'.")

    def _milvus_dtype_to_duckdb(self, milvus_type):
        MilvusDataTypeToDuckDBType = {
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
            raise ValueError(
                f"Unsupported data type: {milvus_type} when converting Milvus data type to DuckDB data type"
            )

    def sample_data(self, collection_name: str, num_samples: int = 100):
        """
        Sample data from the specified DuckDB collection.

        Args:
            collection_name (str): The name of the DuckDB table.
            num_samples (int): Number of rows to sample.

        Returns:
            pd.DataFrame: Sampled data as a DataFrame.
        """
        # Check schema and primary key
        self._get_schema(collection_name)
        # Use DuckDB's SAMPLE clause for efficient sampling
        query = f"SELECT * FROM {collection_name} USING SAMPLE {num_samples} ROWS"
        sampled_df = self.duck_conn.execute(query).fetchdf()
        return sampled_df

    def generate_milvus_filter(self, collection_name: str, num_samples: int = 100) -> str:
        """
        Generate diverse Milvus filter expressions from sample DataFrame using all scalar, JSON, and ARRAY fields.

        Args:
            collection_name (str): The name of the DuckDB/Milvus table.
            num_samples (int): Number of rows to sample.

        Returns:
            str: Milvus filter expression.
        """
        df = self.sample_data(collection_name, num_samples)
        schema = self._get_schema(collection_name)
        scalar_types = {"BOOL", "INT8", "INT16", "INT32", "INT64", "FLOAT", "DOUBLE", "VARCHAR"}
        exprs = []
        for field in [f for f in schema.fields if f.dtype.name in scalar_types and f.name in df.columns]:
            series = df[field.name]
            # Handle null value
            if series.isnull().any():
                exprs.append(f"{field.name} IS NULL")
                exprs.append(f"{field.name} IS NOT NULL")
            # Get unique values, handle unhashable types
            dtype_name = field.dtype.name
            values = series.dropna().unique()
            # Only one unique value
            if len(values) == 1:
                val = values[0]
                if dtype_name == "VARCHAR":
                    exprs.append(f"{field.name} == '{val}'")
                    exprs.append(f"{field.name} != '{val}'")
                    # LIKE expressions for strings
                    if len(val) > 2:
                        exprs.append(f"{field.name} LIKE '{val[:2]}%'")  # prefix
                        exprs.append(f"{field.name} LIKE '%{val[-2:]}'")  # suffix
                        exprs.append(f"{field.name} LIKE '%{val[1:-1]}%'")  # infix
                else:
                    exprs.append(f"{field.name} == {val}")
                    exprs.append(f"{field.name} != {val}")
            elif len(values) > 1:
                # Numeric fields
                if np.issubdtype(series.dtype, np.number):
                    minv, maxv = np.min(values), np.max(values)
                    exprs.append(f"{field.name} > {minv}")
                    exprs.append(f"{field.name} < {maxv}")
                    exprs.append(f"{field.name} >= {minv}")
                    exprs.append(f"{field.name} <= {maxv}")
                    # Modulus example if int
                    if np.issubdtype(series.dtype, np.integer):
                        exprs.append(f"{field.name} % 2 == 0")
                    else:
                        # Arithmetic operators
                        exprs.append(f"{field.name} + 1 == {minv + 1}")
                        exprs.append(f"{field.name} - 1 == {minv - 1}")
                        exprs.append(f"{field.name} * 2 == {minv * 2}")
                        exprs.append(f"{field.name} / 2 == {minv / 2}")
                        exprs.append(f"{field.name} % 2 == 0")
                        exprs.append(f"{field.name} ** 2 == {minv**2}")

                    # Range (between)
                    exprs.append(f"{field.name} >= {minv} AND {field.name} <= {maxv}")
                    # in/not in
                    vals = ", ".join(str(v) for v in values[:5])
                    exprs.append(f"{field.name} in [{vals}]")
                    exprs.append(f"{field.name} not in [{vals}]")
                # String fields
                elif isinstance(values[0], str):
                    vals = ", ".join(f"'{v}'" for v in values[:5])
                    exprs.append(f"{field.name} in [{vals}]")
                    exprs.append(f"{field.name} not in [{vals}]")
                    # LIKE: prefix/suffix/infix
                    for v in values[:3]:
                        if len(v) > 2:
                            exprs.append(f"{field.name} LIKE '{v[:2]}%'")
                            exprs.append(f"{field.name} LIKE '%{v[-2:]}'")
                            exprs.append(f"{field.name} LIKE '%{v[1:-1]}%'")
                # Bool fields
                elif dtype_name == "BOOL":
                    for v in values:
                        exprs.append(f"{field.name} == {str(v).lower()}")
                        exprs.append(f"{field.name} != {str(v).lower()}")
                ## TODO: support JSON and ARRAY fields
                # # JSON fields
                # elif dtype_name == "JSON":
                #     for v in values[:3]:
                #         exprs.append(f"{field.name} == '{json.dumps(v)}'")
                #         exprs.append(f"{field.name} != '{json.dumps(v)}'")
                #         for k, val in v.items():
                #             exprs.append(f"{field.name}.{k} == '{val}'")
                #             exprs.append(f"{field.name}.{k} != '{val}'")
                # # ARRAY fields
                # elif dtype_name == "ARRAY":
                #     for v in values[:3]:
                #         exprs.append(f"{field.name} == {v}")
                #         exprs.append(f"{field.name} != {v}")
                #         for i, val in enumerate(v):
                #             exprs.append(f"{field.name}[{i}] == {val}")
                #             exprs.append(f"{field.name}[{i}] != {val}")
        return exprs

