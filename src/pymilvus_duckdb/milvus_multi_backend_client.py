import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from deepdiff import DeepDiff
from pymilvus import Collection, CollectionSchema, DataType, MilvusClient, connections

from .backends import BaseBackend, DuckDBBackend, ClickHouseBackend
from .logger_config import logger


class MilvusMultiBackendClient(MilvusClient):
    """
    Milvus client with configurable backend support.
    Supports DuckDB and ClickHouse backends for data synchronization.
    """
    
    def __init__(self, backend_type: str = "duckdb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize Milvus connection
        uri = kwargs.get("uri", "")
        token = kwargs.get("token", "")
        connections.connect(uri=uri, token=token)
        
        # Initialize backend
        self.backend_type = backend_type.lower()
        self.backend = self._create_backend(self.backend_type, **kwargs)
        self.backend.connect(**kwargs)
        
        logger.info(f"Initialized MilvusMultiBackendClient with backend: {self.backend_type}")
    
    def _create_backend(self, backend_type: str, **kwargs) -> BaseBackend:
        """Create backend instance based on type."""
        if backend_type == "duckdb":
            return DuckDBBackend(**kwargs)
        elif backend_type == "clickhouse":
            return ClickHouseBackend(**kwargs)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}. Supported: 'duckdb', 'clickhouse'")
    
    def __del__(self):
        try:
            if hasattr(self, 'backend'):
                self.backend.close()
        except Exception as e:
            logger.error(f"Failed to close backend connection: {e}")

    def _get_schema(self, collection_name: str):
        """Get collection schema and update backend field lists."""
        c = Collection(collection_name)
        schema = c.schema
        
        # Reset backend field lists
        self.backend.fields_name_list = []
        self.backend.json_fields = []
        self.backend.array_fields = []
        self.backend.varchar_fields = []
        self.backend.float_vector_fields = []
        
        for field in schema.fields:
            if field.is_primary:
                self.backend.primary_field = field.name
            if field.dtype == DataType.FLOAT_VECTOR:
                self.backend.float_vector_fields.append(field.name)
            if field.dtype == DataType.ARRAY:
                self.backend.array_fields.append(field.name)
            if field.dtype == DataType.JSON:
                self.backend.json_fields.append(field.name)
            if field.dtype == DataType.VARCHAR:
                self.backend.varchar_fields.append(field.name)
            self.backend.fields_name_list.append(field.name)
        return schema

    def create_collection(self, collection_name: str, schema: CollectionSchema = None, **kwargs):
        """Create a collection: first create table in backend, then create collection in Milvus."""
        # Step 1: Prepare backend table schema
        self.backend.fields = []
        schema_fields = []
        
        for field in schema.fields:
            if field.auto_id:
                raise ValueError(f"Auto ID is not supported for {self.backend_type} backend")
            
            name = field.name
            backend_type = self.backend.milvus_dtype_to_backend_dtype(field.dtype)
            
            if field.is_primary:
                self.backend.primary_field = field.name
                if self.backend_type == "duckdb":
                    backend_type = f"{backend_type} PRIMARY KEY"
            
            if field.dtype == DataType.FLOAT_VECTOR:
                dim = field.params.get("dim", 0)
                if self.backend_type == "duckdb":
                    backend_type = f"FLOAT[{dim}]"
                else:  # clickhouse
                    backend_type = "Array(Float32)"
            
            if field.dtype == DataType.ARRAY:
                element_type = self.backend.milvus_dtype_to_backend_dtype(field.element_type)
                if self.backend_type == "duckdb":
                    backend_type = f"{element_type}[]"
                else:  # clickhouse
                    backend_type = f"Array({element_type})"
            
            # Format field definition based on backend
            if self.backend_type == "clickhouse":
                field_def = f"`{name}` {backend_type}"
            else:  # duckdb
                field_def = f"{name} {backend_type}"
            
            schema_fields.append(field_def)
            self.backend.fields_name_list.append(name)
            
            if field.dtype == DataType.JSON:
                self.backend.json_fields.append(name)
            if field.dtype == DataType.ARRAY:
                self.backend.array_fields.append(name)
            if field.dtype == DataType.FLOAT_VECTOR:
                self.backend.float_vector_fields.append(name)
            if field.dtype == DataType.VARCHAR:
                self.backend.varchar_fields.append(name)
        
        try:
            # Step 2: Create table in backend
            self.backend.create_table(collection_name, schema_fields)
            try:
                # Step 3: Create collection in Milvus
                result = super().create_collection(collection_name, schema=schema, consistency_level="Strong", **kwargs)
                return result
            except Exception as milvus_exc:
                # Step 4: If Milvus fails, drop backend table
                self.backend.drop_table(collection_name)
                raise milvus_exc
        except Exception as backend_exc:
            logger.error(f"Failed to create {self.backend_type} table: {backend_exc}")
            raise backend_exc

    def drop_collection(self, collection_name: str):
        """Drop collection in both Milvus and backend."""
        try:
            # Drop backend table first
            self.backend.drop_table(collection_name)
            # Then drop Milvus collection
            super().drop_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to drop collection '{collection_name}': {e}")
            raise e

    def insert(self, collection_name: str, data: list[dict], **kwargs):
        """Insert data with backend-first strategy and rollback on failure."""
        operation_start_time = time.time()
        logger.info(f"开始执行 insert 操作，集合: {collection_name}，数据条数: {len(data)}，后端: {self.backend_type}")
        
        self._get_schema(collection_name)
        df = pd.DataFrame(data)
        
        try:
            backend_start_time = time.time()
            # Insert into backend first
            self.backend.insert_data(collection_name, df)
            backend_end_time = time.time()
            logger.info(f"{self.backend_type.capitalize()} insert 操作耗时: {backend_end_time - backend_start_time:.4f} 秒")
        except Exception as e:
            operation_end_time = time.time()
            logger.error(f"{self.backend_type.capitalize()} insert 操作失败，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            raise RuntimeError(f"{self.backend_type.capitalize()} insert failed: {e}") from e
        
        try:
            # Insert into Milvus
            milvus_start_time = time.time()
            result = super().insert(collection_name, data, **kwargs)
            milvus_end_time = time.time()
            logger.info(f"Milvus insert 操作耗时: {milvus_end_time - milvus_start_time:.4f} 秒")
            operation_end_time = time.time()
            logger.info(f"insert 操作完成，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            return result
        except Exception as e:
            # Rollback backend data
            try:
                pks = [item[self.backend.primary_field] for item in data]
                self.backend.rollback_insert(collection_name, pks)
                logger.info(f"{self.backend_type.capitalize()} data rolled back due to Milvus insert failure")
            except Exception as rollback_exc:
                logger.error(f"Failed to rollback {self.backend_type} data: {rollback_exc}")
            
            operation_end_time = time.time()
            logger.error(f"Milvus insert 操作失败，已回滚，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            raise RuntimeError(f"Milvus insert failed, {self.backend_type} rolled back: {e}") from e

    def delete(self, collection_name: str, ids: list[int | str], **kwargs):
        """Delete data with backend-first strategy and rollback on failure."""
        operation_start_time = time.time()
        logger.info(f"开始执行 delete 操作，集合: {collection_name}，删除ID数量: {len(ids)}，后端: {self.backend_type}")
        
        self._get_schema(collection_name)
        backup_data = None
        
        try:
            backend_start_time = time.time()
            # Delete from backend and get backup for rollback
            backup_data = self.backend.delete_data(collection_name, ids)
            backend_end_time = time.time()
            logger.info(f"{self.backend_type.capitalize()} delete 操作耗时: {backend_end_time - backend_start_time:.4f} 秒")
        except Exception as e:
            operation_end_time = time.time()
            logger.error(f"{self.backend_type.capitalize()} delete 操作失败，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            raise RuntimeError(f"{self.backend_type.capitalize()} delete failed: {e}") from e
        
        try:
            milvus_start_time = time.time()
            result = super().delete(collection_name, ids=ids, **kwargs)
            milvus_end_time = time.time()
            logger.info(f"Milvus delete 操作耗时: {milvus_end_time - milvus_start_time:.4f} 秒")
            operation_end_time = time.time()
            logger.info(f"delete 操作完成，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            return result
        except Exception as e:
            # Rollback: re-insert the backed up data
            self.backend.rollback_delete(collection_name, backup_data)
            logger.info(f"{self.backend_type.capitalize()} data rolled back due to Milvus delete failure")
            
            operation_end_time = time.time()
            logger.error(f"Milvus delete 操作失败，已回滚，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            raise RuntimeError(f"Milvus delete failed, {self.backend_type} rolled back: {e}") from e

    def upsert(self, collection_name: str, data: list[dict], **kwargs):
        """Upsert data with backend-first strategy and rollback on failure."""
        operation_start_time = time.time()
        logger.info(f"开始执行 upsert 操作，集合: {collection_name}，数据条数: {len(data)}，后端: {self.backend_type}")
        
        self._get_schema(collection_name)
        df = pd.DataFrame(data)
        backup_data = None
        
        try:
            backend_start_time = time.time()
            # Upsert into backend and get backup for rollback
            backup_data = self.backend.upsert_data(collection_name, df, self.backend.primary_field)
            backend_end_time = time.time()
            logger.info(f"{self.backend_type.capitalize()} upsert 操作耗时: {backend_end_time - backend_start_time:.4f} 秒")
        except Exception as e:
            operation_end_time = time.time()
            logger.error(f"{self.backend_type.capitalize()} upsert 操作失败，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            raise RuntimeError(f"{self.backend_type.capitalize()} upsert failed: {e}") from e
        
        try:
            milvus_start_time = time.time()
            result = super().upsert(collection_name, data, **kwargs)
            milvus_end_time = time.time()
            logger.info(f"Milvus upsert 操作耗时: {milvus_end_time - milvus_start_time:.4f} 秒")
            operation_end_time = time.time()
            logger.info(f"upsert 操作完成，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            return result
        except Exception as e:
            # Rollback: restore original data
            pks = [item[self.backend.primary_field] for item in data]
            self.backend.rollback_upsert(collection_name, pks, backup_data)
            logger.info(f"{self.backend_type.capitalize()} data rolled back due to Milvus upsert failure")
            
            operation_end_time = time.time()
            logger.error(f"Milvus upsert 操作失败，已回滚，总耗时: {operation_end_time - operation_start_time:.4f} 秒")
            raise RuntimeError(f"Milvus upsert failed, {self.backend_type} rolled back: {e}") from e

    def query(self, collection_name: str, filter: str = "", output_fields: list[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Query data from both Milvus and backend, return both results."""
        if output_fields is None:
            output_fields = ["*"]
        
        milvus_res = super().query(collection_name, filter=filter, output_fields=output_fields)
        milvus_res = pd.DataFrame(milvus_res)
        
        backend_res = self.backend.query_data(collection_name, filter, output_fields)
        milvus_res_df, backend_res_df = self._align_df(milvus_res, backend_res)

        return milvus_res_df, backend_res_df

    def export(self, collection_name: str):
        """Export data from backend only (Milvus does not support export)."""
        return self.backend.export_data(collection_name)

    def count(self, collection_name: str):
        """Count records in both Milvus and backend."""
        milvus_count = super().query(collection_name, filter="", output_fields=["count(*)"])
        backend_count = self.backend.count_records(collection_name)
        
        res = {
            "milvus_count": milvus_count[0]["count(*)"],
            f"{self.backend_type}_count": backend_count,
        }
        return res

    def _align_df(self, milvus_df: pd.DataFrame, backend_df: pd.DataFrame):
        """Align DataFrames from Milvus and backend for comparison."""
        # If primary_field is not already index, set it as index
        if self.backend.primary_field not in milvus_df.index.names and self.backend.primary_field in milvus_df.columns:
            milvus_df.set_index(self.backend.primary_field, inplace=True)
        if self.backend.primary_field not in backend_df.index.names and self.backend.primary_field in backend_df.columns:
            backend_df.set_index(self.backend.primary_field, inplace=True)
        
        # Only keep columns that exist in both DataFrames, and align column order
        common_cols = [col for col in milvus_df.columns if col in backend_df.columns]
        milvus_df = milvus_df[common_cols]
        backend_df = backend_df[common_cols]

        # Standardize types for json/array/vector fields
        for field in self.backend.json_fields:
            if field in backend_df.columns:
                backend_df[field] = backend_df[field].apply(
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
        
        for field in self.backend.array_fields + self.backend.float_vector_fields:
            if field in milvus_df.columns:
                milvus_df[field] = milvus_df[field].apply(
                    lambda x: list(x) if not isinstance(x, list) and x is not None else x
                )
            if field in backend_df.columns:
                backend_df[field] = backend_df[field].apply(
                    lambda x: list(x) if not isinstance(x, list) and x is not None else x
                )

        # Align indexes in batch, only compare primary keys that exist in both sides
        shared_index = milvus_df.index.intersection(backend_df.index)
        milvus_aligned = milvus_df.loc[shared_index].sort_index()
        backend_aligned = backend_df.loc[shared_index].sort_index()

        # Strictly align the order of columns
        milvus_aligned = milvus_aligned.reindex(columns=backend_aligned.columns)
        backend_aligned = backend_aligned.reindex(columns=milvus_aligned.columns)
        return milvus_aligned, backend_aligned

    def _compare_df(self, milvus_df: pd.DataFrame, backend_df: pd.DataFrame):
        """Compare DataFrames from Milvus and backend."""
        milvus_df, backend_df = self._align_df(milvus_df, backend_df)
        milvus_aligned_dict = milvus_df.to_dict(orient="records")
        backend_aligned_dict = backend_df.to_dict(orient="records")
        diff = DeepDiff(milvus_aligned_dict, backend_aligned_dict)
        if diff:
            logger.error(f"Data mismatch between Milvus and {self.backend_type}:\n{diff}")
            return diff
        return diff

    def query_result_compare(self, collection_name: str, filter: str = "", output_fields: list[str] = None):
        """Compare query results between Milvus and backend."""
        if output_fields is None:
            output_fields = ["*"]
        milvus_res, backend_res = self.query(collection_name, filter=filter, output_fields=output_fields)
        milvus_res_df = pd.DataFrame(milvus_res)
        backend_res_df = pd.DataFrame(backend_res)
        logger.info(f"Milvus query result:\n{milvus_res_df}")
        logger.info(f"{self.backend_type.capitalize()} query result:\n{backend_res_df}")
        diff = self._compare_df(milvus_res_df, backend_res_df)
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
        """Compare all entities between Milvus and backend."""
        self._get_schema(collection_name)
        count_res = self.count(collection_name)
        milvus_count = count_res["milvus_count"]
        backend_count = count_res[f"{self.backend_type}_count"]
        
        if milvus_count != backend_count:
            logger.error(
                f"Count mismatch for collection '{collection_name}': Milvus "
                f"({milvus_count}) vs {self.backend_type.capitalize()} ({backend_count}). "
                f"Comparison aborted."
            )
            return False
        
        # Get all primary keys from backend
        all_data = self.backend.export_data(collection_name)
        if self.backend.primary_field not in all_data.columns:
            logger.error(f"Primary field '{self.backend.primary_field}' not found in backend data")
            return False
            
        pks = all_data[self.backend.primary_field].tolist()
        total_pks = len(pks)
        
        logger.debug(f"{self.backend_type.capitalize()} PKs count: {total_pks}")
        logger.info(f"Starting comparison for {total_pks} entries in collection '{collection_name}'.")
        
        # Define logging milestones
        milestones_to_log_at = set()
        if total_pks > 0:
            milestones_to_log_at.add(max(1, total_pks // 4))
            milestones_to_log_at.add(max(1, total_pks // 2))
            milestones_to_log_at.add(max(1, (total_pks * 3) // 4))
            milestones_to_log_at.add(total_pks)
        
        if not pks:
            logger.info(f"No primary keys found in {self.backend_type} for collection '{collection_name}'.")
            return True
        
        compared_count = 0
        for batch_start in range(0, len(pks), batch_size):
            batch_pks = pks[batch_start : batch_start + batch_size]
            
            # Query Milvus in batches
            milvus_filter = f"{self.backend.primary_field} in {list(batch_pks)}"
            milvus_data = super().query(collection_name, filter=milvus_filter, output_fields=["*"])
            milvus_data_df = pd.DataFrame(milvus_data)
            
            # Query backend in batches
            backend_filter = f"{self.backend.primary_field} in {batch_pks}"
            backend_data_df = self.backend.query_data(collection_name, backend_filter, ["*"])

            diff = self._compare_df(milvus_data_df, backend_data_df)
            if diff:
                logger.error(f"Found difference(s) between Milvus and {self.backend_type} for batch PKs:\n{diff}")

            # Check for primary keys that exist in backend but not in Milvus
            if hasattr(milvus_data_df, 'index') and hasattr(backend_data_df, 'index'):
                only_in_backend = backend_data_df.index.difference(milvus_data_df.index)
                if len(only_in_backend) > 0:
                    logger.error(f"PK(s) only in {self.backend_type}, not found in Milvus: {list(only_in_backend)}")

                # Check for primary keys that exist in Milvus but not in backend
                only_in_milvus = milvus_data_df.index.difference(backend_data_df.index)
                if len(only_in_milvus) > 0:
                    logger.error(f"PK(s) only in Milvus, not found in {self.backend_type}: {list(only_in_milvus)}")

            compared_count += len(batch_pks)
            if compared_count in milestones_to_log_at:
                percentage = (compared_count * 100) // total_pks if total_pks > 0 else 100
                logger.info(
                    f"Comparison progress for '{collection_name}': {compared_count}/{total_pks} "
                    f"({percentage}%) checked."
                )

        logger.info(f"Successfully completed comparison for {total_pks} entries in collection '{collection_name}'.")
        return True

    def sample_data(self, collection_name: str, num_samples: int = 100):
        """Sample data from the backend."""
        self._get_schema(collection_name)
        return self.backend.sample_data(collection_name, num_samples)

    def generate_milvus_filter(self, collection_name: str, num_samples: int = 100) -> List[str]:
        """Generate diverse Milvus filter expressions from sample data."""
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
        return exprs
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "backend_type": self.backend_type,
            "backend_class": self.backend.__class__.__name__,
            "supported_operations": [
                "create_collection", "drop_collection", "insert", "delete", 
                "upsert", "query", "count", "export", "sample_data",
                "entity_compare", "query_result_compare"
            ]
        } 