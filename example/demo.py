# Configuration section
# Define the Milvus client and collection name
from pymilvus_duckdb import MilvusDuckDBClient as MilvusClient
from pymilvus.milvus_client import IndexParams
from pymilvus import DataType
import random
import time

milvus_client = MilvusClient(
    uri="http://10.104.21.143:19530", duckdb_dir="./tmp/duckdb"
)
collection_name = f"test_collection_demo_{int(time.time())}"

# Define the schema for the collection
schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("name", DataType.VARCHAR, max_length=100)
schema.add_field("age", DataType.INT64)
schema.add_field("json_field", DataType.JSON)
schema.add_field(
    "array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10
)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

milvus_client.create_collection(collection_name, schema)
index_params = IndexParams()
index_params.add_index(
    "embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128}
)

milvus_client.create_index(collection_name, index_params)

milvus_client.load_collection(collection_name)


milvus_client.insert(
    collection_name,
    [
        {
            "id": i,
            "name": f"test_{i}",
            "age": i,
            "json_field": {"a": i, "b": i + 1},
            "array_field": [i, i + 1, i + 2],
            "embedding": [random.random() for _ in range(8)],
        }
        for i in range(10)
    ],
)

milvus_client.delete(collection_name, ids=[1, 2, 3])

milvus_client.upsert(
    collection_name,
    [
        {
            "id": i,
            "name": f"test_{i + 100}",
            "age": i + 100,
            "json_field": {"a": i + 100, "b": i + 101},
            "array_field": [i + 100, i + 101, i + 102],
            "embedding": [random.random() for _ in range(8)],
        }
        for i in range(4, 8)
    ],
)

time.sleep(1)
res = milvus_client.query(collection_name, "id > 0")
print(res)

res = milvus_client.export(collection_name)
print(res)

res = milvus_client.count(collection_name)
print(res)

milvus_client.compare(collection_name)
