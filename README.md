# PyMilvus DuckDB

`pymilvus_duckdb` is a Python library primarily designed for **validating Milvus data correctness**. It achieves this by synchronizing Milvus write operations (inserts, deletes, upserts) to a DuckDB database in real-time. By comparing the data in Milvus with the synchronized data in DuckDB, users can verify the consistency and accuracy of their Milvus deployments. While it facilitates data synchronization, its core utility lies in providing a robust mechanism for data validation.

## Features

*   **Milvus Client Extension**: Extends the `MilvusClient` functionality.
*   **Data Synchronization**: Keeps data in Milvus and a local DuckDB instance synchronized.
*   **Data Export**: Allows exporting collection data from the synchronized DuckDB instance.
*   **Query Correctness Validation**: Enables verification of Milvus query results by comparing it against a synchronized DuckDB instance.
*   **Milvus Data Correctness Validation**: Enables verification of Milvus data by comparing it against a synchronized DuckDB instance.

## Installation

To install `pymilvus_duckdb`, you can use pip after installing PDM or directly if the package is published:

```bash
# Ensure you have pdm installed if you are working with the source
# pip install pdm

# Install dependencies using pdm (from project root)
# pdm install

# Or install the package if available on PyPI (example)
# pip install pymilvus_duckdb
```

## Usage

Here's a basic example of how to use `pymilvus_duckdb`:

```python
from pymilvus_duckdb import MilvusDuckDBClient as MilvusClient
from pymilvus.milvus_client import IndexParams
from pymilvus import DataType
import random
import time

# Initialize the client
# Replace with your Milvus URI and desired DuckDB directory
milvus_client = MilvusClient(uri="http://localhost:19530", duckdb_dir="./tmp/duckdb_sync")

collection_name = f"my_collection_{int(time.time())}"

# 1. Create schema
schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("name", DataType.VARCHAR, max_length=100)
schema.add_field("age", DataType.INT64)
schema.add_field("json_field", DataType.JSON)
schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=8)

# 2. Create collection
milvus_client.create_collection(collection_name, schema)

# 3. Create index for the vector field
index_params = IndexParams()
index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
milvus_client.create_index(collection_name, index_params)

# 4. Load collection
milvus_client.load_collection(collection_name)

# 5. Insert data
data_to_insert = [
    {
        "id": i,
        "name": f"item_{i}",
        "age": 20 + i,
        "json_field": {"category": f"cat_{i%3}", "value": i * 10},
        "array_field": [i, i + 1, i + 2],
        "embedding": [random.random() for _ in range(8)]
    } for i in range(10)
]
milvus_client.insert(collection_name, data_to_insert)
print(f"Inserted {len(data_to_insert)} entities.")

# 6. Query data (from Milvus, synchronized to DuckDB)
# Wait a bit for synchronization if operations are very fast
time.sleep(1) 
query_res = milvus_client.query(collection_name, filter_expression="age > 25")
print("Query results (age > 25):")
for entity in query_res:
    print(entity)

# 7. Delete data
ids_to_delete = [0, 1, 2]
milvus_client.delete(collection_name, ids=ids_to_delete)
print(f"Deleted entities with IDs: {ids_to_delete}")

# 8. Upsert data
data_to_upsert = [
    {
        "id": i,
        "name": f"updated_item_{i}",
        "age": 30 + i,
        "json_field": {"category": f"cat_updated_{i%3}", "value": i * 100},
        "array_field": [i*2, i*2 + 1, i*2 + 2],
        "embedding": [random.random() for _ in range(8)]
    } for i in range(3, 7) # Upserting IDs 3,4,5,6 (some new, some existing)
]
milvus_client.upsert(collection_name, data_to_upsert)
print(f"Upserted {len(data_to_upsert)} entities.")

# 9. Export data (from DuckDB)
# Wait for sync
time.sleep(1)
exported_data = milvus_client.export(collection_name)
print(f"Exported data from DuckDB for collection '{collection_name}':")
for row in exported_data:
    print(row)

# Clean up (optional)
# milvus_client.drop_collection(collection_name)

print("Demo finished.")
```


## License

This project is licensed under the MIT License. See the `LICENSE` file for details (if one exists, otherwise specified in `pyproject.toml`).


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
