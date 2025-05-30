# coding: utf-8
"""
Test script for MilvusDuckDBClient._milvus_filter_to_sql
Verify the correctness of converting Milvus filter expressions to SQL WHERE expressions.
"""

from pymilvus_duckdb import MilvusDuckDBClient
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus.milvus_client import IndexParams


# Build test collection schema
def build_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(
            name="status", dtype=DataType.VARCHAR, max_length=20, nullable=True
        ),
        FieldSchema(name="age", dtype=DataType.INT32, nullable=True),
        FieldSchema(name="price", dtype=DataType.FLOAT, nullable=True),
        FieldSchema(name="rating", dtype=DataType.INT32, nullable=True),
        FieldSchema(name="discount", dtype=DataType.FLOAT, nullable=True),
        FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=10, nullable=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=32, nullable=True),
        FieldSchema(
            name="description", dtype=DataType.VARCHAR, max_length=128, nullable=True
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8),
    ]
    return CollectionSchema(fields, description="test schema")


# Build test data
def build_data():
    return [
        {
            "id": 1,
            "status": "active",
            "age": 35,
            "price": 120.5,
            "rating": 5,
            "discount": 8,
            "color": "red",
            "name": "ProdA",
            "description": "good",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
        {
            "id": 2,
            "status": "inactive",
            "age": 28,
            "price": 80,
            "rating": 3,
            "discount": 15,
            "color": "blue",
            "name": "ProdB",
            "description": None,
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
        {
            "id": 3,
            "status": "active",
            "age": 40,
            "price": 60,
            "rating": 4,
            "discount": 10,
            "color": "green",
            "name": "SuperPro",
            "description": "",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
        {
            "id": 4,
            "status": "active",
            "age": 22,
            "price": 200,
            "rating": 2,
            "discount": 5,
            "color": "red",
            "name": "ProductXYZ",
            "description": "cheap",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
        {
            "id": 5,
            "status": "inactive",
            "age": 33,
            "price": 150,
            "rating": 4,
            "discount": 10,
            "color": "green",
            "name": "ProLine",
            "description": "premium",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
    ]


def main():
    # --- Configuration ---
    MILVUS_URI = "http://10.104.21.143:19530"  # URI for Milvus server
    DUCKDB_DIR = "./data/duckdb_sql"  # Directory for DuckDB data
    client = MilvusDuckDBClient(uri=MILVUS_URI, duckdb_dir=DUCKDB_DIR)
    collection_name = "test_filter_to_sql"
    # Clean up old collection if exists
    try:
        client.drop_collection(collection_name)
    except Exception:
        pass
    # Create collection
    schema = build_schema()
    index_params = IndexParams()
    index_params.add_index(
        "embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128}
    )
    client.create_collection(collection_name, schema=schema, index_params=index_params)
    client.load_collection(collection_name)
    # Insert test data
    data = build_data()
    client.insert(collection_name, data)
    print("Inserted data:")
    for d in data:
        print(d)
    # Verify part of filter conversion and query results
    test_filters = [
        'status == "active"',
        "age > 30 and price < 200",
        'color in ["red", "green"]',
        'name LIKE "Pro%"',
        "description IS NULL",
        "discount <= 10 or rating >= 5",
    ]
    print("\n=== Query verification ===")
    for f in test_filters:
        sql_where = client._milvus_filter_to_sql(f)
        milvus_res, duckdb_res = client.query(
            collection_name, filter=f, output_fields=["*"]
        )
        print(f"\nMilvus filter: {f}")
        print(f"SQL where:    {sql_where}")
        print("Milvus result:")
        print(milvus_res)
        print("DuckDB result:")
        print(duckdb_res)
        client.query_result_compare(collection_name, filter=f, output_fields=["*"])


if __name__ == "__main__":
    main()
