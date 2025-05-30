import logging

from pymilvus_duckdb import MilvusDuckDBClient as MilvusClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
MILVUS_URI = "http://10.104.21.143:19530"  # URI for Milvus server
DUCKDB_DIR = "./data/duckdb_complex"  # Directory for DuckDB data
COLLECTION_NAME = "complex_test_collection_1748571650"

milvus_client = MilvusClient(uri=MILVUS_URI, duckdb_dir=DUCKDB_DIR)
print("compare")
milvus_client.compare(COLLECTION_NAME)
