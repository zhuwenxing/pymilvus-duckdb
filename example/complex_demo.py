import time
import random
import logging
from pymilvus_duckdb import MilvusDuckDBClient as MilvusClient
from pymilvus.milvus_client import IndexParams
from pymilvus import DataType

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration section
MILVUS_URI = "http://10.104.21.143:19530"  # Milvus server URI
DUCKDB_DIR = "./tmp/duckdb_complex"  # Directory to store DuckDB data
COLLECTION_NAME_PREFIX = "complex_test_collection"

DIMENSION = 8  # Embedding vector dimension
BATCH_SIZE = 50000  # Number of records per batch operation

# Total number of operations to perform
TOTAL_OPERATIONS = 10_000_000
INSERT_RATIO = 0.6
DELETE_RATIO = 0.2
UPSERT_RATIO = 0.2

NUM_INSERTS = int(TOTAL_OPERATIONS * INSERT_RATIO)
NUM_DELETES = int(TOTAL_OPERATIONS * DELETE_RATIO)
NUM_UPSERTS = int(TOTAL_OPERATIONS * UPSERT_RATIO)


# --- Helper function to generate sample data ---
def generate_data(start_id, count, for_upsert=False):
    data = []
    for i in range(count):
        current_id = start_id + i
        record = {
            "id": current_id,
            "name": f"name_{current_id}{'_upserted' if for_upsert else ''}",
            "age": random.randint(18, 60)
            + (100 if for_upsert else 0),  # Differentiate upserted data
            "json_field": {"attr1": current_id, "attr2": f"val_{current_id}"},
            "array_field": [
                current_id,
                current_id + 1,
                current_id + 2,
                random.randint(0, 100),
            ],
            "embedding": [random.random() for _ in range(DIMENSION)],
        }
        data.append(record)
    return data


def main():
    logging.info(f"Starting complex demo with {TOTAL_OPERATIONS} operations.")
    logging.info(
        f"Inserts: {NUM_INSERTS}, Deletes: {NUM_DELETES}, Upserts: {NUM_UPSERTS}"
    )

    # --- Initialize Milvus Client ---
    milvus_client = MilvusClient(uri=MILVUS_URI, duckdb_dir=DUCKDB_DIR)
    collection_name = f"{COLLECTION_NAME_PREFIX}_{int(time.time())}"
    logging.info(f"Using collection: {collection_name}")

    # --- Create Collection ---
    if milvus_client.has_collection(collection_name):
        logging.warning(f"Collection '{collection_name}' already exists. Dropping it.")
        milvus_client.drop_collection(collection_name)

    schema = milvus_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(
        "name", DataType.VARCHAR, max_length=256
    )  # Increased max_length for longer names
    schema.add_field("age", DataType.INT64)
    schema.add_field("json_field", DataType.JSON)
    schema.add_field(
        "array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=20
    )  # Increased max_capacity
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)

    milvus_client.create_collection(collection_name, schema)
    logging.info(f"Collection '{collection_name}' created successfully.")

    # --- Create Index ---
    index_params = IndexParams()
    index_params.add_index(
        "embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128}
    )
    milvus_client.create_index(collection_name, index_params)
    logging.info("Index created successfully.")

    # --- Load Collection ---
    milvus_client.load_collection(collection_name)
    logging.info(f"Collection '{collection_name}' loaded.")
    time.sleep(2)  # Give time for loading to complete

    # --- Perform Insert Operations (6M) ---
    logging.info(
        f"Starting {NUM_INSERTS} insert operations in batches of {BATCH_SIZE}..."
    )
    insert_start_id = 0
    for i in range(0, NUM_INSERTS, BATCH_SIZE):
        batch_data = generate_data(
            insert_start_id + i, min(BATCH_SIZE, NUM_INSERTS - i)
        )
        milvus_client.insert(collection_name, batch_data)
        logging.info(
            f"Inserted batch {i // BATCH_SIZE + 1}/{(NUM_INSERTS + BATCH_SIZE - 1) // BATCH_SIZE}, {len(batch_data)} records."
        )
    logging.info(f"Finished {NUM_INSERTS} insert operations.")

    # --- Perform Delete Operations (2M) ---
    # These IDs were inserted in the previous step.
    logging.info(
        f"Starting {NUM_DELETES} delete operations in batches of {BATCH_SIZE}..."
    )
    delete_start_id = 0  # Delete from the beginning of the inserted IDs
    for i in range(0, NUM_DELETES, BATCH_SIZE):
        ids_batch = list(
            range(
                delete_start_id + i,
                delete_start_id + i + min(BATCH_SIZE, NUM_DELETES - i),
            )
        )
        milvus_client.delete(collection_name, ids=ids_batch)
        logging.info(
            f"Deleted batch {i // BATCH_SIZE + 1}/{(NUM_DELETES + BATCH_SIZE - 1) // BATCH_SIZE}, {len(ids_batch)} records."
        )
    logging.info(f"Finished {NUM_DELETES} delete operations.")

    # Wait for deletes to be processed
    time.sleep(5)
    # Query to check count after delete (optional, can be slow)
    # res_after_delete = milvus_client.query(collection_name, expr="id >= 0", output_fields=["count(*)"])
    # logging.info(f"Count after delete: {res_after_delete[0]['count(*)'] if res_after_delete else 'N/A'}")

    # --- Perform Upsert Operations (2M) ---
    # 1M updates (IDs that were inserted and not deleted)
    # 1M new inserts
    logging.info(
        f"Starting {NUM_UPSERTS} upsert operations in batches of {BATCH_SIZE}..."
    )

    num_updates_via_upsert = NUM_UPSERTS // 2
    num_inserts_via_upsert = NUM_UPSERTS - num_updates_via_upsert

    # IDs for update: e.g., from 2,000,000 to 2,999,999 (assuming NUM_DELETES was 2M starting from ID 0)
    upsert_update_start_id = NUM_DELETES
    # IDs for new inserts: e.g., from 6,000,000 to 6,999,999
    upsert_new_insert_start_id = NUM_INSERTS

    # Upserts that will update existing records
    for i in range(0, num_updates_via_upsert, BATCH_SIZE):
        batch_data = generate_data(
            upsert_update_start_id + i,
            min(BATCH_SIZE, num_updates_via_upsert - i),
            for_upsert=True,
        )
        milvus_client.upsert(collection_name, batch_data)
        logging.info(
            f"Upserted (update) batch {i // BATCH_SIZE + 1}/{(num_updates_via_upsert + BATCH_SIZE - 1) // BATCH_SIZE}, {len(batch_data)} records."
        )

    # Upserts that will insert new records
    for i in range(0, num_inserts_via_upsert, BATCH_SIZE):
        batch_data = generate_data(
            upsert_new_insert_start_id + i,
            min(BATCH_SIZE, num_inserts_via_upsert - i),
            for_upsert=True,
        )
        milvus_client.upsert(collection_name, batch_data)
        logging.info(
            f"Upserted (new insert) batch {i // BATCH_SIZE + 1}/{(num_inserts_via_upsert + BATCH_SIZE - 1) // BATCH_SIZE}, {len(batch_data)} records."
        )
    logging.info(f"Finished {NUM_UPSERTS} upsert operations.")

    # --- Final Verification ---
    time.sleep(5)  # Allow time for operations to settle
    logging.info("Performing final query to check data...")
    try:
        # Query a few records
        query_expr = f"id >= {upsert_new_insert_start_id} and id < {upsert_new_insert_start_id + 5}"
        res_query = milvus_client.query(
            collection_name, expr=query_expr, output_fields=["id", "name", "age"]
        )
        logging.info(f"Query results for '{query_expr}': {res_query}")

        # Get total count (can be slow on very large datasets)
        # res_count = milvus_client.query(collection_name, expr="id >= 0", output_fields=["count(*)"])
        # logging.info(f"Final entity count: {res_count[0]['count(*)'] if res_count else 'N/A'}")
        # Expected final count: (NUM_INSERTS - NUM_DELETES) + num_inserts_via_upsert
        # = (6M - 2M) + 1M = 5M
        expected_final_count = (NUM_INSERTS - NUM_DELETES) + num_inserts_via_upsert
        logging.info(f"Expected final entity count: {expected_final_count}")
        logging.warning(
            "Actual count query is commented out as it can be slow. You can enable it if needed."
        )

    except Exception as e:
        logging.error(f"Error during final verification: {e}")

    # --- Export Data (Optional) ---
    # logging.info("Exporting collection data...")
    # try:
    #     export_res = milvus_client.export(collection_name)
    #     logging.info(f"Export result: {export_res}") # This might be very large
    # except Exception as e:
    #     logging.error(f"Error during export: {e}")
    logging.warning(
        "Export is commented out as it can produce very large output for 5M records."
    )

    # --- Cleanup (Optional) ---
    # milvus_client.drop_collection(collection_name)
    # logging.info(f"Collection '{collection_name}' dropped.")
    logging.info(
        f"Demo finished. Collection '{collection_name}' still exists. You may want to drop it manually if not needed."
    )


if __name__ == "__main__":
    main()
