import argparse
import logging
import random
import threading
import time

from pymilvus import DataType
from pymilvus.milvus_client import IndexParams

from pymilvus_duckdb import MilvusDuckDBClient as MilvusClient

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration section
MILVUS_URI = "http://10.104.21.143:19530"  # Milvus server URI
DUCKDB_DIR = "./tmp/duckdb_complex"  # Directory to store DuckDB data
COLLECTION_NAME_PREFIX = "complex_test_collection"

DIMENSION = 8  # Embedding vector dimension
BATCH_SIZE = 1000  # Number of records per batch operation

# Ratios of insert, delete, upsert operations
INSERT_RATIO = 0.6
DELETE_RATIO = 0.2
UPSERT_RATIO = 0.2



global_id = 0
id_lock = threading.Lock()


def generate_data(start_id, count, for_upsert=False):
    data = []
    for i in range(count):
        current_id = start_id + i
        record = {
            "id": current_id,
            "name": f"name_{current_id}{'_upserted' if for_upsert else ''}",
            "age": random.randint(18, 60) + (100 if for_upsert else 0),  # Differentiate upserted data
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


def insert_worker(stop_event, milvus_client, collection_name):
    """Insert worker thread, each loop allocates id from shared global_id."""
    global global_id
    while not stop_event.is_set():
        try:
            # 在执行操作前再次检查停止信号
            if stop_event.is_set():
                break
                
            # Thread-safe allocation of unique id range for insert
            with id_lock:
                start_id = global_id
                batch_size = BATCH_SIZE
                global_id += batch_size
            batch_data = generate_data(start_id, batch_size)
            milvus_client.insert(collection_name, batch_data)
            logging.info(f"[INSERT] Inserted {batch_size} records, start id: {start_id}")
        except Exception as e:
            logging.error(f"[INSERT] Exception: {e}", exc_info=True)
        
        # 使用较短的检查间隔，能更快响应停止信号
        for _ in range(10):  # 分成10次，每次0.1秒
            if stop_event.is_set():
                break
            time.sleep(0.1)

def delete_worker(stop_event, milvus_client: MilvusClient, collection_name):
    """Delete worker thread, each loop allocates id from shared global_id, id range according to ratio."""
    global global_id
    while not stop_event.is_set():
        try:
            # 在执行操作前再次检查停止信号
            if stop_event.is_set():
                break
                
            # Thread-safe allocation of unique id range for delete
            with id_lock:
                start_id = global_id - BATCH_SIZE*2
                delete_batch_size = max(1, int(BATCH_SIZE * (DELETE_RATIO / INSERT_RATIO)))
                # global_id += delete_batch_size
            ids_batch = list(range(start_id, start_id + delete_batch_size))
            milvus_client.delete(collection_name, ids=ids_batch)
            logging.info(f"[DELETE] Deleted {len(ids_batch)} records, start id: {start_id}")
        except Exception as e:
            logging.error(f"[DELETE] Exception: {e}", exc_info=True)
        
        # 使用较短的检查间隔，能更快响应停止信号
        for _ in range(100):  # 分成100次，每次0.1秒，总共10秒
            if stop_event.is_set():
                break
            time.sleep(0.1)


def upsert_worker(stop_event, milvus_client, collection_name):
    """Upsert worker thread, each loop allocates id from shared global_id, id range according to ratio."""
    global global_id
    while not stop_event.is_set():
        try:
            # 在执行操作前再次检查停止信号
            if stop_event.is_set():
                break
                
            # Thread-safe allocation of unique id range for upsert
            with id_lock:
                start_id = global_id - BATCH_SIZE
                upsert_batch_size = max(1, int(BATCH_SIZE * (UPSERT_RATIO / INSERT_RATIO)))
                # global_id += upsert_batch_size
            batch_data = generate_data(start_id, upsert_batch_size, for_upsert=True)
            milvus_client.upsert(collection_name, batch_data)
            logging.info(f"[UPSERT] Upserted {len(batch_data)} records, start id: {start_id}")
        except Exception as e:
            logging.error(f"[UPSERT] Exception: {e}", exc_info=True)
        
        # 使用较短的检查间隔，能更快响应停止信号
        for _ in range(10):  # 分成10次，每次0.1秒
            if stop_event.is_set():
                break
            time.sleep(0.1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long run checker with error catching and duration control.")
    parser.add_argument('--insert_duration', type=int, default=30, help='How many seconds to run insert (default 120)')
    parser.add_argument('--duration', type=int, default=60, help='How many seconds to run (default 360)')
    args = parser.parse_args()

    stop_event = threading.Event()

    # create collection
    milvus_client = MilvusClient(uri=MILVUS_URI, duckdb_dir=DUCKDB_DIR)
    collection_name = f"{COLLECTION_NAME_PREFIX}_{int(time.time())}"
    logging.info(f"Using collection: {collection_name}")

    if milvus_client.has_collection(collection_name):
        logging.warning(f"Collection '{collection_name}' already exists. Dropping it.")
        milvus_client.drop_collection(collection_name)

    schema = milvus_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("name", DataType.VARCHAR, max_length=256)
    schema.add_field("age", DataType.INT64)
    schema.add_field("json_field", DataType.JSON)
    schema.add_field("array_field", DataType.ARRAY, element_type=DataType.INT64, max_capacity=20)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION)
    milvus_client.create_collection(collection_name, schema)
    logging.info(f"Collection '{collection_name}' created successfully.")

    index_params = IndexParams()
    index_params.add_index("embedding", metric_type="L2", index_type="IVF_FLAT", params={"nlist": 128})
    milvus_client.create_index(collection_name, index_params)
    logging.info("Index created successfully.")
    milvus_client.load_collection(collection_name)
    logging.info(f"Collection '{collection_name}' loaded.")
    time.sleep(2)

    # start three worker threads
    insert_thread = threading.Thread(target=insert_worker, 
                                     args=(stop_event, milvus_client, collection_name), 
                                     daemon=True)
    delete_thread = threading.Thread(target=delete_worker, 
                                     args=(stop_event, milvus_client, collection_name), 
                                     daemon=True)
    upsert_thread = threading.Thread(target=upsert_worker, 
                                     args=(stop_event, milvus_client, collection_name), 
                                     daemon=True)
    
    # 启动插入线程
    insert_thread.start()
    logging.info(f"Insert thread started, will run for {args.insert_duration} seconds...")
    
    # 等待指定时间后启动删除和更新线程
    time.sleep(args.insert_duration)
    delete_thread.start()
    upsert_thread.start()
    logging.info("Delete and upsert threads started...")
    
    logging.info(f"Long run checker started, will run for {args.duration} seconds...")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt, exiting early...")
    
    # 设置停止事件，通知所有线程停止
    stop_event.set()
    logging.info("Stop event set, waiting for threads to finish...")
    
    # 等待所有线程结束，不设置超时确保真正等待线程完成
    if insert_thread.is_alive():
        logging.info("Waiting for insert thread to finish...")
        insert_thread.join()
    if delete_thread.is_alive():
        logging.info("Waiting for delete thread to finish...")
        delete_thread.join()
    if upsert_thread.is_alive():
        logging.info("Waiting for upsert thread to finish...")
        upsert_thread.join()
    
    logging.info("All threads finished.")

    logging.info("Long run checker finished.")
    # check collection
    milvus_client.entity_compare(collection_name)
