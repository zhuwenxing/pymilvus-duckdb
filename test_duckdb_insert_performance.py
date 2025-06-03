import time
import duckdb
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path


class DuckDBInsertTester:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Create a test database in memory or temporary location
            test_dir = "./data/test"
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            db_path = f"{test_dir}/performance_test_{uuid.uuid4()}.db"
        
        print(f"初始化 DuckDB 连接: {db_path}")
        self.duck_conn = duckdb.connect(db_path)
        self.db_path = db_path
        
    def __del__(self):
        try:
            self.duck_conn.close()
        except Exception as e:
            print(f"关闭 DuckDB 连接失败: {e}")
    
    def create_test_table(self, table_name: str = "test_performance"):
        """Create test table similar to the structure in milvus_duckdb_client"""
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            age INTEGER,
            score FLOAT,
            metadata JSON,
            tags VARCHAR[],
            embedding FLOAT[128]
        );
        """
        print(f"创建测试表: {table_name}")
        self.duck_conn.execute(create_sql)
        
    def generate_test_data(self, num_records: int) -> list[dict]:
        """Generate test data similar to the format used in milvus_duckdb_client"""
        print(f"生成 {num_records} 条测试数据")
        data = []
        for i in range(num_records):
            record = {
                "id": i,
                "name": f"test_user_{i}",
                "age": np.random.randint(18, 80),
                "score": np.random.random() * 100,
                "metadata": {"category": f"cat_{i % 5}", "level": i % 10},
                "tags": [f"tag_{i % 3}", f"tag_{(i+1) % 3}"],
                "embedding": np.random.random(128).tolist()
            }
            data.append(record)
        return data
    
    def insert_without_transaction(self, table_name: str, data: list[dict]):
        """Insert data without transaction, similar to milvus_duckdb_client logic"""
        start_time = time.time()
        
        # Convert data to DataFrame and handle JSON fields like in the original code
        df = pd.DataFrame(data)
        df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x))
        
        # Register DataFrame with DuckDB
        self.duck_conn.register("df", df)
        
        # Execute insert without transaction
        insert_sql = f"INSERT INTO {table_name} SELECT * FROM df"
        self.duck_conn.execute(insert_sql)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"单次批量 insert (无事务) 耗时: {duration:.4f} 秒，插入 {len(data)} 条记录")
        return duration
    
    def insert_with_transaction(self, table_name: str, data: list[dict]):
        """Insert data with transaction for comparison"""
        start_time = time.time()
        
        # Convert data to DataFrame and handle JSON fields
        df = pd.DataFrame(data)
        df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x))
        
        # Register DataFrame with DuckDB
        self.duck_conn.register("df", df)
        
        try:
            # Execute insert with transaction
            self.duck_conn.execute("BEGIN TRANSACTION;")
            insert_sql = f"INSERT INTO {table_name} SELECT * FROM df"
            self.duck_conn.execute(insert_sql)
            self.duck_conn.execute("COMMIT;")
        except Exception as e:
            self.duck_conn.execute("ROLLBACK;")
            raise e
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"单次批量 insert (有事务) 耗时: {duration:.4f} 秒，插入 {len(data)} 条记录")
        return duration
    
    def multiple_inserts_test(self, table_name: str, batch_size: int, num_batches: int, use_transaction: bool = False):
        """Test multiple insert operations"""
        print(f"\n开始测试多次 insert 性能:")
        print(f"  批次大小: {batch_size}")
        print(f"  批次数量: {num_batches}")
        print(f"  总记录数: {batch_size * num_batches}")
        print(f"  使用事务: {use_transaction}")
        print("-" * 50)
        
        total_start_time = time.time()
        durations = []
        
        for batch_idx in range(num_batches):
            print(f"执行第 {batch_idx + 1}/{num_batches} 批次...")
            
            # Generate data for this batch
            data = self.generate_test_data_batch(batch_idx * batch_size, batch_size)
            
            # Insert data
            if use_transaction:
                duration = self.insert_with_transaction(table_name, data)
            else:
                duration = self.insert_without_transaction(table_name, data)
            
            durations.append(duration)
            
            # Progress reporting
            if (batch_idx + 1) % max(1, num_batches // 4) == 0:
                progress = (batch_idx + 1) * 100 // num_batches
                print(f"进度: {progress}% ({batch_idx + 1}/{num_batches} 批次完成)")
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Performance statistics
        avg_duration = np.mean(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        total_records = batch_size * num_batches
        
        print("\n" + "=" * 60)
        print("性能测试结果:")
        print("=" * 60)
        print(f"总耗时: {total_duration:.4f} 秒")
        print(f"平均每批次耗时: {avg_duration:.4f} 秒")
        print(f"最快批次耗时: {min_duration:.4f} 秒")  
        print(f"最慢批次耗时: {max_duration:.4f} 秒")
        print(f"总记录数: {total_records}")
        print(f"平均每秒插入记录数: {total_records / total_duration:.2f} 条/秒")
        print(f"使用事务: {use_transaction}")
        
        return {
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "total_records": total_records,
            "records_per_second": total_records / total_duration,
            "use_transaction": use_transaction
        }
    
    def generate_test_data_batch(self, start_id: int, batch_size: int) -> list[dict]:
        """Generate a batch of test data with sequential IDs"""
        data = []
        for i in range(start_id, start_id + batch_size):
            record = {
                "id": i,
                "name": f"test_user_{i}",
                "age": np.random.randint(18, 80),
                "score": np.random.random() * 100,
                "metadata": {"category": f"cat_{i % 5}", "level": i % 10},
                "tags": [f"tag_{i % 3}", f"tag_{(i+1) % 3}"],
                "embedding": np.random.random(128).tolist()
            }
            data.append(record)
        return data
    
    def clear_table(self, table_name: str):
        """Clear all data in the table"""
        self.duck_conn.execute(f"DELETE FROM {table_name}")
        print(f"清空表 {table_name}")
    
    def get_record_count(self, table_name: str) -> int:
        """Get the number of records in the table"""
        result = self.duck_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return result[0] if result else 0


def main():
    """Main test function"""
    print("DuckDB Insert 性能测试")
    print("=" * 60)
    
    # Initialize tester
    tester = DuckDBInsertTester()
    table_name = "performance_test"
    
    # Create test table
    tester.create_test_table(table_name)
    
    # Test configurations
    test_configs = [
        {"batch_size": 100, "num_batches": 10},    # 1,000 records
        {"batch_size": 500, "num_batches": 10},    # 5,000 records  
        {"batch_size": 1000, "num_batches": 10},   # 10,000 records
        {"batch_size": 1000, "num_batches": 20},   # 20,000 records
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置: 批次大小={config['batch_size']}, 批次数量={config['num_batches']}")
        
        # Clear table before each test
        tester.clear_table(table_name)
        
        # Test without transaction
        print(f"\n测试 1: 不使用事务")
        result_no_tx = tester.multiple_inserts_test(
            table_name, 
            config["batch_size"], 
            config["num_batches"], 
            use_transaction=False
        )
        result_no_tx["config"] = config.copy()
        results.append(result_no_tx)
        
        # Verify record count
        count = tester.get_record_count(table_name)
        expected_count = config["batch_size"] * config["num_batches"]
        print(f"验证: 表中记录数 = {count}, 预期 = {expected_count}")
        
        # Clear table for next test
        tester.clear_table(table_name)
        
        # Test with transaction for comparison
        print(f"\n测试 2: 使用事务")
        result_with_tx = tester.multiple_inserts_test(
            table_name, 
            config["batch_size"], 
            config["num_batches"], 
            use_transaction=True
        )
        result_with_tx["config"] = config.copy()
        results.append(result_with_tx)
        
        # Verify record count
        count = tester.get_record_count(table_name)
        print(f"验证: 表中记录数 = {count}, 预期 = {expected_count}")
        
        # Performance comparison
        speedup = result_with_tx["total_duration"] / result_no_tx["total_duration"]
        print(f"\n性能对比:")
        print(f"  无事务用时: {result_no_tx['total_duration']:.4f}s")
        print(f"  有事务用时: {result_with_tx['total_duration']:.4f}s") 
        print(f"  性能倍数: {speedup:.2f}x ({'无事务更快' if speedup > 1 else '有事务更快'})")
    
    # Summary
    print(f"\n{'='*60}")
    print("测试总结")
    print("=" * 60)
    
    for i, result in enumerate(results):
        config = result["config"]
        tx_status = "有事务" if result["use_transaction"] else "无事务"
        print(f"配置 {i//2 + 1} ({tx_status}): "
              f"批次大小={config['batch_size']}, "
              f"批次数量={config['num_batches']}, "
              f"总耗时={result['total_duration']:.4f}s, "
              f"速度={result['records_per_second']:.2f}条/秒")


if __name__ == "__main__":
    main() 