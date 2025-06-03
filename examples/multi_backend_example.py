#!/usr/bin/env python3
"""
MilvusMultiBackendClient 使用示例

这个示例展示了如何使用 MilvusMultiBackendClient 来支持不同的后端：
1. DuckDB 后端 (本地嵌入式数据库)
2. ClickHouse 后端 (分布式列式数据库)
"""

import json
from pymilvus import CollectionSchema, FieldSchema, DataType
from src.pymilvus_duckdb.milvus_multi_backend_client import MilvusMultiBackendClient
from src.pymilvus_duckdb.milvus_duckdb_client import MilvusDuckDBClient  # 向后兼容

def test_backend(backend_type: str, **backend_config):
    """测试指定后端的功能"""
    print(f"\n{'='*50}")
    print(f"测试 {backend_type.upper()} 后端")
    print(f"{'='*50}")
    
    # 创建客户端
    if backend_type == "duckdb":
        client = MilvusMultiBackendClient(
            backend_type="duckdb",
            uri="http://localhost:19530",
            token="",
            **backend_config
        )
    elif backend_type == "clickhouse":
        client = MilvusMultiBackendClient(
            backend_type="clickhouse",
            uri="http://localhost:19530",
            token="",
            **backend_config
        )
    else:
        raise ValueError(f"Unsupported backend: {backend_type}")
    
    # 打印后端信息
    backend_info = client.get_backend_info()
    print(f"后端信息: {backend_info}")
    
    # 定义集合 schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="score", dtype=DataType.FLOAT),
        FieldSchema(name="is_active", dtype=DataType.BOOL),
        FieldSchema(name="created_at", dtype=DataType.INT64)
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description=f"Test collection with {backend_type} backend"
    )
    
    collection_name = f"test_{backend_type}_collection"
    
    try:
        # 创建集合
        print(f"正在创建集合: {collection_name}")
        client.create_collection(collection_name, schema)
        print("✅ 集合创建成功!")
        
        # 准备测试数据
        data = [
            {
                "id": 1,
                "title": "Python 编程入门",
                "content": "这是一篇关于 Python 编程基础知识的文章...",
                "category": "技术",
                "tags": ["Python", "编程", "入门"],
                "metadata": {"author": "张三", "difficulty": "初级", "rating": 4.5},
                "embedding": [0.1] * 384,
                "score": 85.5,
                "is_active": True,
                "created_at": 1699520000
            },
            {
                "id": 2,
                "title": "机器学习应用",
                "content": "机器学习在实际项目中的应用案例...",
                "category": "技术",
                "tags": ["机器学习", "AI", "应用"],
                "metadata": {"author": "李四", "difficulty": "中级", "rating": 4.8},
                "embedding": [0.2] * 384,
                "score": 92.3,
                "is_active": True,
                "created_at": 1699520100
            },
            {
                "id": 3,
                "title": "数据科学导论",
                "content": "数据科学的基础理论和实践方法...",
                "category": "技术",
                "tags": ["数据科学", "统计", "分析"],
                "metadata": {"author": "王五", "difficulty": "高级", "rating": 4.7},
                "embedding": [0.3] * 384,
                "score": 88.9,
                "is_active": False,
                "created_at": 1699520200
            }
        ]
        
        # 插入数据
        print(f"正在插入 {len(data)} 条记录...")
        result = client.insert(collection_name, data)
        print(f"✅ 插入成功，插入了 {result['insert_count']} 条记录")
        
        # 查询数据并比较
        print("\n查询数据并比较 Milvus 和后端...")
        filter_expr = "score > 85"
        milvus_res, backend_res = client.query(
            collection_name, 
            filter=filter_expr, 
            output_fields=["id", "title", "score", "is_active"]
        )
        
        print(f"Milvus 查询结果 ({len(milvus_res)} 条):")
        print(milvus_res)
        print(f"\n{backend_type.upper()} 查询结果 ({len(backend_res)} 条):")
        print(backend_res)
        
        # 数据一致性检查
        print("\n执行数据一致性检查...")
        diff = client.query_result_compare(
            collection_name,
            filter=filter_expr,
            output_fields=["id", "title", "score", "is_active"]
        )
        
        if not diff:
            print("✅ 数据一致性检查通过!")
        else:
            print("❌ 发现数据不一致:", diff)
        
        # 统计记录数
        count_result = client.count(collection_name)
        print(f"\n记录数统计:")
        print(f"Milvus: {count_result['milvus_count']}")
        print(f"{backend_type.upper()}: {count_result[f'{backend_type}_count']}")
        
        # 更新数据
        print("\n更新数据...")
        update_data = [
            {
                "id": 1,
                "title": "Python 编程入门 (更新版)",
                "content": "这是一篇更新的 Python 编程基础知识文章...",
                "category": "技术",
                "tags": ["Python", "编程", "入门", "更新"],
                "metadata": {"author": "张三", "difficulty": "初级", "rating": 4.6},
                "embedding": [0.15] * 384,
                "score": 87.0,
                "is_active": True,
                "created_at": 1699520000
            }
        ]
        
        client.upsert(collection_name, update_data)
        print("✅ 数据更新成功!")
        
        # 删除数据
        print("\n删除一条记录...")
        client.delete(collection_name, [3])
        print("✅ 删除成功!")
        
        # 再次检查计数
        final_count = client.count(collection_name)
        print(f"\n最终记录数:")
        print(f"Milvus: {final_count['milvus_count']}")
        print(f"{backend_type.upper()}: {final_count[f'{backend_type}_count']}")
        
        # 导出后端数据
        print(f"\n导出 {backend_type.upper()} 数据...")
        exported_data = client.export(collection_name)
        print(f"导出的数据 ({len(exported_data)} 条):")
        print(exported_data[["id", "title", "score"]].head())
        
        # 数据采样
        print(f"\n从 {backend_type.upper()} 采样数据...")
        sampled_data = client.sample_data(collection_name, num_samples=2)
        print(f"采样数据 ({len(sampled_data)} 条):")
        print(sampled_data[["id", "title", "score"]])
        
        # 生成过滤器表达式
        print(f"\n生成 Milvus 过滤器表达式...")
        filter_exprs = client.generate_milvus_filter(collection_name, num_samples=10)
        print(f"生成了 {len(filter_exprs)} 个过滤器表达式，示例:")
        for expr in filter_exprs[:5]:
            print(f"  - {expr}")
        
        # 完整实体比较
        print(f"\n执行完整实体比较...")
        comparison_result = client.entity_compare(collection_name, batch_size=100)
        if comparison_result:
            print("✅ 实体比较完成，数据一致!")
        else:
            print("❌ 实体比较发现不一致!")
            
        print(f"\n✅ {backend_type.upper()} 后端测试完成!")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        try:
            print(f"\n清理集合: {collection_name}")
            client.drop_collection(collection_name)
            print("✅ 清理完成!")
        except Exception as e:
            print(f"❌ 清理时发生错误: {e}")


def test_backward_compatibility():
    """测试向后兼容性"""
    print(f"\n{'='*50}")
    print("测试向后兼容性 (原 MilvusDuckDBClient)")
    print(f"{'='*50}")
    
    # 使用原有的 MilvusDuckDBClient API
    client = MilvusDuckDBClient(
        uri="http://localhost:19530",
        token="",
        duckdb_dir="./data/duckdb"
    )
    
    print("✅ 原有的 MilvusDuckDBClient API 仍然可用!")
    print(f"后端信息: {client.get_backend_info()}")


def main():
    """主函数"""
    print("MilvusMultiBackendClient 多后端支持演示")
    print("支持的后端: DuckDB (嵌入式), ClickHouse (分布式)")
    
    # 测试 DuckDB 后端
    test_backend("duckdb", duckdb_dir="./data/duckdb")
    
    # 测试 ClickHouse 后端 (需要 ClickHouse 服务器运行)
    try:
        test_backend("clickhouse", 
                    clickhouse_host="localhost",
                    clickhouse_port=8123,
                    clickhouse_username="default",
                    clickhouse_password="",
                    clickhouse_database="default")
    except Exception as e:
        print(f"\n⚠️  ClickHouse 后端测试跳过 (可能服务器未运行): {e}")
    
    # 测试向后兼容性
    test_backward_compatibility()
    
    print(f"\n{'='*50}")
    print("所有测试完成!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 