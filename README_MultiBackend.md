# Milvus 多后端客户端

这是一个支持多种后端数据库的 Milvus 客户端，用于在 Milvus 向量数据库和传统数据库之间进行数据同步。

## 🚀 特性

### 支持的后端
- **DuckDB**: 高性能嵌入式分析数据库
- **ClickHouse**: 分布式列式数据库，适合大规模数据分析

### 核心功能
- ✅ **多后端支持**: 可配置使用 DuckDB 或 ClickHouse 作为后端
- ✅ **数据一致性**: 确保 Milvus 和后端数据库数据同步
- ✅ **事务性操作**: 支持原子性操作和自动回滚
- ✅ **向后兼容**: 保持原有 API 不变
- ✅ **灵活部署**: 支持本地嵌入式和分布式部署
- ✅ **性能优化**: 针对不同后端的性能优化

## 📋 架构设计

```
MilvusMultiBackendClient
    ├── BaseBackend (抽象接口)
    │   ├── DuckDBBackend (DuckDB 实现)
    │   └── ClickHouseBackend (ClickHouse 实现)
    └── MilvusDuckDBClient (向后兼容)
```

### 后端选择指南

| 特性 | DuckDB | ClickHouse |
|------|--------|------------|
| **部署方式** | 嵌入式，无需独立服务 | 需要独立服务器 |
| **数据规模** | 中小规模 (GB级) | 大规模 (TB/PB级) |
| **查询性能** | 单机高性能 | 分布式高并发 |
| **运维复杂度** | 简单 | 中等 |
| **事务支持** | 完整 ACID | 有限支持 |
| **适用场景** | 开发测试、小型项目 | 生产环境、大数据分析 |

## 🛠 安装

### 基础依赖
```bash
pip install pymilvus pandas numpy deepdiff
```

### DuckDB 后端
```bash
pip install duckdb
```

### ClickHouse 后端
```bash
pip install clickhouse-connect
```

## 📖 使用方法

### 1. 基本使用

#### 使用 DuckDB 后端
```python
from src.pymilvus_duckdb import MilvusMultiBackendClient

client = MilvusMultiBackendClient(
    backend_type="duckdb",
    uri="http://localhost:19530",
    token="",
    duckdb_dir="./data/duckdb"
)
```

#### 使用 ClickHouse 后端
```python
from src.pymilvus_duckdb import MilvusMultiBackendClient

client = MilvusMultiBackendClient(
    backend_type="clickhouse",
    uri="http://localhost:19530",
    token="",
    clickhouse_host="localhost",
    clickhouse_port=8123,
    clickhouse_username="default",
    clickhouse_password="",
    clickhouse_database="default"
)
```

#### 向后兼容方式 (原 DuckDB 客户端)
```python
from src.pymilvus_duckdb import MilvusDuckDBClient

# 原有代码无需修改
client = MilvusDuckDBClient(
    uri="http://localhost:19530",
    token="",
    duckdb_dir="./data/duckdb"
)
```

### 2. 创建集合

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10),
    FieldSchema(name="score", dtype=DataType.FLOAT),
    FieldSchema(name="is_active", dtype=DataType.BOOL)
]

schema = CollectionSchema(fields=fields, description="示例集合")
client.create_collection("my_collection", schema)
```

### 3. 数据操作

```python
# 插入数据
data = [
    {
        "id": 1,
        "title": "示例文档",
        "embedding": [0.1] * 768,
        "metadata": {"author": "张三", "category": "技术"},
        "tags": ["AI", "机器学习"],
        "score": 95.5,
        "is_active": True
    }
]
client.insert("my_collection", data)

# 查询数据 (返回 Milvus 和后端两个结果)
milvus_result, backend_result = client.query(
    "my_collection", 
    filter="score > 90", 
    output_fields=["id", "title", "score"]
)

# 更新数据
update_data = [{"id": 1, "title": "更新的文档", "score": 98.0, ...}]
client.upsert("my_collection", update_data)

# 删除数据
client.delete("my_collection", [1])

# 统计记录数
count_result = client.count("my_collection")
print(f"Milvus: {count_result['milvus_count']}")
print(f"Backend: {count_result['duckdb_count']}")  # 或 clickhouse_count
```

### 4. 数据一致性检查

```python
# 查询结果比较
diff = client.query_result_compare(
    "my_collection",
    filter="score > 90",
    output_fields=["id", "title", "score"]
)

# 完整实体比较
client.entity_compare("my_collection", batch_size=1000)
```

### 5. 数据分析功能

```python
# 导出后端数据进行分析
df = client.export("my_collection")

# 数据采样
sample_df = client.sample_data("my_collection", num_samples=100)

# 生成过滤器表达式
filter_expressions = client.generate_milvus_filter("my_collection")
```

### 6. 获取后端信息

```python
backend_info = client.get_backend_info()
print(f"后端类型: {backend_info['backend_type']}")
print(f"后端类: {backend_info['backend_class']}")
print(f"支持的操作: {backend_info['supported_operations']}")
```

## 🔧 配置说明

### DuckDB 配置参数
```python
duckdb_config = {
    "duckdb_dir": "./data/duckdb",  # DuckDB 数据目录
    "host": "localhost"             # 主机名 (用于目录命名)
}
```

### ClickHouse 配置参数
```python
clickhouse_config = {
    "clickhouse_host": "localhost",      # ClickHouse 服务器地址
    "clickhouse_port": 8123,            # HTTP 端口
    "clickhouse_username": "default",   # 用户名
    "clickhouse_password": "",          # 密码
    "clickhouse_database": "default"    # 数据库名
}
```

## 📊 性能对比

### DuckDB 后端
- **优势**: 无需独立服务，部署简单，单机性能优秀
- **劣势**: 不支持分布式，数据规模受限
- **适用**: 开发测试、中小型应用

### ClickHouse 后端
- **优势**: 分布式架构，大数据处理能力强，查询性能优秀
- **劣势**: 需要独立部署，运维复杂度较高
- **适用**: 生产环境、大规模数据分析

## 🧪 测试和示例

### 运行示例
```bash
# 运行多后端示例
python examples/multi_backend_example.py

# 运行原有的 ClickHouse 示例 (如果需要)
python examples/clickhouse_client_example.py
```

### 启动 ClickHouse 服务 (Docker)
```bash
docker run -d --name clickhouse-server \
  -p 8123:8123 -p 9000:9000 \
  clickhouse/clickhouse-server
```

## 🔍 故障排除

### 常见问题

1. **ClickHouse 连接失败**
   - 检查服务器是否运行: `curl http://localhost:8123/ping`
   - 验证用户权限和密码
   - 检查防火墙设置

2. **DuckDB 文件权限问题**
   - 确保数据目录有写权限
   - 检查磁盘空间

3. **数据类型不匹配**
   - 检查 schema 定义
   - 确认向量维度设置
   - 验证 JSON 数据格式

4. **性能问题**
   - 调整批次大小 (batch_size)
   - 优化查询条件
   - 检查网络延迟 (ClickHouse)

### 调试技巧

```python
# 启用详细日志
from src.pymilvus_duckdb import set_logger_level
import logging
set_logger_level(logging.DEBUG)

# 检查后端连接状态
try:
    client.count("test_collection")
    print("后端连接正常")
except Exception as e:
    print(f"后端连接异常: {e}")
```

## 🤝 贡献指南

### 添加新后端

1. 继承 `BaseBackend` 类
2. 实现所有抽象方法
3. 在 `MilvusMultiBackendClient._create_backend()` 中注册
4. 添加相应的依赖项
5. 编写测试用例

```python
from .backends.base_backend import BaseBackend

class MyNewBackend(BaseBackend):
    def connect(self, **kwargs):
        # 实现连接逻辑
        pass
    
    def insert_data(self, table_name, data):
        # 实现插入逻辑
        pass
    
    # ... 实现其他方法
```

### 代码规范
- 使用中文注释和日志
- 英文代码注释
- 遵循 PEP 8 编码规范
- 添加类型提示

## 📄 许可证

本项目遵循原有项目的许可证。

## 🙏 致谢

- Milvus 社区提供的向量数据库
- DuckDB 项目提供的高性能分析数据库
- ClickHouse 项目提供的分布式列式数据库 