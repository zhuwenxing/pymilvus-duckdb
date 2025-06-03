from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Union


class BaseBackend(ABC):
    """
    Abstract base class for database backends.
    Defines the interface that all backends must implement.
    """
    
    def __init__(self, **kwargs):
        self.fields = []
        self.primary_field = ""
        self.fields_name_list = []
        self.json_fields = []
        self.array_fields = []
        self.varchar_fields = []
        self.float_vector_fields = []
    
    @abstractmethod
    def connect(self, **kwargs) -> None:
        """Establish connection to the backend database."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connection to the backend database."""
        pass
    
    @abstractmethod
    def create_table(self, table_name: str, schema_fields: List[str]) -> None:
        """Create a table with the given schema."""
        pass
    
    @abstractmethod
    def drop_table(self, table_name: str) -> None:
        """Drop a table."""
        pass
    
    @abstractmethod
    def insert_data(self, table_name: str, data: pd.DataFrame) -> None:
        """Insert data into the table."""
        pass
    
    @abstractmethod
    def delete_data(self, table_name: str, primary_key_values: List[Union[int, str]]) -> pd.DataFrame:
        """Delete data from the table and return deleted data for rollback."""
        pass
    
    @abstractmethod
    def upsert_data(self, table_name: str, data: pd.DataFrame, primary_key: str) -> pd.DataFrame:
        """Upsert data into the table and return original data for rollback."""
        pass
    
    @abstractmethod
    def query_data(self, table_name: str, filter_expr: str, output_fields: List[str]) -> pd.DataFrame:
        """Query data from the table."""
        pass
    
    @abstractmethod
    def count_records(self, table_name: str) -> int:
        """Count total records in the table."""
        pass
    
    @abstractmethod
    def export_data(self, table_name: str) -> pd.DataFrame:
        """Export all data from the table."""
        pass
    
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        pass
    
    @abstractmethod
    def sample_data(self, table_name: str, num_samples: int) -> pd.DataFrame:
        """Sample data from the table."""
        pass
    
    @abstractmethod
    def convert_milvus_filter_to_sql(self, filter_expr: str) -> str:
        """Convert Milvus filter expression to backend-specific SQL."""
        pass
    
    @abstractmethod
    def get_backend_type(self) -> str:
        """Return the backend type name."""
        pass 