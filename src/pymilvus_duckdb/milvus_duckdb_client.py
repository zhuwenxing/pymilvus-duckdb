"""
Legacy DuckDB client for backward compatibility.
This class inherits from MilvusMultiBackendClient with backend_type="duckdb".
"""

from .milvus_multi_backend_client import MilvusMultiBackendClient


class MilvusDuckDBClient(MilvusMultiBackendClient):
    """
    Legacy MilvusDuckDBClient for backward compatibility.
    
    This class is now a thin wrapper around MilvusMultiBackendClient
    with backend_type fixed to "duckdb".
    
    All original APIs are preserved for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        # Force backend_type to duckdb for backward compatibility
        kwargs['backend_type'] = 'duckdb'
        super().__init__(*args, **kwargs)
    
    # All other methods are inherited from MilvusMultiBackendClient
    # The original API is preserved for backward compatibility 