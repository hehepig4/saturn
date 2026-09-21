"""
Base Table Manager for LanceDB - v2.0

Provides common functionality for all table managers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import lancedb
import pyarrow as pa

class BaseTableManager(ABC):
    """
    Base class for all table managers.
    
    Provides typed CRUD operations and SQL query interface.
    """
    
    def __init__(self, db: lancedb.DBConnection, table_name: str):
        """
        Initialize table manager.
        
        Args:
            db: LanceDB database connection
            table_name: Name of the table to manage
        """
        self.db = db
        self.table_name = table_name
        self._table: Optional[lancedb.table.Table] = None
        
    @property
    def table(self) -> lancedb.table.Table:
        """
        Get or create the managed table.
        
        Returns:
            LanceDB table instance
        """
        if self._table is None:
            # Try to open existing table first
            try:
                self._table = self.db.open_table(self.table_name)
            except Exception:
                # Table doesn't exist, create it
                schema = self.get_schema()
                try:
                    self._table = self.db.create_table(
                        self.table_name,
                        schema=schema,
                        mode="create"
                    )
                    # Create indices after table creation
                    self._create_indices()
                except ValueError as e:
                    # Handle race condition: table was created by another instance
                    if "already exists" in str(e):
                        self._table = self.db.open_table(self.table_name)
                    else:
                        raise
        return self._table
    
    @abstractmethod
    def get_schema(self) -> pa.Schema:
        """
        Define the PyArrow schema for this table.
        
        Returns:
            PyArrow schema defining table structure
        """
        pass
    
    def _create_indices(self) -> None:
        """
        Create indices for the table.
        Override in subclasses to add specific indices.
        """
        pass
    
    def insert(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Insert one or more records into the table.
        
        Args:
            data: Single record dict or list of record dicts
        """
        if isinstance(data, dict):
            data = [data]
        
        # Convert PyArrow arrays to Python native types to avoid schema mismatch errors
        sanitized_data = []
        for record in data:
            sanitized_record = {}
            for key, value in record.items():
                # Convert PyArrow arrays and scalars to Python native types
                if isinstance(value, (pa.Array, pa.ChunkedArray)):
                    sanitized_record[key] = value.to_pylist()
                elif isinstance(value, pa.Scalar):
                    sanitized_record[key] = value.as_py()
                # Ensure lists stay as lists (don't convert them)
                elif isinstance(value, list):
                    sanitized_record[key] = value
                else:
                    sanitized_record[key] = value
            sanitized_data.append(sanitized_record)
        
        self.table.add(sanitized_data)
    
    def get_by_id(self, id_field: str, id_value: Any) -> Optional[Dict[str, Any]]:
        """
        Get a single record by ID field.
        
        Args:
            id_field: Name of the ID field
            id_value: Value to search for
            
        Returns:
            Record dict if found, None otherwise
        """
        results = self.table.search() \
            .where(f"{id_field} = '{id_value}'") \
            .limit(1) \
            .to_pandas()
        
        if len(results) > 0:
            return results.iloc[0].to_dict()
        return None
    
    def delete(self, id_field: str, id_value: Any) -> bool:
        """
        Delete a record by ID.
        
        Args:
            id_field: Name of the ID field
            id_value: Value to search for
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.table.delete(f"{id_field} = '{id_value}'")
            return True
        except Exception:
            return False
    
    def query(self, filter_expr: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query records with optional filter.
        
        Args:
            filter_expr: SQL-like filter expression
            limit: Maximum number of records to return
            
        Returns:
            List of matching records
        """
        query = self.table.search()
        
        if filter_expr:
            query = query.where(filter_expr)
        
        if limit:
            query = query.limit(limit)
        
        results = query.to_pandas()
        return results.to_dict('records')
    
    def count(self, filter_expr: Optional[str] = None) -> int:
        """
        Count records matching filter.
        
        Args:
            filter_expr: SQL-like filter expression
            
        Returns:
            Number of matching records
        """
        return len(self.query(filter_expr=filter_expr))
    
    def exists(self, id_field: str, id_value: Any) -> bool:
        """
        Check if a record exists.
        
        Args:
            id_field: Name of the ID field
            id_value: Value to search for
            
        Returns:
            True if record exists, False otherwise
        """
        return self.get_by_id(id_field, id_value) is not None
    
    def batch_insert(self, records: List[Dict[str, Any]], batch_size: int = 1000) -> None:
        """
        Insert records in batches for better performance.
        
        Args:
            records: List of record dicts
            batch_size: Number of records per batch
        """
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.insert(batch)
    
    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all records from the table.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of all records
        """
        return self.query(limit=limit)
    
    def clear(self) -> None:
        """Clear all records from the table."""
        # Drop and recreate table
        if self.table_name in self.db.table_names(limit=1000000):
            self.db.drop_table(self.table_name)
        self._table = None
        # Accessing .table property will recreate it
        _ = self.table
    
    def drop_table(self) -> None:
        """Drop the table completely."""
        if self.table_name in self.db.table_names(limit=1000000):
            self.db.drop_table(self.table_name)
        self._table = None
    
    # ========== SQL Query Interface ==========
    
    def sql_query(self, sql: str) -> pa.Table:
        """
        Execute SQL query on table.
        
        Args:
            sql: SQL query - WHERE clause or full SELECT statement
        
        Returns:
            Query results as PyArrow table
        
        Examples:
            results = mgr.sql_query("modality = 'IMAGE'")
            results = mgr.sql_query("SELECT namespace, COUNT(*) FROM table GROUP BY namespace")
        """
        sql_upper = sql.strip().upper()
        
        if sql_upper.startswith("SELECT"):
            try:
                import duckdb
                arrow_table = self.table.to_lance()
                sql_query = sql.replace("FROM table", "FROM arrow_table") \
                              .replace(f"FROM {self.table_name}", "FROM arrow_table")
                results = duckdb.query(sql_query).to_arrow_table()
            except ImportError:
                raise ImportError("DuckDB required for full SQL queries. Install: pip install duckdb")
        else:
            query_builder = self.table.search()
            results = query_builder.where(sql).to_arrow()
        
        return results
