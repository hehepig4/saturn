"""
Unified Store Manager - v3.0
SQL-based access for LanceDB.
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import lancedb
import pyarrow as pa
from datetime import datetime

from core.paths import get_db_path
from .ontology.ontology_manager import OntologyManager


class StoreManager:
    """Unified store manager with SQL interface."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(get_db_path())
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self._db: Optional[lancedb.DBConnection] = None
        self._connect()
        
        self._ontology_manager: Optional[OntologyManager] = None
    
    def _connect(self) -> None:
        self._db = lancedb.connect(str(self.db_path))
    
    @property
    def db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._connect()
        return self._db
    
    @property
    def ontology_manager(self) -> OntologyManager:
        """Ontology manager for complete TBox storage and retrieval."""
        if self._ontology_manager is None:
            self._ontology_manager = OntologyManager(self.db)
        return self._ontology_manager
    
    # ========== SQL Query Interface ==========
    
    def query(
        self,
        sql: str,
        table: Optional[str] = None,
    ) -> pa.Table:
        """
        Run SQL query on a table.
        
        Args:
            sql: SQL query (WHERE clause or full SELECT)
            table: Table name (required if sql is WHERE clause)
        
        Returns:
            PyArrow Table with results
        """
        if table is None:
            table = self._extract_table_from_sql(sql)
        
        if table not in self.db.table_names(limit=1000000):
            raise ValueError(f"Table '{table}' not found")
        
        tbl = self.db.open_table(table)
        
        # If sql is just a WHERE clause, convert to full query
        if "SELECT" not in sql.upper():
            sql = f"SELECT * FROM {table} WHERE {sql}"
        
        return tbl.to_lance().to_table(filter=None)
    
    def _extract_table_from_sql(self, sql: str) -> Optional[str]:
        """Extract table name from SQL."""
        sql_upper = sql.strip().upper()
        
        if "FROM" in sql_upper:
            parts = sql_upper.split("FROM")
            if len(parts) > 1:
                table_part = parts[1].split()[0].strip()
                return table_part.lower()
        
        return None
    
    # ========== Insert/Update/Delete Operations ==========
    
    def insert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        create_if_missing: bool = True
    ) -> int:
        """
        Insert data into table.
        
        Args:
            table: Table name
            data: Single record or list of records
            create_if_missing: Create table if it doesn't exist
        
        Returns:
            Number of records inserted
        """
        if isinstance(data, dict):
            data = [data]
        
        if not data:
            return 0
        
        if table not in self.db.table_names(limit=1000000):
            if create_if_missing:
                tbl = self.db.create_table(table, data)
            else:
                raise ValueError(f"Table '{table}' not found")
        else:
            tbl = self.db.open_table(table)
            tbl.add(data)
        
        return len(data)
    
    def update(
        self,
        table: str,
        updates: Dict[str, Any],
        where: str
    ) -> int:
        """Update records matching condition."""
        if table not in self.db.table_names(limit=1000000):
            return 0
        
        tbl = self.db.open_table(table)
        tbl.update(where=where, values=updates)
        return 1  # LanceDB 不返回更新数，假设成功
    
    def delete(self, table: str, where: str) -> bool:
        """Delete records matching condition."""
        if table not in self.db.table_names(limit=1000000):
            return False
        
        tbl = self.db.open_table(table)
        tbl.delete(where)
        return True
    
    # ========== Utility Methods ==========
    
    def list_tables(self) -> List[str]:
        """List all tables."""
        return self.db.table_names(limit=1000000)
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        return table_name in self.db.table_names(limit=1000000)
    
    def table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table information."""
        if table_name not in self.db.table_names(limit=1000000):
            return None
        
        table = self.db.open_table(table_name)
        
        return {
            "name": table_name,
            "schema": table.schema,
            "count": table.count_rows(),
        }
    
    def drop_table(self, table_name: str) -> bool:
        """Drop a table."""
        if table_name not in self.db.table_names(limit=1000000):
            return False
        
        self.db.drop_table(table_name)
        return True
    
    def close(self) -> None:
        """Close database connection."""
        pass
    
    # ========== Context Manager Support ==========
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
