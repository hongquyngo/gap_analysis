# utils/net_gap/data_loader.py

"""
Data Loader for GAP Analysis - Version 3.2 ENHANCED
- Added exclusion support for products and brands
- Added exclude expired inventory option
- Enhanced entity display with company_code
- Enhanced product display with package_size
"""

import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging
import re
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

import os
import sys
from pathlib import Path
project_root = os.environ.get('PROJECT_ROOT', Path(__file__).parent.parent.parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils.db import get_db_engine

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL_DATA = 300
CACHE_TTL_REFERENCE = 600
CACHE_TTL_SAFETY = 900

# Validation constants
MAX_ENTITY_NAME_LENGTH = 200
MAX_PRODUCT_IDS = 1000
MAX_BRANDS = 100


class DataLoadError(Exception):
    """Base exception for data loading errors"""
    pass


class ValidationError(DataLoadError):
    """Exception for input validation failures"""
    pass


class DatabaseConnectionError(DataLoadError):
    """Exception for database connection issues"""
    pass


class GAPDataLoader:
    """Handles all data loading operations for GAP analysis with exclusion support"""
    
    def __init__(self):
        self._engine = None
        self._safety_stock_available = None
        self._entity_id_cache = {}
    
    @property
    def engine(self):
        """Lazy load database engine with connection validation"""
        if self._engine is None:
            try:
                self._engine = get_db_engine()
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Database connection established successfully")
            except Exception as e:
                logger.error(f"Failed to establish database connection: {e}", exc_info=True)
                raise DatabaseConnectionError(f"Cannot connect to database: {str(e)}")
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            raise DatabaseConnectionError(f"Database connection failed: {str(e)}")
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    
    def _normalize_text_field(self, value: Any, field_name: str = '') -> str:
        """Normalize text field for consistency"""
        if pd.isna(value) or value is None:
            return ''
        
        str_value = str(value).strip()
        
        if field_name == 'pt_code':
            return str_value
        elif field_name in ['brand', 'product_name', 'standard_uom']:
            return str_value.upper()
        else:
            return str_value
    
    # Input Validation
    def _validate_entity_name(self, entity_name: Optional[str]) -> None:
        """Validate entity name input"""
        if entity_name is None:
            return
        
        if not isinstance(entity_name, str):
            raise ValidationError(f"Entity name must be string, got {type(entity_name)}")
        
        if len(entity_name) == 0:
            raise ValidationError("Entity name cannot be empty")
        
        if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
            raise ValidationError(f"Entity name too long (max {MAX_ENTITY_NAME_LENGTH} chars)")
        
        if re.search(r'[;\'"\\]', entity_name):
            raise ValidationError("Entity name contains invalid characters")
    
    def _validate_product_ids(self, product_ids: Optional[List[int]]) -> None:
        """Validate product IDs list"""
        if product_ids is None or len(product_ids) == 0:
            return
        
        if not isinstance(product_ids, (list, tuple)):
            raise ValidationError(f"Product IDs must be list or tuple, got {type(product_ids)}")
        
        if len(product_ids) > MAX_PRODUCT_IDS:
            raise ValidationError(f"Too many product IDs: {len(product_ids)} (max {MAX_PRODUCT_IDS})")
        
        for pid in product_ids:
            if not isinstance(pid, int):
                raise ValidationError(f"Product ID must be integer, got {type(pid)}: {pid}")
            if pid <= 0:
                raise ValidationError(f"Invalid product ID: {pid}")
    
    def _validate_list_input(self, items: Optional[List[str]], name: str, max_items: int) -> None:
        """Validate string list inputs"""
        if items is None or len(items) == 0:
            return
        
        if not isinstance(items, (list, tuple)):
            raise ValidationError(f"{name} must be list or tuple, got {type(items)}")
        
        if len(items) > max_items:
            raise ValidationError(f"Too many {name}: {len(items)} (max {max_items})")
        
        for item in items:
            if not isinstance(item, str):
                raise ValidationError(f"{name} item must be string, got {type(item)}")
            if len(item) > 200:
                raise ValidationError(f"{name} item too long: {len(item)} chars")
    
    # Entity Mapping
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_entity_id(_self, entity_name: str) -> Optional[int]:
        """Map entity name to entity ID"""
        _self._validate_entity_name(entity_name)
        
        if entity_name in _self._entity_id_cache:
            return _self._entity_id_cache[entity_name]
        
        try:
            query = """
                SELECT id 
                FROM companies 
                WHERE english_name = :entity_name
                  AND delete_flag = 0
                LIMIT 1
            """
            
            with _self.get_connection() as conn:
                result = conn.execute(text(query), {'entity_name': entity_name}).fetchone()
                
                if result:
                    entity_id = int(result[0])
                    _self._entity_id_cache[entity_name] = entity_id
                    logger.info(f"Entity ID mapping: '{entity_name}' -> {entity_id}")
                    return entity_id
                else:
                    logger.warning(f"Entity not found: {entity_name}")
                    return None
                    
        except SQLAlchemyError as e:
            logger.error(f"Error mapping entity name to ID: {e}", exc_info=True)
            raise DataLoadError(f"Failed to get entity ID: {str(e)}")
    
    # Safety Stock
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def check_safety_stock_availability(_self) -> bool:
        """Check if safety stock data is available"""
        if _self._safety_stock_available is not None:
            return _self._safety_stock_available
            
        try:
            query = """
                SELECT COUNT(*) as count
                FROM safety_stock_levels
                WHERE delete_flag = 0 
                  AND is_active = 1
                  AND CURRENT_DATE() BETWEEN effective_from 
                  AND COALESCE(effective_to, '2999-12-31')
                LIMIT 1
            """
            
            with _self.get_connection() as conn:
                result = conn.execute(text(query)).fetchone()
                _self._safety_stock_available = result[0] > 0 if result else False
                
            logger.info(f"Safety stock availability: {_self._safety_stock_available}")
            return _self._safety_stock_available
            
        except Exception as e:
            logger.warning(f"Safety stock tables not available: {e}")
            _self._safety_stock_available = False
            return False
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_date_range(_self) -> Dict[str, date]:
        """Get min/max dates from data"""
        try:
            query = """
                SELECT 
                    MIN(date_col) as min_date,
                    MAX(date_col) as max_date
                FROM (
                    SELECT availability_date as date_col 
                    FROM unified_supply_view 
                    WHERE availability_date IS NOT NULL
                    UNION ALL
                    SELECT required_date as date_col 
                    FROM unified_demand_view 
                    WHERE required_date IS NOT NULL
                ) AS all_dates
            """
            
            with _self.get_connection() as conn:
                result = conn.execute(text(query)).fetchone()
                
                if result and result[0] and result[1]:
                    min_date = pd.to_datetime(result[0]).date()
                    max_date = pd.to_datetime(result[1]).date()
                    
                    logger.info(f"Data date range: {min_date} to {max_date}")
                    
                    return {
                        'min_date': min_date,
                        'max_date': max_date
                    }
            
            logger.warning("Could not get date range from database")
            return {
                'min_date': date.today() - timedelta(days=90),
                'max_date': date.today() + timedelta(days=90)
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting date range: {e}", exc_info=True)
            return {
                'min_date': date.today() - timedelta(days=90),
                'max_date': date.today() + timedelta(days=90)
            }
    
    @st.cache_data(ttl=CACHE_TTL_SAFETY)
    def load_safety_stock_data(
        _self,
        entity_name: Optional[str] = None,
        product_ids: Optional[Tuple[int, ...]] = None,
        include_customer_specific: bool = True
    ) -> pd.DataFrame:
        """Load safety stock requirements"""
        try:
            _self._validate_entity_name(entity_name)
            if product_ids:
                _self._validate_product_ids(list(product_ids))
            
            entity_id = None
            if entity_name:
                entity_id = _self.get_entity_id(entity_name)
                if entity_id is None:
                    logger.warning(f"Entity '{entity_name}' not found")
                    return pd.DataFrame()
            
            query_parts = ["""
                SELECT 
                    product_id, product_name, pt_code, package_size,
                    brand, standard_uom, entity_id, entity_name,
                    customer_id, customer_name,
                    safety_stock_qty, reorder_point,
                    calculation_method, avg_daily_demand,
                    safety_days, lead_time_days, service_level_percent,
                    days_since_calculation, rule_type, priority_level
                FROM safety_stock_current_view
                WHERE 1=1
            """]
            
            params = {}
            
            if entity_id:
                query_parts.append("AND entity_id = :entity_id")
                params['entity_id'] = entity_id
            
            if product_ids:
                product_list = list(product_ids)
                placeholders = [f":prod_{i}" for i in range(len(product_list))]
                query_parts.append(f"AND product_id IN ({','.join(placeholders)})")
                for i, pid in enumerate(product_list):
                    params[f'prod_{i}'] = pid
            
            if not include_customer_specific:
                query_parts.append("AND customer_id IS NULL")
            
            query_parts.append("ORDER BY product_id, priority_level")
            query = " ".join(query_parts)
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            df = _self._process_safety_stock_dataframe(df)
            
            logger.info(f"Loaded {len(df)} safety stock rules")
            return df
            
        except ValidationError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error loading safety stock: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load safety stock data: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading safety stock: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load safety stock data: {str(e)}")
    
    def _process_safety_stock_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize safety stock dataframe"""
        if df.empty:
            return df
        
        text_cols = ['product_name', 'pt_code', 'package_size', 'brand', 
                     'standard_uom', 'entity_name', 'customer_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._normalize_text_field(x, col))
        
        numeric_columns = [
            'safety_stock_qty', 'reorder_point', 'avg_daily_demand',
            'safety_days', 'lead_time_days', 'service_level_percent',
            'days_since_calculation', 'priority_level'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df = df.sort_values(['product_id', 'priority_level'])
        df = df.groupby('product_id').first().reset_index()
        
        return df
    
    @st.cache_data(ttl=CACHE_TTL_DATA)
    def load_supply_data(
        _self,
        entity_name: Optional[str] = None,
        product_ids: Optional[Tuple[int, ...]] = None,
        brands: Optional[Tuple[str, ...]] = None,
        exclude_products: bool = False,
        exclude_brands: bool = False,
        exclude_expired: bool = True
    ) -> pd.DataFrame:
        """
        Load supply data with exclusion support
        
        Args:
            entity_name: Filter by entity
            product_ids: Product IDs to include/exclude
            brands: Brands to include/exclude
            exclude_products: If True, exclude specified products
            exclude_brands: If True, exclude specified brands
            exclude_expired: If True, exclude expired inventory
        """
        try:
            _self._validate_entity_name(entity_name)
            if product_ids:
                _self._validate_product_ids(list(product_ids))
            if brands:
                _self._validate_list_input(list(brands), "brands", MAX_BRANDS)
            
            query, params = _self._build_supply_query(
                entity_name, product_ids, brands, 
                exclude_products, exclude_brands, exclude_expired
            )
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            df = _self._process_supply_dataframe(df)
            
            logger.info(f"Loaded {len(df)} supply records (exclude_expired={exclude_expired})")
            return df
            
        except ValidationError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error loading supply: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load supply data: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading supply: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load supply data: {str(e)}")
    
    @st.cache_data(ttl=CACHE_TTL_DATA)
    def load_demand_data(
        _self,
        entity_name: Optional[str] = None,
        product_ids: Optional[Tuple[int, ...]] = None,
        brands: Optional[Tuple[str, ...]] = None,
        exclude_products: bool = False,
        exclude_brands: bool = False
    ) -> pd.DataFrame:
        """
        Load demand data with exclusion support
        
        Args:
            entity_name: Filter by entity
            product_ids: Product IDs to include/exclude
            brands: Brands to include/exclude
            exclude_products: If True, exclude specified products
            exclude_brands: If True, exclude specified brands
        """
        try:
            _self._validate_entity_name(entity_name)
            if product_ids:
                _self._validate_product_ids(list(product_ids))
            if brands:
                _self._validate_list_input(list(brands), "brands", MAX_BRANDS)
            
            query, params = _self._build_demand_query(
                entity_name, product_ids, brands,
                exclude_products, exclude_brands
            )
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            df = _self._process_demand_dataframe(df)
            
            logger.info(f"Loaded {len(df)} demand records")
            return df
            
        except ValidationError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error loading demand: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load demand data: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading demand: {e}", exc_info=True)
            raise DataLoadError(f"Failed to load demand data: {str(e)}")
    
    def _build_supply_query(
        self,
        entity_name: Optional[str],
        product_ids: Optional[Tuple[int, ...]],
        brands: Optional[Tuple[str, ...]],
        exclude_products: bool,
        exclude_brands: bool,
        exclude_expired: bool
    ) -> Tuple[str, Dict[str, Any]]:
        """Build supply query with exclusion support"""
        
        query_parts = ["""
            SELECT 
                supply_source, supply_priority,
                product_id, product_name, pt_code, package_size,
                brand, standard_uom,
                batch_number, expiry_date, days_to_expiry,
                available_quantity, availability_date, days_to_available,
                availability_status,
                warehouse_id, warehouse_name, to_location,
                entity_id, entity_name,
                unit_cost_usd, total_value_usd,
                supply_reference_id, aging_days,
                supplier_name, completion_percentage
            FROM unified_supply_view
            WHERE 1=1
        """]
        
        params = {}
        
        # Entity filter
        if entity_name:
            query_parts.append("AND entity_name = :entity_name")
            params['entity_name'] = entity_name
        
        # Product filter with exclusion support
        if product_ids:
            product_list = list(product_ids)
            placeholders = [f":prod_{i}" for i in range(len(product_list))]
            
            if exclude_products:
                query_parts.append(f"AND product_id NOT IN ({','.join(placeholders)})")
            else:
                query_parts.append(f"AND product_id IN ({','.join(placeholders)})")
            
            for i, pid in enumerate(product_list):
                params[f'prod_{i}'] = pid
        
        # Brand filter with exclusion support
        if brands:
            brand_list = list(brands)
            placeholders = [f":brand_{i}" for i in range(len(brand_list))]
            
            if exclude_brands:
                query_parts.append(f"AND brand NOT IN ({','.join(placeholders)})")
            else:
                query_parts.append(f"AND brand IN ({','.join(placeholders)})")
            
            for i, brand in enumerate(brand_list):
                params[f'brand_{i}'] = brand
        
        # Exclude expired inventory
        if exclude_expired:
            query_parts.append("""
                AND (expiry_date IS NULL OR expiry_date > CURRENT_DATE())
            """)
        
        query_parts.append("ORDER BY product_id, supply_priority, days_to_available")
        
        return " ".join(query_parts), params
    
    def _build_demand_query(
        self,
        entity_name: Optional[str],
        product_ids: Optional[Tuple[int, ...]],
        brands: Optional[Tuple[str, ...]],
        exclude_products: bool,
        exclude_brands: bool
    ) -> Tuple[str, Dict[str, Any]]:
        """Build demand query with exclusion support"""
        
        query_parts = ["""
            SELECT 
                demand_source, demand_priority,
                product_id, product_name, pt_code, package_size,
                brand, standard_uom,
                customer, customer_code,
                required_quantity, required_date, days_to_required,
                demand_status, urgency_level,
                is_allocated, allocation_coverage_percent,
                allocated_quantity, unallocated_quantity,
                is_over_committed, over_committed_qty_standard,
                selling_unit_price, total_value_usd,
                demand_reference_id, entity_name, aging_days
            FROM unified_demand_view
            WHERE 1=1
        """]
        
        params = {}
        
        # Entity filter
        if entity_name:
            query_parts.append("AND entity_name = :entity_name")
            params['entity_name'] = entity_name
        
        # Product filter with exclusion support
        if product_ids:
            product_list = list(product_ids)
            placeholders = [f":prod_{i}" for i in range(len(product_list))]
            
            if exclude_products:
                query_parts.append(f"AND product_id NOT IN ({','.join(placeholders)})")
            else:
                query_parts.append(f"AND product_id IN ({','.join(placeholders)})")
            
            for i, pid in enumerate(product_list):
                params[f'prod_{i}'] = pid
        
        # Brand filter with exclusion support
        if brands:
            brand_list = list(brands)
            placeholders = [f":brand_{i}" for i in range(len(brand_list))]
            
            if exclude_brands:
                query_parts.append(f"AND brand NOT IN ({','.join(placeholders)})")
            else:
                query_parts.append(f"AND brand IN ({','.join(placeholders)})")
            
            for i, brand in enumerate(brand_list):
                params[f'brand_{i}'] = brand
        
        query_parts.append("ORDER BY product_id, demand_priority, days_to_required")
        
        return " ".join(query_parts), params
    
    def _process_supply_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize supply dataframe"""
        if df.empty:
            return df
        
        text_cols = ['product_name', 'pt_code', 'package_size', 'brand', 
                     'standard_uom', 'warehouse_name', 'entity_name', 'supplier_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._normalize_text_field(x, col))
        
        date_columns = ['availability_date', 'expiry_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        numeric_columns = [
            'available_quantity', 'days_to_available', 'days_to_expiry',
            'unit_cost_usd', 'total_value_usd', 'aging_days', 'completion_percentage'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _process_demand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize demand dataframe"""
        if df.empty:
            return df
        
        text_cols = ['product_name', 'pt_code', 'package_size', 'brand', 
                     'standard_uom', 'customer', 'customer_code', 'entity_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._normalize_text_field(x, col))
        
        date_columns = ['required_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        numeric_columns = [
            'required_quantity', 'days_to_required', 'allocation_coverage_percent',
            'allocated_quantity', 'unallocated_quantity', 'over_committed_qty_standard',
            'selling_unit_price', 'total_value_usd', 'aging_days'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        bool_columns = ['is_allocated', 'is_over_committed']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({
                    'Yes': True, 'No': False,
                    1: True, 0: False,
                    True: True, False: False
                }).fillna(False)
        
        return df
    
    # Reference Data - Enhanced
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_entities(_self) -> List[str]:
        """Get list of unique entity names"""
        try:
            query = """
                SELECT DISTINCT entity_name 
                FROM (
                    SELECT DISTINCT entity_name FROM unified_supply_view 
                    WHERE entity_name IS NOT NULL
                    UNION
                    SELECT DISTINCT entity_name FROM unified_demand_view 
                    WHERE entity_name IS NOT NULL
                ) AS entities
                ORDER BY entity_name
            """
            
            with _self.get_connection() as conn:
                result = conn.execute(text(query))
                entities = [row[0] for row in result if row[0]]
            
            logger.info(f"Loaded {len(entities)} entities")
            return entities
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting entities: {e}", exc_info=True)
            raise DataLoadError(f"Failed to get entities: {str(e)}")
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_entities_formatted(_self) -> pd.DataFrame:
        """Get entities with company_code for formatted display"""
        try:
            query = """
                SELECT DISTINCT 
                    c.english_name,
                    c.company_code
                FROM companies c
                WHERE c.delete_flag = 0
                  AND c.english_name IN (
                    SELECT DISTINCT entity_name FROM unified_supply_view
                    WHERE entity_name IS NOT NULL
                    UNION
                    SELECT DISTINCT entity_name FROM unified_demand_view
                    WHERE entity_name IS NOT NULL
                  )
                ORDER BY c.english_name
            """
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn)
            
            logger.info(f"Loaded {len(df)} formatted entities")
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting formatted entities: {e}", exc_info=True)
            raise DataLoadError(f"Failed to get formatted entities: {str(e)}")
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_products(_self, entity_name: Optional[str] = None) -> pd.DataFrame:
        """Get list of products with package_size"""
        try:
            _self._validate_entity_name(entity_name)
            
            params = {}
            entity_filter = ""
            
            if entity_name:
                entity_filter = "WHERE entity_name = :entity_name"
                params['entity_name'] = entity_name
            
            query = f"""
                SELECT DISTINCT 
                    product_id, product_name, pt_code, package_size,
                    brand, standard_uom
                FROM (
                    SELECT product_id, product_name, pt_code, package_size, 
                           brand, standard_uom, entity_name
                    FROM unified_supply_view
                    WHERE product_id IS NOT NULL
                    UNION
                    SELECT product_id, product_name, pt_code, package_size, 
                           brand, standard_uom, entity_name
                    FROM unified_demand_view
                    WHERE product_id IS NOT NULL
                ) AS products
                {entity_filter}
                ORDER BY pt_code, product_name
            """
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            text_cols = ['product_name', 'pt_code', 'package_size', 'brand', 'standard_uom']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: _self._normalize_text_field(x, col))
            
            logger.info(f"Loaded {len(df)} products")
            return df
            
        except ValidationError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error getting products: {e}", exc_info=True)
            raise DataLoadError(f"Failed to get products: {str(e)}")
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_brands(_self, entity_name: Optional[str] = None) -> List[str]:
        """Get list of unique brands"""
        try:
            _self._validate_entity_name(entity_name)
            
            params = {}
            entity_filter = ""
            
            if entity_name:
                entity_filter = "WHERE entity_name = :entity_name"
                params['entity_name'] = entity_name
            
            query = f"""
                SELECT DISTINCT brand
                FROM (
                    SELECT DISTINCT brand, entity_name FROM unified_supply_view 
                    WHERE brand IS NOT NULL
                    UNION
                    SELECT DISTINCT brand, entity_name FROM unified_demand_view 
                    WHERE brand IS NOT NULL
                ) AS brands
                {entity_filter}
                ORDER BY brand
            """
            
            with _self.get_connection() as conn:
                result = conn.execute(text(query), params)
                brands = []
                for row in result:
                    if row[0]:
                        normalized = _self._normalize_text_field(row[0], 'brand')
                        if normalized and normalized not in brands:
                            brands.append(normalized)
            
            brands.sort()
            logger.info(f"Loaded {len(brands)} brands")
            return brands
            
        except ValidationError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error getting brands: {e}", exc_info=True)
            raise DataLoadError(f"Failed to get brands: {str(e)}")