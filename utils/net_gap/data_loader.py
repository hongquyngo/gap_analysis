# utils/net_gap/data_loader.py

"""
Data loader module for GAP Analysis System - Version 2.2 FIXED
- Added data normalization to prevent JOIN issues
- Improved consistency checks
- Better handling of text fields
- Enhanced validation
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

# Import database connection
import os
import sys
from pathlib import Path
project_root = os.environ.get('PROJECT_ROOT', Path(__file__).parent.parent.parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from utils.db import get_db_engine

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL_DATA = 300  # 5 minutes for transactional data
CACHE_TTL_REFERENCE = 600  # 10 minutes for reference data
CACHE_TTL_SAFETY = 900  # 15 minutes for safety stock

# Validation constants
MAX_ENTITY_NAME_LENGTH = 200
MAX_PRODUCT_IDS = 1000
MAX_BRANDS = 100
MAX_CUSTOMERS = 500
DATE_RANGE_MAX_DAYS = 730  # 2 years


# Custom Exceptions
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
    """Handles all data loading operations for GAP analysis with validation and normalization"""
    
    def __init__(self):
        """Initialize data loader with database engine"""
        self._engine = None
        self._safety_stock_available = None
        self._entity_id_cache = {}  # Cache for entity name to ID mapping
    
    @property
    def engine(self):
        """Lazy load database engine with connection validation"""
        if self._engine is None:
            try:
                self._engine = get_db_engine()
                # Test connection
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Database connection established successfully")
            except Exception as e:
                logger.error(f"Failed to establish database connection: {e}", exc_info=True)
                raise DatabaseConnectionError(f"Cannot connect to database: {str(e)}")
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with error handling"""
        try:
            conn = self.engine.connect()
            yield conn
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            raise DatabaseConnectionError(f"Database connection failed: {str(e)}")
        finally:
            try:
                conn.close()
            except:
                pass
    
    def _normalize_text_field(self, value: Any, field_name: str = '') -> str:
        """
        Normalize a text field value for consistency
        
        Args:
            value: Value to normalize
            field_name: Name of field for special handling
            
        Returns:
            Normalized string value
        """
        if pd.isna(value) or value is None:
            return ''
        
        # Convert to string
        str_value = str(value).strip()
        
        # Special handling for certain fields
        if field_name == 'pt_code':
            # Keep pt_code case-sensitive
            return str_value
        elif field_name in ['brand', 'product_name', 'standard_uom']:
            # Uppercase for consistency
            return str_value.upper()
        else:
            return str_value
    
    # Input Validation Methods
    def _validate_entity_name(self, entity_name: Optional[str]) -> None:
        """Validate entity name input"""
        if entity_name is None:
            return  # None is valid (means all entities)
        
        if not isinstance(entity_name, str):
            raise ValidationError(f"Entity name must be string, got {type(entity_name)}")
        
        if len(entity_name) == 0:
            raise ValidationError("Entity name cannot be empty")
        
        if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
            raise ValidationError(f"Entity name too long (max {MAX_ENTITY_NAME_LENGTH} chars)")
        
        # Check for suspicious characters (basic SQL injection prevention)
        if re.search(r'[;\'"\\]', entity_name):
            raise ValidationError("Entity name contains invalid characters")
    
    def _validate_date_range(self, date_from: Optional[date], date_to: Optional[date]) -> None:
        """Validate date range input"""
        if date_from is None and date_to is None:
            return
        
        if date_from is not None and date_to is not None:
            if date_from > date_to:
                raise ValidationError(f"Invalid date range: {date_from} > {date_to}")
            
            days_diff = (date_to - date_from).days
            if days_diff > DATE_RANGE_MAX_DAYS:
                raise ValidationError(f"Date range too large: {days_diff} days (max {DATE_RANGE_MAX_DAYS})")
        
        # Check for reasonable date bounds
        min_date = date.today() - timedelta(days=365 * 2)
        max_date = date.today() + timedelta(days=365 * 2)
        
        if date_from is not None and date_from < min_date:
            raise ValidationError(f"Start date too far in past: {date_from}")
        
        if date_to is not None and date_to > max_date:
            raise ValidationError(f"End date too far in future: {date_to}")
    
    def _validate_product_ids(self, product_ids: Optional[List[int]]) -> None:
        """Validate product IDs list"""
        if product_ids is None or len(product_ids) == 0:
            return
        
        if not isinstance(product_ids, (list, tuple)):
            raise ValidationError(f"Product IDs must be list or tuple, got {type(product_ids)}")
        
        if len(product_ids) > MAX_PRODUCT_IDS:
            raise ValidationError(f"Too many product IDs: {len(product_ids)} (max {MAX_PRODUCT_IDS})")
        
        # Validate each ID
        for pid in product_ids:
            if not isinstance(pid, int):
                raise ValidationError(f"Product ID must be integer, got {type(pid)}: {pid}")
            if pid <= 0:
                raise ValidationError(f"Invalid product ID: {pid}")
    
    def _validate_list_input(self, items: Optional[List[str]], name: str, max_items: int) -> None:
        """Validate string list inputs (brands, customers, etc.)"""
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
    
    # Entity ID Mapping
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_entity_id(_self, entity_name: str) -> Optional[int]:
        """Map entity name to entity ID"""
        _self._validate_entity_name(entity_name)
        
        # Check cache first
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
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def check_safety_stock_availability(_self) -> bool:
        """Check if safety stock data is available in the database"""
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
    
    @st.cache_data(ttl=CACHE_TTL_SAFETY)
    def load_safety_stock_data(
        _self,
        entity_name: Optional[str] = None,
        product_ids: Optional[Tuple[int, ...]] = None,
        include_customer_specific: bool = True
    ) -> pd.DataFrame:
        """Load safety stock requirements from safety_stock_current_view"""
        try:
            # Validate inputs
            _self._validate_entity_name(entity_name)
            if product_ids:
                _self._validate_product_ids(list(product_ids))
            
            # Get entity ID if entity name provided
            entity_id = None
            if entity_name:
                entity_id = _self.get_entity_id(entity_name)
                if entity_id is None:
                    logger.warning(f"Entity '{entity_name}' not found, returning empty DataFrame")
                    return pd.DataFrame()
            
            # Build query with filters
            query_parts = ["""
                SELECT 
                    product_id,
                    product_name,
                    pt_code,
                    brand,
                    standard_uom,
                    entity_id,
                    entity_name,
                    customer_id,
                    customer_name,
                    safety_stock_qty,
                    reorder_point,
                    calculation_method,
                    avg_daily_demand,
                    safety_days,
                    lead_time_days,
                    service_level_percent,
                    days_since_calculation,
                    rule_type,
                    priority_level
                FROM safety_stock_current_view
                WHERE 1=1
            """]
            
            params = {}
            
            # Add filters
            if entity_id:
                query_parts.append("AND entity_id = :entity_id")
                params['entity_id'] = entity_id
            
            if product_ids:
                # Convert tuple to list for query
                product_list = list(product_ids)
                product_placeholders = [f":prod_{i}" for i in range(len(product_list))]
                query_parts.append(f"AND product_id IN ({','.join(product_placeholders)})")
                for i, pid in enumerate(product_list):
                    params[f'prod_{i}'] = pid
            
            if not include_customer_specific:
                query_parts.append("AND customer_id IS NULL")
            
            query_parts.append("ORDER BY product_id, priority_level")
            
            query = " ".join(query_parts)
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Process and normalize dataframe
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
        
        # Normalize text fields
        text_cols = ['product_name', 'pt_code', 'brand', 'standard_uom', 
                     'entity_name', 'customer_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._normalize_text_field(x, col))
        
        # Convert numeric columns
        numeric_columns = [
            'safety_stock_qty', 'reorder_point', 'avg_daily_demand',
            'safety_days', 'lead_time_days', 'service_level_percent',
            'days_since_calculation', 'priority_level'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = self._safe_numeric_conversion(df, col)
        
        # For multiple rules per product, keep only highest priority
        df = df.sort_values(['product_id', 'priority_level'])
        df = df.groupby('product_id').first().reset_index()
        
        return df
    
    def _safe_numeric_conversion(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Convert column to numeric with logging of failures"""
        original = df[col]
        converted = pd.to_numeric(original, errors='coerce')
        failed = converted.isna() & original.notna()
        
        if failed.any():
            failed_count = failed.sum()
            logger.warning(f"Failed to convert {failed_count} values in column '{col}'")
            if failed_count <= 5:
                logger.debug(f"Failed values in {col}: {original[failed].tolist()}")
        
        return converted.fillna(0)
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_date_range(_self) -> Dict[str, date]:
        """Get min and max dates from supply and demand views"""
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
                WHERE date_col >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
                  AND date_col <= DATE_ADD(CURRENT_DATE, INTERVAL 1 YEAR)
            """
            
            with _self.get_connection() as conn:
                result = conn.execute(text(query)).fetchone()
                
                if result and result[0] and result[1]:
                    min_date = pd.to_datetime(result[0]).date()
                    max_date = pd.to_datetime(result[1]).date()
                    
                    logger.info(f"Date range from data: {min_date} to {max_date}")
                    
                    return {
                        'min_date': min_date,
                        'max_date': max_date
                    }
            
            # Default fallback
            logger.warning("Could not get date range from database, using defaults")
            return {
                'min_date': date.today(),
                'max_date': date.today() + timedelta(days=30)
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting date range: {e}", exc_info=True)
            # Return default range on error
            return {
                'min_date': date.today(),
                'max_date': date.today() + timedelta(days=30)
            }
    
    @st.cache_data(ttl=CACHE_TTL_DATA)
    def load_supply_data(
        _self,
        entity_name: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        product_ids: Optional[Tuple[int, ...]] = None,
        brands: Optional[Tuple[str, ...]] = None
    ) -> pd.DataFrame:
        """Load supply data from unified_supply_view with normalization"""
        try:
            # Validate inputs
            _self._validate_entity_name(entity_name)
            _self._validate_date_range(date_from, date_to)
            if product_ids:
                _self._validate_product_ids(list(product_ids))
            if brands:
                _self._validate_list_input(list(brands), "brands", MAX_BRANDS)
            
            # Build parameterized query
            query, params = _self._build_supply_query(
                entity_name, date_from, date_to, product_ids, brands
            )
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Process and normalize dataframe
            df = _self._process_supply_dataframe(df)
            
            logger.info(f"Loaded {len(df)} supply records")
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
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        product_ids: Optional[Tuple[int, ...]] = None,
        brands: Optional[Tuple[str, ...]] = None,
        customers: Optional[Tuple[str, ...]] = None
    ) -> pd.DataFrame:
        """Load demand data from unified_demand_view with normalization"""
        try:
            # Validate inputs
            _self._validate_entity_name(entity_name)
            _self._validate_date_range(date_from, date_to)
            if product_ids:
                _self._validate_product_ids(list(product_ids))
            if brands:
                _self._validate_list_input(list(brands), "brands", MAX_BRANDS)
            if customers:
                _self._validate_list_input(list(customers), "customers", MAX_CUSTOMERS)
            
            # Build parameterized query
            query, params = _self._build_demand_query(
                entity_name, date_from, date_to, product_ids, brands, customers
            )
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Process and normalize dataframe
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
        date_from: Optional[date],
        date_to: Optional[date],
        product_ids: Optional[Tuple[int, ...]],
        brands: Optional[Tuple[str, ...]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Build parameterized supply query"""
        
        query_parts = ["""
            SELECT 
                supply_source,
                supply_priority,
                product_id,
                product_name,
                pt_code,
                brand,
                package_size,
                standard_uom,
                batch_number,
                expiry_date,
                days_to_expiry,
                available_quantity,
                availability_date,
                days_to_available,
                availability_status,
                warehouse_id,
                warehouse_name,
                to_location,
                entity_id,
                entity_name,
                unit_cost_usd,
                total_value_usd,
                supply_reference_id,
                aging_days,
                supplier_name,
                completion_percentage
            FROM unified_supply_view
            WHERE 1=1
        """]
        
        params = {}
        
        if entity_name:
            query_parts.append("AND entity_name = :entity_name")
            params['entity_name'] = entity_name
        
        if date_from:
            query_parts.append("AND availability_date >= :date_from")
            params['date_from'] = date_from
        
        if date_to:
            query_parts.append("AND availability_date <= :date_to")
            params['date_to'] = date_to
        
        if product_ids:
            product_list = list(product_ids)
            product_placeholders = [f":prod_{i}" for i in range(len(product_list))]
            query_parts.append(f"AND product_id IN ({','.join(product_placeholders)})")
            for i, pid in enumerate(product_list):
                params[f'prod_{i}'] = pid
        
        if brands:
            brand_list = list(brands)
            brand_placeholders = [f":brand_{i}" for i in range(len(brand_list))]
            query_parts.append(f"AND brand IN ({','.join(brand_placeholders)})")
            for i, brand in enumerate(brand_list):
                params[f'brand_{i}'] = brand
        
        query_parts.append("ORDER BY product_id, supply_priority, days_to_available")
        
        return " ".join(query_parts), params
    
    def _build_demand_query(
        self,
        entity_name: Optional[str],
        date_from: Optional[date],
        date_to: Optional[date],
        product_ids: Optional[Tuple[int, ...]],
        brands: Optional[Tuple[str, ...]],
        customers: Optional[Tuple[str, ...]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Build parameterized demand query"""
        
        query_parts = ["""
            SELECT 
                demand_source,
                demand_priority,
                product_id,
                product_name,
                pt_code,
                brand,
                package_size,
                standard_uom,
                customer,
                customer_code,
                required_quantity,
                required_date,
                days_to_required,
                demand_status,
                urgency_level,
                is_allocated,
                allocation_coverage_percent,
                allocated_quantity,
                unallocated_quantity,
                is_over_committed,
                over_committed_qty_standard,
                selling_unit_price,
                total_value_usd,
                demand_reference_id,
                entity_name,
                aging_days
            FROM unified_demand_view
            WHERE 1=1
        """]
        
        params = {}
        
        if entity_name:
            query_parts.append("AND entity_name = :entity_name")
            params['entity_name'] = entity_name
        
        if date_from:
            query_parts.append("AND required_date >= :date_from")
            params['date_from'] = date_from
        
        if date_to:
            query_parts.append("AND required_date <= :date_to")
            params['date_to'] = date_to
        
        if product_ids:
            product_list = list(product_ids)
            product_placeholders = [f":prod_{i}" for i in range(len(product_list))]
            query_parts.append(f"AND product_id IN ({','.join(product_placeholders)})")
            for i, pid in enumerate(product_list):
                params[f'prod_{i}'] = pid
        
        if brands:
            brand_list = list(brands)
            brand_placeholders = [f":brand_{i}" for i in range(len(brand_list))]
            query_parts.append(f"AND brand IN ({','.join(brand_placeholders)})")
            for i, brand in enumerate(brand_list):
                params[f'brand_{i}'] = brand
        
        if customers:
            customer_list = list(customers)
            customer_placeholders = [f":cust_{i}" for i in range(len(customer_list))]
            query_parts.append(f"AND customer IN ({','.join(customer_placeholders)})")
            for i, customer in enumerate(customer_list):
                params[f'cust_{i}'] = customer
        
        query_parts.append("ORDER BY product_id, demand_priority, days_to_required")
        
        return " ".join(query_parts), params
    
    def _process_supply_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize supply dataframe"""
        if df.empty:
            return df
        
        # Normalize text fields FIRST
        text_cols = ['product_name', 'pt_code', 'brand', 'standard_uom', 
                     'warehouse_name', 'entity_name', 'supplier_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._normalize_text_field(x, col))
        
        # Convert date columns
        date_columns = ['availability_date', 'expiry_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = [
            'available_quantity', 'days_to_available', 'days_to_expiry',
            'unit_cost_usd', 'total_value_usd', 'aging_days', 'completion_percentage'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = self._safe_numeric_conversion(df, col)
        
        # Log sample for debugging
        if len(df) > 0:
            sample = df.iloc[0]
            logger.debug(f"Supply sample - product_id: {sample.get('product_id')}, "
                        f"pt_code: '{sample.get('pt_code')}', "
                        f"brand: '{sample.get('brand')}'")
        
        return df
    
    def _process_demand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize demand dataframe"""
        if df.empty:
            return df
        
        # Normalize text fields FIRST
        text_cols = ['product_name', 'pt_code', 'brand', 'standard_uom', 
                     'customer', 'customer_code', 'entity_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self._normalize_text_field(x, col))
        
        # Convert date columns
        date_columns = ['required_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = [
            'required_quantity', 'days_to_required', 'allocation_coverage_percent',
            'allocated_quantity', 'unallocated_quantity', 'over_committed_qty_standard',
            'selling_unit_price', 'total_value_usd', 'aging_days'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = self._safe_numeric_conversion(df, col)
        
        # Convert boolean columns
        bool_columns = ['is_allocated', 'is_over_committed']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({
                    'Yes': True, 'No': False,
                    1: True, 0: False,
                    True: True, False: False
                }).fillna(False)
        
        # Log sample for debugging
        if len(df) > 0:
            sample = df.iloc[0]
            logger.debug(f"Demand sample - product_id: {sample.get('product_id')}, "
                        f"pt_code: '{sample.get('pt_code')}', "
                        f"brand: '{sample.get('brand')}'")
        
        return df
    
    # Reference Data Methods
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_entities(_self) -> List[str]:
        """Get list of unique entities from both views"""
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
                entities = [row[0] for row in result]
            
            logger.info(f"Loaded {len(entities)} entities")
            return entities
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting entities: {e}", exc_info=True)
            raise DataLoadError(f"Failed to get entities: {str(e)}")
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_products(_self, entity_name: Optional[str] = None) -> pd.DataFrame:
        """Get list of products with basic info (normalized)"""
        try:
            _self._validate_entity_name(entity_name)
            
            params = {}
            entity_filter = ""
            
            if entity_name:
                entity_filter = "WHERE entity_name = :entity_name"
                params['entity_name'] = entity_name
            
            query = f"""
                SELECT DISTINCT 
                    product_id,
                    product_name,
                    pt_code,
                    brand,
                    standard_uom
                FROM (
                    SELECT product_id, product_name, pt_code, brand, 
                           standard_uom, entity_name
                    FROM unified_supply_view
                    WHERE product_id IS NOT NULL
                    UNION
                    SELECT product_id, product_name, pt_code, brand,
                           standard_uom, entity_name
                    FROM unified_demand_view
                    WHERE product_id IS NOT NULL
                ) AS products
                {entity_filter}
                ORDER BY pt_code, product_name
            """
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Normalize text fields
            text_cols = ['product_name', 'pt_code', 'brand', 'standard_uom']
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
        """Get list of unique brands (normalized)"""
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
                        # Normalize brand name
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
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_customers(_self, entity_name: Optional[str] = None) -> List[str]:
        """Get list of unique customers (normalized)"""
        try:
            _self._validate_entity_name(entity_name)
            
            query = """
                SELECT DISTINCT customer
                FROM unified_demand_view
                WHERE customer IS NOT NULL
            """
            
            params = {}
            
            if entity_name:
                query += " AND entity_name = :entity_name"
                params['entity_name'] = entity_name
            
            query += " ORDER BY customer"
            
            with _self.get_connection() as conn:
                result = conn.execute(text(query), params)
                customers = []
                for row in result:
                    if row[0]:
                        # Normalize customer name
                        normalized = _self._normalize_text_field(row[0], 'customer')
                        if normalized and normalized not in customers:
                            customers.append(normalized)
            
            customers.sort()
            logger.info(f"Loaded {len(customers)} customers")
            return customers
            
        except ValidationError:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error getting customers: {e}", exc_info=True)
            raise DataLoadError(f"Failed to get customers: {str(e)}")