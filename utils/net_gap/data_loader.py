# utils/net_gap/data_loader.py

"""
Data loader module for GAP Analysis System - Version 2.0
- Added safety stock data loading
- Improved caching strategy
- Better error handling
"""

import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging
from sqlalchemy import text
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
CACHE_TTL_SAFETY = 900  # 15 minutes for safety stock (changes less frequently)


class GAPDataLoader:
    """Handles all data loading operations for GAP analysis including safety stock"""
    
    def __init__(self):
        """Initialize data loader with database engine"""
        self._engine = None
        self._safety_stock_available = None  # Cache availability check
    
    @property
    def engine(self):
        """Lazy load database engine"""
        if self._engine is None:
            self._engine = get_db_engine()
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def check_safety_stock_availability(_self) -> bool:
        """
        Check if safety stock data is available in the database
        
        Returns:
            True if safety stock tables exist and have data
        """
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
        entity_id: Optional[int] = None,
        product_ids: Optional[List[int]] = None,
        include_customer_specific: bool = True
    ) -> pd.DataFrame:
        """
        Load safety stock requirements from safety_stock_current_view
        
        Args:
            entity_id: Filter by entity ID
            product_ids: List of product IDs to filter
            include_customer_specific: Whether to include customer-specific rules
            
        Returns:
            DataFrame with safety stock requirements
        """
        try:
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
                # Handle list of product IDs
                product_placeholders = [f":prod_{i}" for i in range(len(product_ids))]
                query_parts.append(f"AND product_id IN ({','.join(product_placeholders)})")
                for i, pid in enumerate(product_ids):
                    params[f'prod_{i}'] = pid
            
            if not include_customer_specific:
                query_parts.append("AND customer_id IS NULL")
            
            query_parts.append("ORDER BY product_id, priority_level")
            
            query = " ".join(query_parts)
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Process dataframe
            df = _self._process_safety_stock_dataframe(df)
            
            logger.info(f"Loaded {len(df)} safety stock rules")
            return df
            
        except Exception as e:
            logger.error(f"Error loading safety stock data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _process_safety_stock_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean safety stock dataframe"""
        if df.empty:
            return df
        
        # Convert numeric columns
        numeric_columns = [
            'safety_stock_qty', 'reorder_point', 'avg_daily_demand',
            'safety_days', 'lead_time_days', 'service_level_percent',
            'days_since_calculation', 'priority_level'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # For multiple rules per product, keep only highest priority (lowest number)
        # This handles customer-specific overrides automatically
        df = df.sort_values(['product_id', 'priority_level'])
        df = df.groupby('product_id').first().reset_index()
        
        return df
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_date_range(_self) -> Dict[str, date]:
        """
        Get min and max dates from supply and demand views
        Returns dict with 'min_date' and 'max_date'
        """
        try:
            query = """
                SELECT 
                    MIN(date_col) as min_date,
                    MAX(date_col) as max_date
                FROM (
                    -- Supply dates
                    SELECT availability_date as date_col 
                    FROM unified_supply_view 
                    WHERE availability_date IS NOT NULL
                    
                    UNION ALL
                    
                    -- Demand dates
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
            
        except Exception as e:
            logger.error(f"Error getting date range: {e}", exc_info=True)
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
        product_ids: Optional[List[int]] = None,
        brands: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load supply data from unified_supply_view
        
        Args:
            entity_name: Filter by entity
            date_from: Start date for availability_date filter
            date_to: End date for availability_date filter
            product_ids: List of product IDs to filter
            brands: List of brands to filter
            
        Returns:
            DataFrame with supply data
        """
        try:
            # Build parameterized query
            query, params = _self._build_supply_query(
                entity_name, date_from, date_to, product_ids, brands
            )
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Process dataframe
            df = _self._process_supply_dataframe(df)
            
            logger.info(f"Loaded {len(df)} supply records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading supply data: {e}", exc_info=True)
            return pd.DataFrame()
    
    @st.cache_data(ttl=CACHE_TTL_DATA)
    def load_demand_data(
        _self,
        entity_name: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        product_ids: Optional[List[int]] = None,
        brands: Optional[List[str]] = None,
        customers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load demand data from unified_demand_view
        
        Args:
            entity_name: Filter by entity
            date_from: Start date for required_date filter
            date_to: End date for required_date filter
            product_ids: List of product IDs to filter
            brands: List of brands to filter
            customers: List of customers to filter
            
        Returns:
            DataFrame with demand data
        """
        try:
            # Build parameterized query
            query, params = _self._build_demand_query(
                entity_name, date_from, date_to, product_ids, brands, customers
            )
            
            with _self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Process dataframe
            df = _self._process_demand_dataframe(df)
            
            logger.info(f"Loaded {len(df)} demand records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading demand data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _build_supply_query(
        self,
        entity_name: Optional[str],
        date_from: Optional[date],
        date_to: Optional[date],
        product_ids: Optional[List[int]],
        brands: Optional[List[str]]
    ) -> Tuple[str, Dict[str, Any]]:
        """Build parameterized supply query"""
        
        # Base query with required columns
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
        
        # Add filters
        if entity_name:
            query_parts.append("AND entity_name = :entity_name")
            params['entity_name'] = entity_name
        
        if date_from:
            query_parts.append("AND availability_date >= :date_from")
            params['date_from'] = date_from
        
        if date_to:
            query_parts.append("AND availability_date <= :date_to")
            params['date_to'] = date_to
        
        # Handle multiselect product_ids
        if product_ids:
            product_placeholders = [f":prod_{i}" for i in range(len(product_ids))]
            query_parts.append(f"AND product_id IN ({','.join(product_placeholders)})")
            for i, pid in enumerate(product_ids):
                params[f'prod_{i}'] = pid
        
        # Handle multiselect brands
        if brands:
            brand_placeholders = [f":brand_{i}" for i in range(len(brands))]
            query_parts.append(f"AND brand IN ({','.join(brand_placeholders)})")
            for i, brand in enumerate(brands):
                params[f'brand_{i}'] = brand
        
        query_parts.append("ORDER BY product_id, supply_priority, days_to_available")
        
        return " ".join(query_parts), params
    
    def _build_demand_query(
        self,
        entity_name: Optional[str],
        date_from: Optional[date],
        date_to: Optional[date],
        product_ids: Optional[List[int]],
        brands: Optional[List[str]],
        customers: Optional[List[str]]
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
        
        # Add filters
        if entity_name:
            query_parts.append("AND entity_name = :entity_name")
            params['entity_name'] = entity_name
        
        if date_from:
            query_parts.append("AND required_date >= :date_from")
            params['date_from'] = date_from
        
        if date_to:
            query_parts.append("AND required_date <= :date_to")
            params['date_to'] = date_to
        
        # Handle multiselect product_ids
        if product_ids:
            product_placeholders = [f":prod_{i}" for i in range(len(product_ids))]
            query_parts.append(f"AND product_id IN ({','.join(product_placeholders)})")
            for i, pid in enumerate(product_ids):
                params[f'prod_{i}'] = pid
        
        # Handle multiselect brands
        if brands:
            brand_placeholders = [f":brand_{i}" for i in range(len(brands))]
            query_parts.append(f"AND brand IN ({','.join(brand_placeholders)})")
            for i, brand in enumerate(brands):
                params[f'brand_{i}'] = brand
        
        # Handle multiselect customers
        if customers:
            customer_placeholders = [f":cust_{i}" for i in range(len(customers))]
            query_parts.append(f"AND customer IN ({','.join(customer_placeholders)})")
            for i, customer in enumerate(customers):
                params[f'cust_{i}'] = customer
        
        query_parts.append("ORDER BY product_id, demand_priority, days_to_required")
        
        return " ".join(query_parts), params
    
    def _process_supply_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean supply dataframe"""
        if df.empty:
            return df
        
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
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _process_demand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean demand dataframe"""
        if df.empty:
            return df
        
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
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert boolean columns
        bool_columns = ['is_allocated', 'is_over_committed']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({
                    'Yes': True, 'No': False,
                    1: True, 0: False,
                    True: True, False: False
                }).fillna(False)
        
        return df
    
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
            
            return entities
            
        except Exception as e:
            logger.error(f"Error getting entities: {e}", exc_info=True)
            return []
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_products(_self, entity_name: Optional[str] = None) -> pd.DataFrame:
        """Get list of products with basic info"""
        try:
            params = {}
            
            if entity_name:
                entity_filter = "WHERE entity_name = :entity_name"
                params['entity_name'] = entity_name
            else:
                entity_filter = ""
            
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting products: {e}", exc_info=True)
            return pd.DataFrame()
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_brands(_self, entity_name: Optional[str] = None) -> List[str]:
        """Get list of unique brands"""
        try:
            params = {}
            
            if entity_name:
                entity_filter = "WHERE entity_name = :entity_name"
                params['entity_name'] = entity_name
            else:
                entity_filter = ""
            
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
                brands = [row[0] for row in result if row[0]]
            
            return brands
            
        except Exception as e:
            logger.error(f"Error getting brands: {e}", exc_info=True)
            return []
    
    @st.cache_data(ttl=CACHE_TTL_REFERENCE)
    def get_customers(_self, entity_name: Optional[str] = None) -> List[str]:
        """Get list of unique customers"""
        try:
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
                customers = [row[0] for row in result if row[0]]
            
            return customers
            
        except Exception as e:
            logger.error(f"Error getting customers: {e}", exc_info=True)
            return []