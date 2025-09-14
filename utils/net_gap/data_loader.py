# utils/gap/data_loader.py

"""
Data loader module for GAP Analysis System
Handles loading and caching of supply and demand data from unified views
"""

import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, Dict, List
import logging
from sqlalchemy import text

# Import from parent directory utils
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Go up to project root
from utils.db import get_db_engine

logger = logging.getLogger(__name__)


class GAPDataLoader:
    """Handles all data loading operations for GAP analysis"""
    
    def __init__(self):
        self.engine = get_db_engine()
    
    @st.cache_data(ttl=300)  # 5-minute cache
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
            # Build dynamic WHERE clause
            where_conditions = ["1=1"]  # Always true, makes building easier
            params = {}
            
            if entity_name:
                where_conditions.append("entity_name = :entity_name")
                params['entity_name'] = entity_name
            
            if date_from:
                where_conditions.append("availability_date >= :date_from")
                params['date_from'] = date_from
            
            if date_to:
                where_conditions.append("availability_date <= :date_to")
                params['date_to'] = date_to
            
            if product_ids:
                # Convert list to comma-separated string for IN clause
                product_ids_str = ','.join(map(str, product_ids))
                where_conditions.append(f"product_id IN ({product_ids_str})")
            
            if brands:
                # Handle brands with proper escaping
                brands_str = ','.join([f"'{b}'" for b in brands])
                where_conditions.append(f"brand IN ({brands_str})")
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
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
                source_line_id,
                source_document_number,
                source_document_date,
                aging_days,
                supplier_name,
                completion_percentage
            FROM unified_supply_view
            WHERE {where_clause}
            ORDER BY product_id, supply_priority, days_to_available
            """
            
            with _self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Convert dates to datetime
            date_columns = ['availability_date', 'source_document_date', 'expiry_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Ensure numeric columns
            numeric_columns = ['available_quantity', 'days_to_available', 'days_to_expiry',
                             'unit_cost_usd', 'total_value_usd', 'aging_days', 'completion_percentage']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            logger.info(f"Loaded {len(df)} supply records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading supply data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)  # 5-minute cache
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
            # Build dynamic WHERE clause
            where_conditions = ["1=1"]
            params = {}
            
            if entity_name:
                where_conditions.append("entity_name = :entity_name")
                params['entity_name'] = entity_name
            
            if date_from:
                where_conditions.append("required_date >= :date_from")
                params['date_from'] = date_from
            
            if date_to:
                where_conditions.append("required_date <= :date_to")
                params['date_to'] = date_to
            
            if product_ids:
                product_ids_str = ','.join(map(str, product_ids))
                where_conditions.append(f"product_id IN ({product_ids_str})")
            
            if brands:
                brands_str = ','.join([f"'{b}'" for b in brands])
                where_conditions.append(f"brand IN ({brands_str})")
            
            if customers:
                customers_str = ','.join([f"'{c}'" for c in customers])
                where_conditions.append(f"customer IN ({customers_str})")
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
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
                customer_po_number,
                required_quantity,
                required_date,
                days_to_required,
                demand_status,
                urgency_level,
                is_allocated,
                allocation_count,
                allocation_coverage_percent,
                allocated_quantity,
                unallocated_quantity,
                is_over_committed,
                is_pending_over_allocated,
                over_committed_qty_standard,
                pending_over_allocated_qty_standard,
                selling_unit_price,
                total_value_usd,
                demand_reference_id,
                source_line_id,
                source_document_number,
                source_document_date,
                entity_name,
                aging_days,
                selling_uom,
                uom_conversion,
                total_delivered_standard_quantity,
                original_standard_quantity
            FROM unified_demand_view
            WHERE {where_clause}
            ORDER BY product_id, demand_priority, days_to_required
            """
            
            with _self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Convert dates to datetime
            date_columns = ['required_date', 'source_document_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Ensure numeric columns
            numeric_columns = ['required_quantity', 'days_to_required', 'allocation_count',
                             'allocation_coverage_percent', 'allocated_quantity', 'unallocated_quantity',
                             'over_committed_qty_standard', 'pending_over_allocated_qty_standard',
                             'selling_unit_price', 'total_value_usd', 'aging_days',
                             'total_delivered_standard_quantity', 'original_standard_quantity']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Convert boolean columns
            bool_columns = ['is_allocated', 'is_over_committed', 'is_pending_over_allocated']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].map({'Yes': True, 'No': False, 1: True, 0: False}).fillna(False)
            
            logger.info(f"Loaded {len(df)} demand records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading demand data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)  # 10-minute cache for reference data
    def get_entities(_self) -> List[str]:
        """Get list of unique entities from both views"""
        try:
            query = """
            SELECT DISTINCT entity_name 
            FROM (
                SELECT DISTINCT entity_name FROM unified_supply_view WHERE entity_name IS NOT NULL
                UNION
                SELECT DISTINCT entity_name FROM unified_demand_view WHERE entity_name IS NOT NULL
            ) AS entities
            ORDER BY entity_name
            """
            
            with _self.engine.connect() as conn:
                result = conn.execute(text(query))
                entities = [row[0] for row in result]
            
            return entities
            
        except Exception as e:
            logger.error(f"Error getting entities: {e}")
            return []
    
    @st.cache_data(ttl=600)  # 10-minute cache
    def get_products(_self, entity_name: Optional[str] = None) -> pd.DataFrame:
        """Get list of products with basic info"""
        try:
            where_clause = ""
            params = {}
            
            if entity_name:
                where_clause = "WHERE entity_name = :entity_name"
                params['entity_name'] = entity_name
            
            query = f"""
            SELECT DISTINCT 
                product_id,
                product_name,
                pt_code,
                brand,
                standard_uom
            FROM (
                SELECT DISTINCT 
                    product_id,
                    product_name,
                    pt_code,
                    brand,
                    standard_uom,
                    entity_name
                FROM unified_supply_view
                UNION
                SELECT DISTINCT 
                    product_id,
                    product_name,
                    pt_code,
                    brand,
                    standard_uom,
                    entity_name
                FROM unified_demand_view
            ) AS products
            {where_clause}
            ORDER BY product_name
            """
            
            with _self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting products: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)  # 10-minute cache
    def get_brands(_self, entity_name: Optional[str] = None) -> List[str]:
        """Get list of unique brands"""
        try:
            where_clause = ""
            params = {}
            
            if entity_name:
                where_clause = "WHERE entity_name = :entity_name"
                params['entity_name'] = entity_name
            
            query = f"""
            SELECT DISTINCT brand
            FROM (
                SELECT DISTINCT brand, entity_name FROM unified_supply_view WHERE brand IS NOT NULL
                UNION
                SELECT DISTINCT brand, entity_name FROM unified_demand_view WHERE brand IS NOT NULL
            ) AS brands
            {where_clause}
            ORDER BY brand
            """
            
            with _self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                brands = [row[0] for row in result]
            
            return brands
            
        except Exception as e:
            logger.error(f"Error getting brands: {e}")
            return []
    
    @st.cache_data(ttl=600)  # 10-minute cache
    def get_customers(_self, entity_name: Optional[str] = None) -> List[str]:
        """Get list of unique customers"""
        try:
            where_conditions = ["customer IS NOT NULL"]
            params = {}
            
            if entity_name:
                where_conditions.append("entity_name = :entity_name")
                params['entity_name'] = entity_name
            
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            query = f"""
            SELECT DISTINCT customer
            FROM unified_demand_view
            {where_clause}
            ORDER BY customer
            """
            
            with _self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                customers = [row[0] for row in result]
            
            return customers
            
        except Exception as e:
            logger.error(f"Error getting customers: {e}")
            return []