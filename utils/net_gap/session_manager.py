# utils/net_gap/session_manager.py

"""
Session State Manager for GAP Analysis System
Centralizes all session state management to prevent conflicts and ensure consistency
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional, List, Tuple
from datetime import date, datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SessionStateManager:
    """
    Centralized manager for all GAP analysis session state
    Prevents conflicts and provides type-safe access to state
    """
    
    # State key constants
    KEY_FILTERS = 'gap_filters'
    KEY_CURRENT_PAGE = 'current_page'
    KEY_SHOW_CUSTOMER_DIALOG = 'show_customer_dialog'
    KEY_DIALOG_PAGE = 'dlg_page'
    KEY_TABLE_COL_BASIC = 'table_col_basic'
    KEY_TABLE_COL_SUPPLY = 'table_col_supply'
    KEY_TABLE_COL_SAFETY = 'table_col_safety'
    KEY_TABLE_COL_ANALYSIS = 'table_col_analysis'
    KEY_TABLE_COL_FINANCIAL = 'table_col_financial'
    KEY_TABLE_COL_DETAILS = 'table_col_details'
    
    # Calculation state keys (track if GAP has been calculated)
    KEY_GAP_CALCULATED = 'gap_calculated'
    KEY_GAP_RESULTS = 'gap_results'
    KEY_GAP_METRICS = 'gap_metrics'
    KEY_CALCULATION_TIMESTAMP = 'calculation_timestamp'
    
    # Temporary dialog data keys (should be cleared after use)
    KEY_DIALOG_SHORTAGE_IDS = 'dialog_shortage_product_ids'
    KEY_DIALOG_METRICS = 'dialog_metrics'
    
    def __init__(self):
        """Initialize session state manager"""
        self._initialize_defaults()
    
    def _initialize_defaults(self) -> None:
        """Initialize default values for all state keys"""
        defaults = {
            self.KEY_FILTERS: self._get_default_filters(),
            self.KEY_CURRENT_PAGE: 1,
            self.KEY_SHOW_CUSTOMER_DIALOG: False,
            self.KEY_DIALOG_PAGE: 1,
            self.KEY_TABLE_COL_BASIC: True,
            self.KEY_TABLE_COL_SUPPLY: True,
            self.KEY_TABLE_COL_SAFETY: True,
            self.KEY_TABLE_COL_ANALYSIS: True,
            self.KEY_TABLE_COL_FINANCIAL: False,
            self.KEY_TABLE_COL_DETAILS: False,
            self.KEY_GAP_CALCULATED: False,
            self.KEY_GAP_RESULTS: None,
            self.KEY_GAP_METRICS: None,
            self.KEY_CALCULATION_TIMESTAMP: None,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _get_default_filters(self) -> Dict[str, Any]:
        """Get default filter configuration"""
        # Note: date_range should be set by filters module from actual data
        # Using None as placeholder to indicate it needs to be initialized
        return {
            'entity': None,
            'date_range': None,  # Will be set by filters from data
            'products': [],
            'brands': [],
            'customers': [],
            'quick_filter': 'all',
            'group_by': 'product',
            'supply_sources': ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER'],
            'demand_sources': ['OC_PENDING'],
            'include_safety_stock': True
        }
    
    # Filter Management
    def get_filters(self) -> Dict[str, Any]:
        """Get current filter values"""
        return st.session_state.get(self.KEY_FILTERS, self._get_default_filters())
    
    def set_filters(self, filters: Dict[str, Any]) -> None:
        """Set filter values"""
        st.session_state[self.KEY_FILTERS] = filters
        logger.debug(f"Filters updated: {self._get_filter_summary(filters)}")
    
    def reset_filters(self) -> None:
        """Reset filters to default values"""
        st.session_state[self.KEY_FILTERS] = self._get_default_filters()
        logger.info("Filters reset to defaults")
    
    def update_filter(self, key: str, value: Any) -> None:
        """Update a single filter value"""
        filters = self.get_filters()
        filters[key] = value
        self.set_filters(filters)
    
    def _get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """Generate summary of active filters"""
        active = []
        if filters.get('entity'):
            active.append(f"entity={filters['entity']}")
        if filters.get('products'):
            active.append(f"products={len(filters['products'])}")
        if filters.get('brands'):
            active.append(f"brands={len(filters['brands'])}")
        if filters.get('customers'):
            active.append(f"customers={len(filters['customers'])}")
        return ", ".join(active) if active else "no filters"
    
    # GAP Calculation State Management
    def is_gap_calculated(self) -> bool:
        """Check if GAP has been calculated"""
        return st.session_state.get(self.KEY_GAP_CALCULATED, False)
    
    def set_gap_calculated(
        self, 
        gap_df: pd.DataFrame, 
        metrics: Dict[str, Any],
        filters_used: Dict[str, Any]
    ) -> None:
        """
        Mark GAP as calculated and store results
        
        Args:
            gap_df: Calculated GAP dataframe
            metrics: Summary metrics
            filters_used: Filters used for this calculation
        """
        st.session_state[self.KEY_GAP_CALCULATED] = True
        st.session_state[self.KEY_GAP_RESULTS] = gap_df
        st.session_state[self.KEY_GAP_METRICS] = metrics
        st.session_state[self.KEY_CALCULATION_TIMESTAMP] = datetime.now()
        
        # Store filters hash to detect when filters change
        st.session_state['_gap_filters_hash'] = self._hash_filters(filters_used)
        
        logger.info(f"GAP calculation state saved: {len(gap_df)} items")
    
    def get_gap_results(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Get stored GAP results
        
        Returns:
            Tuple of (gap_df, metrics) or (None, None) if not calculated
        """
        if not self.is_gap_calculated():
            return None, None
        
        return (
            st.session_state.get(self.KEY_GAP_RESULTS),
            st.session_state.get(self.KEY_GAP_METRICS)
        )
    
    def clear_gap_calculation(self) -> None:
        """Clear GAP calculation state (force recalculation)"""
        st.session_state[self.KEY_GAP_CALCULATED] = False
        st.session_state[self.KEY_GAP_RESULTS] = None
        st.session_state[self.KEY_GAP_METRICS] = None
        st.session_state[self.KEY_CALCULATION_TIMESTAMP] = None
        if '_gap_filters_hash' in st.session_state:
            del st.session_state['_gap_filters_hash']
        logger.info("GAP calculation state cleared")
    
    def should_recalculate(self, current_filters: Dict[str, Any]) -> bool:
        """
        Check if GAP should be recalculated based on filter changes
        
        Args:
            current_filters: Current filter values
            
        Returns:
            True if recalculation needed
        """
        if not self.is_gap_calculated():
            return True
        
        # Check if filters have changed
        stored_hash = st.session_state.get('_gap_filters_hash')
        current_hash = self._hash_filters(current_filters)
        
        if stored_hash != current_hash:
            logger.info("Filters changed, recalculation needed")
            return True
        
        return False
    
    def _hash_filters(self, filters: Dict[str, Any]) -> str:
        """
        Create a hash of filters for comparison
        
        Args:
            filters: Filter dictionary
            
        Returns:
            Hash string
        """
        # Only hash the filters that affect calculation
        key_parts = [
            str(filters.get('entity')),
            str(filters.get('date_range')),
            str(sorted(filters.get('products', []))),
            str(sorted(filters.get('brands', []))),
            str(sorted(filters.get('customers', []))),
            str(sorted(filters.get('supply_sources', []))),
            str(sorted(filters.get('demand_sources', []))),
            str(filters.get('include_safety_stock', False)),
            str(filters.get('group_by', 'product'))
        ]
        return '|'.join(key_parts)
    
    # Pagination Management
    def get_current_page(self) -> int:
        """Get current page number"""
        return st.session_state.get(self.KEY_CURRENT_PAGE, 1)
    
    def set_current_page(self, page: int, total_pages: int) -> None:
        """
        Set current page with validation
        
        Args:
            page: Desired page number
            total_pages: Total number of pages available
        """
        # Validate and constrain page number
        validated_page = max(1, min(page, total_pages))
        st.session_state[self.KEY_CURRENT_PAGE] = validated_page
        
        if validated_page != page:
            logger.debug(f"Page number adjusted from {page} to {validated_page} (total: {total_pages})")
    
    def reset_pagination(self) -> None:
        """Reset pagination to first page"""
        st.session_state[self.KEY_CURRENT_PAGE] = 1
    
    def increment_page(self, total_pages: int) -> None:
        """Go to next page"""
        current = self.get_current_page()
        self.set_current_page(current + 1, total_pages)
    
    def decrement_page(self, total_pages: int) -> None:
        """Go to previous page"""
        current = self.get_current_page()
        self.set_current_page(current - 1, total_pages)
    
    def goto_first_page(self) -> None:
        """Go to first page"""
        st.session_state[self.KEY_CURRENT_PAGE] = 1
    
    def goto_last_page(self, total_pages: int) -> None:
        """Go to last page"""
        st.session_state[self.KEY_CURRENT_PAGE] = total_pages
    
    # Customer Dialog Management
    def show_customer_dialog(self) -> bool:
        """Check if customer dialog should be shown"""
        return st.session_state.get(self.KEY_SHOW_CUSTOMER_DIALOG, False)
    
    def open_customer_dialog(self, shortage_product_ids: List[int], metrics: Dict[str, Any]) -> None:
        """
        Open customer dialog with minimal data
        
        Args:
            shortage_product_ids: List of product IDs with shortages
            metrics: Pre-calculated metrics to display
        """
        st.session_state[self.KEY_SHOW_CUSTOMER_DIALOG] = True
        st.session_state[self.KEY_DIALOG_SHORTAGE_IDS] = shortage_product_ids
        st.session_state[self.KEY_DIALOG_METRICS] = metrics
        st.session_state[self.KEY_DIALOG_PAGE] = 1
        logger.info(f"Customer dialog opened for {len(shortage_product_ids)} products")
    
    def close_customer_dialog(self) -> None:
        """Close customer dialog and cleanup state"""
        # Clear dialog state
        keys_to_clear = [
            self.KEY_SHOW_CUSTOMER_DIALOG,
            self.KEY_DIALOG_SHORTAGE_IDS,
            self.KEY_DIALOG_METRICS,
            self.KEY_DIALOG_PAGE,
            'dlg_search',
            'dlg_size'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        logger.info("Customer dialog closed and state cleared")
    
    def get_dialog_data(self) -> Tuple[Optional[List[int]], Optional[Dict[str, Any]]]:
        """
        Get dialog data (shortage IDs and metrics)
        
        Returns:
            Tuple of (shortage_product_ids, metrics)
        """
        shortage_ids = st.session_state.get(self.KEY_DIALOG_SHORTAGE_IDS)
        metrics = st.session_state.get(self.KEY_DIALOG_METRICS)
        return shortage_ids, metrics
    
    def get_dialog_page(self) -> int:
        """Get current dialog page"""
        return st.session_state.get(self.KEY_DIALOG_PAGE, 1)
    
    def set_dialog_page(self, page: int, total_pages: int) -> None:
        """Set dialog page with validation"""
        validated_page = max(1, min(page, total_pages))
        st.session_state[self.KEY_DIALOG_PAGE] = validated_page
    
    # Table Column Configuration
    def get_table_columns_config(self) -> Dict[str, bool]:
        """Get current table column configuration"""
        return {
            'basic': st.session_state.get(self.KEY_TABLE_COL_BASIC, True),
            'supply': st.session_state.get(self.KEY_TABLE_COL_SUPPLY, True),
            'safety': st.session_state.get(self.KEY_TABLE_COL_SAFETY, True),
            'analysis': st.session_state.get(self.KEY_TABLE_COL_ANALYSIS, True),
            'financial': st.session_state.get(self.KEY_TABLE_COL_FINANCIAL, False),
            'details': st.session_state.get(self.KEY_TABLE_COL_DETAILS, False),
        }
    
    def set_table_columns_config(self, config: Dict[str, bool]) -> None:
        """Set table column configuration"""
        st.session_state[self.KEY_TABLE_COL_BASIC] = config.get('basic', True)
        st.session_state[self.KEY_TABLE_COL_SUPPLY] = config.get('supply', True)
        st.session_state[self.KEY_TABLE_COL_SAFETY] = config.get('safety', True)
        st.session_state[self.KEY_TABLE_COL_ANALYSIS] = config.get('analysis', True)
        st.session_state[self.KEY_TABLE_COL_FINANCIAL] = config.get('financial', False)
        st.session_state[self.KEY_TABLE_COL_DETAILS] = config.get('details', False)
    
    def apply_table_preset(self, preset: str) -> None:
        """
        Apply predefined table column presets
        
        Args:
            preset: One of 'standard', 'safety', 'financial', 'all'
        """
        presets = {
            'standard': {
                'basic': True,
                'supply': True,
                'safety': False,
                'analysis': True,
                'financial': False,
                'details': False
            },
            'safety': {
                'basic': True,
                'supply': False,
                'safety': True,
                'analysis': True,
                'financial': False,
                'details': False
            },
            'financial': {
                'basic': True,
                'supply': True,
                'safety': False,
                'analysis': False,
                'financial': True,
                'details': False
            },
            'all': {
                'basic': True,
                'supply': True,
                'safety': True,
                'analysis': True,
                'financial': True,
                'details': True
            }
        }
        
        if preset in presets:
            self.set_table_columns_config(presets[preset])
            logger.debug(f"Applied table preset: {preset}")
        else:
            logger.warning(f"Unknown table preset: {preset}")
    
    # Utility Methods
    def clear_all(self) -> None:
        """Clear all GAP analysis state (use with caution)"""
        keys_to_clear = [
            self.KEY_FILTERS,
            self.KEY_CURRENT_PAGE,
            self.KEY_SHOW_CUSTOMER_DIALOG,
            self.KEY_DIALOG_PAGE,
            self.KEY_DIALOG_SHORTAGE_IDS,
            self.KEY_DIALOG_METRICS,
            self.KEY_TABLE_COL_BASIC,
            self.KEY_TABLE_COL_SUPPLY,
            self.KEY_TABLE_COL_SAFETY,
            self.KEY_TABLE_COL_ANALYSIS,
            self.KEY_TABLE_COL_FINANCIAL,
            self.KEY_TABLE_COL_DETAILS,
            self.KEY_GAP_CALCULATED,
            self.KEY_GAP_RESULTS,
            self.KEY_GAP_METRICS,
            self.KEY_CALCULATION_TIMESTAMP,
            '_gap_filters_hash'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        logger.info("All GAP analysis state cleared")
        self._initialize_defaults()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state for debugging"""
        gap_calculated = self.is_gap_calculated()
        calc_time = st.session_state.get(self.KEY_CALCULATION_TIMESTAMP)
        
        return {
            'filters_active': self._get_filter_summary(self.get_filters()),
            'current_page': self.get_current_page(),
            'dialog_open': self.show_customer_dialog(),
            'table_config': self.get_table_columns_config(),
            'gap_calculated': gap_calculated,
            'calculation_time': calc_time.strftime('%Y-%m-%d %H:%M:%S') if calc_time else None,
            'results_count': len(st.session_state.get(self.KEY_GAP_RESULTS, [])) if gap_calculated else 0
        }
    
    def log_state(self) -> None:
        """Log current state for debugging"""
        summary = self.get_state_summary()
        logger.info(f"Session State: {summary}")


# Singleton instance
_session_manager = None

def get_session_manager() -> SessionStateManager:
    """
    Get or create the singleton SessionStateManager instance
    
    Returns:
        SessionStateManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionStateManager()
    return _session_manager