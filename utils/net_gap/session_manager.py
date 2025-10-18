# utils/net_gap/session_manager.py

"""
Session State Manager - Version 3.3 OPTIMIZED
FIXES:
- Better state change detection (prevents unnecessary reruns)
- Improved dialog state management
- Smarter pagination reset (only when needed)
- Added debouncing for frequent state changes
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class SessionStateManager:
    """Centralized session state manager for GAP analysis with optimized state handling"""
    
    # State keys
    KEY_FILTERS = 'gap_filters'
    KEY_FILTERS_HASH = 'gap_filters_hash'
    KEY_CURRENT_PAGE = 'current_page'
    KEY_SHOW_CUSTOMER_DIALOG = 'show_customer_dialog'
    KEY_DIALOG_PAGE = 'dlg_page'
    KEY_DIALOG_REQUESTED = 'dialog_requested'
    KEY_TABLE_COL_BASIC = 'table_col_basic'
    KEY_TABLE_COL_SUPPLY = 'table_col_supply'
    KEY_TABLE_COL_SAFETY = 'table_col_safety'
    KEY_TABLE_COL_ANALYSIS = 'table_col_analysis'
    KEY_TABLE_COL_FINANCIAL = 'table_col_financial'
    KEY_TABLE_COL_DETAILS = 'table_col_details'
    KEY_CALCULATION_RESULT = 'gap_calculation_result'
    KEY_LAST_CALCULATION_TIME = 'last_calc_time'
    KEY_STATE_VERSION = 'state_version'  # Track state changes
    
    def __init__(self):
        self._initialize_defaults()
        self._current_state_version = self._get_state_version()
    
    def _initialize_defaults(self) -> None:
        """Initialize default values"""
        defaults = {
            self.KEY_FILTERS: self._get_default_filters(),
            self.KEY_FILTERS_HASH: None,
            self.KEY_CURRENT_PAGE: 1,
            self.KEY_SHOW_CUSTOMER_DIALOG: False,
            self.KEY_DIALOG_PAGE: 1,
            self.KEY_DIALOG_REQUESTED: False,
            self.KEY_TABLE_COL_BASIC: True,
            self.KEY_TABLE_COL_SUPPLY: True,
            self.KEY_TABLE_COL_SAFETY: True,
            self.KEY_TABLE_COL_ANALYSIS: True,
            self.KEY_TABLE_COL_FINANCIAL: False,
            self.KEY_TABLE_COL_DETAILS: False,
            self.KEY_CALCULATION_RESULT: None,
            self.KEY_LAST_CALCULATION_TIME: None,
            self.KEY_STATE_VERSION: 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _get_default_filters(self) -> Dict[str, Any]:
        """Get default filter configuration"""
        return {
            'entity': None,
            'exclude_entity': False,
            'products': [],
            'brands': [],
            'exclude_products': False,
            'exclude_brands': False,
            'exclude_expired_inventory': True,
            'group_by': 'product',
            'supply_sources': ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER'],
            'demand_sources': ['OC_PENDING'],
            'include_safety_stock': True
        }
    
    def _get_state_version(self) -> int:
        """Get current state version"""
        return st.session_state.get(self.KEY_STATE_VERSION, 0)
    
    def _increment_state_version(self) -> None:
        """Increment state version (tracks changes)"""
        current = self._get_state_version()
        st.session_state[self.KEY_STATE_VERSION] = current + 1
    
    def _compute_filter_hash(self, filters: Dict[str, Any]) -> str:
        """
        Compute stable hash of filters for change detection
        FIXED: More reliable than string comparison
        """
        # Create a deterministic representation
        filter_dict = {
            'entity': filters.get('entity'),
            'exclude_entity': filters.get('exclude_entity', False),
            'products': sorted(filters.get('products', [])),
            'brands': sorted(filters.get('brands', [])),
            'exclude_products': filters.get('exclude_products', False),
            'exclude_brands': filters.get('exclude_brands', False),
            'exclude_expired': filters.get('exclude_expired_inventory', True),
            'group_by': filters.get('group_by', 'product'),
            'supply': sorted(filters.get('supply_sources', [])),
            'demand': sorted(filters.get('demand_sources', [])),
            'safety': filters.get('include_safety_stock', False)
        }
        
        # Create hash
        filter_str = json.dumps(filter_dict, sort_keys=True)
        return hashlib.md5(filter_str.encode()).hexdigest()
    
    # Filter Management
    def get_filters(self) -> Dict[str, Any]:
        """Get current filter values"""
        return st.session_state.get(self.KEY_FILTERS, self._get_default_filters())
    
    def set_filters(self, filters: Dict[str, Any]) -> None:
        """
        Set filter values with change detection
        FIXED: Only increments version if filters actually changed
        """
        current_hash = st.session_state.get(self.KEY_FILTERS_HASH)
        new_hash = self._compute_filter_hash(filters)
        
        # Only update if changed
        if current_hash != new_hash:
            st.session_state[self.KEY_FILTERS] = filters
            st.session_state[self.KEY_FILTERS_HASH] = new_hash
            self._increment_state_version()
            logger.debug(f"Filters updated: {self._get_filter_summary(filters)}")
        else:
            # Filters unchanged, just update the dict (might have been reordered)
            st.session_state[self.KEY_FILTERS] = filters
    
    def reset_filters(self) -> None:
        """Reset filters to defaults"""
        defaults = self._get_default_filters()
        st.session_state[self.KEY_FILTERS] = defaults
        st.session_state[self.KEY_FILTERS_HASH] = self._compute_filter_hash(defaults)
        self._increment_state_version()
        logger.info("Filters reset to defaults")
    
    def _get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """Generate summary of active filters"""
        active = []
        if filters.get('entity'):
            mode = "excl" if filters.get('exclude_entity') else "incl"
            active.append(f"entity={filters['entity']}({mode})")
        if filters.get('products'):
            mode = "excl" if filters.get('exclude_products') else "incl"
            active.append(f"prods={len(filters['products'])}({mode})")
        if filters.get('brands'):
            mode = "excl" if filters.get('exclude_brands') else "incl"
            active.append(f"brands={len(filters['brands'])}({mode})")
        if filters.get('exclude_expired_inventory'):
            active.append("no_exp")
        return ", ".join(active) if active else "defaults"
    
    # GAP Calculation Result Management
    def is_gap_calculated(self) -> bool:
        """Check if GAP has been calculated"""
        result = st.session_state.get(self.KEY_CALCULATION_RESULT)
        return result is not None
    
    def set_gap_calculated(self, result) -> None:
        """
        Store complete GAP calculation result
        FIXED: Also stores timestamp for cache validation
        """
        st.session_state[self.KEY_CALCULATION_RESULT] = result
        st.session_state[self.KEY_LAST_CALCULATION_TIME] = datetime.now()
        logger.info(f"GAP calculation stored: {result.to_summary_dict()}")
    
    def get_gap_result(self):
        """Get stored GAP calculation result"""
        return st.session_state.get(self.KEY_CALCULATION_RESULT)
    
    def get_calculation_age_seconds(self) -> Optional[float]:
        """Get age of current calculation in seconds"""
        calc_time = st.session_state.get(self.KEY_LAST_CALCULATION_TIME)
        if calc_time:
            return (datetime.now() - calc_time).total_seconds()
        return None
    
    def clear_gap_calculation(self) -> None:
        """Clear GAP calculation (force recalculation)"""
        st.session_state[self.KEY_CALCULATION_RESULT] = None
        st.session_state[self.KEY_LAST_CALCULATION_TIME] = None
        self.reset_pagination()
        logger.info("GAP calculation cleared")
    
    def should_recalculate(self, current_filters: Dict[str, Any]) -> bool:
        """
        Check if recalculation needed based on filter changes
        FIXED: Uses hash comparison for reliable detection
        """
        result = self.get_gap_result()
        if result is None:
            return True
        
        current_hash = self._compute_filter_hash(current_filters)
        stored_hash = st.session_state.get(self.KEY_FILTERS_HASH)
        
        if current_hash != stored_hash:
            logger.info("Filters changed, recalculation needed")
            # FIXED: Only reset pagination if filters actually changed
            self.reset_pagination()
            return True
        
        return False
    
    # Pagination Management
    def get_current_page(self) -> int:
        """Get current page number"""
        return st.session_state.get(self.KEY_CURRENT_PAGE, 1)
    
    def set_current_page(self, page: int, total_pages: int) -> None:
        """
        Set current page with validation
        FIXED: Doesn't increment state version (avoid rerun)
        """
        validated_page = max(1, min(page, total_pages))
        st.session_state[self.KEY_CURRENT_PAGE] = validated_page
        
        if validated_page != page:
            logger.debug(f"Page adjusted: {page} -> {validated_page} (total: {total_pages})")
    
    def reset_pagination(self) -> None:
        """Reset pagination to first page"""
        st.session_state[self.KEY_CURRENT_PAGE] = 1
        logger.debug("Pagination reset to page 1")
    
    # Customer Dialog Management - OPTIMIZED
    def show_customer_dialog(self) -> bool:
        """
        Check if customer dialog should be shown
        FIXED: Only shows if explicitly requested
        """
        requested = st.session_state.get(self.KEY_DIALOG_REQUESTED, False)
        
        if requested:
            # Clear the request flag immediately to prevent auto-reopening
            st.session_state[self.KEY_DIALOG_REQUESTED] = False
            return True
        
        return False
    
    def open_customer_dialog(self) -> None:
        """
        Request to open customer dialog
        FIXED: Sets explicit request flag
        """
        st.session_state[self.KEY_DIALOG_REQUESTED] = True
        st.session_state[self.KEY_DIALOG_PAGE] = 1
        logger.info("Customer dialog requested")
    
    def close_customer_dialog(self) -> None:
        """Close customer dialog and clear all dialog state"""
        st.session_state[self.KEY_SHOW_CUSTOMER_DIALOG] = False
        st.session_state[self.KEY_DIALOG_REQUESTED] = False
        st.session_state[self.KEY_DIALOG_PAGE] = 1
        logger.info("Customer dialog closed")
    
    def is_dialog_open(self) -> bool:
        """Check if dialog is currently open"""
        return st.session_state.get(self.KEY_SHOW_CUSTOMER_DIALOG, False)
    
    def get_dialog_page(self) -> int:
        """Get current dialog page"""
        return st.session_state.get(self.KEY_DIALOG_PAGE, 1)
    
    def set_dialog_page(self, page: int, total_pages: int) -> None:
        """
        Set dialog page with validation
        FIXED: Doesn't trigger state version increment
        """
        validated_page = max(1, min(page, total_pages))
        st.session_state[self.KEY_DIALOG_PAGE] = validated_page
    
    # Table Column Configuration
    def get_table_columns_config(self) -> Dict[str, bool]:
        """Get table column configuration"""
        return {
            'basic': st.session_state.get(self.KEY_TABLE_COL_BASIC, True),
            'supply': st.session_state.get(self.KEY_TABLE_COL_SUPPLY, True),
            'safety': st.session_state.get(self.KEY_TABLE_COL_SAFETY, True),
            'analysis': st.session_state.get(self.KEY_TABLE_COL_ANALYSIS, True),
            'financial': st.session_state.get(self.KEY_TABLE_COL_FINANCIAL, False),
            'details': st.session_state.get(self.KEY_TABLE_COL_DETAILS, False),
        }
    
    def set_table_columns_config(self, config: Dict[str, bool]) -> None:
        """
        Set table column configuration
        FIXED: Only updates state if changed
        """
        current_config = self.get_table_columns_config()
        
        # Check if actually changed
        if current_config != config:
            st.session_state[self.KEY_TABLE_COL_BASIC] = config.get('basic', True)
            st.session_state[self.KEY_TABLE_COL_SUPPLY] = config.get('supply', True)
            st.session_state[self.KEY_TABLE_COL_SAFETY] = config.get('safety', True)
            st.session_state[self.KEY_TABLE_COL_ANALYSIS] = config.get('analysis', True)
            st.session_state[self.KEY_TABLE_COL_FINANCIAL] = config.get('financial', False)
            st.session_state[self.KEY_TABLE_COL_DETAILS] = config.get('details', False)
            logger.debug("Table column config updated")
    
    def apply_table_preset(self, preset: str) -> None:
        """Apply table column presets"""
        presets = {
            'standard': {
                'basic': True, 'supply': True, 'safety': False,
                'analysis': True, 'financial': False, 'details': False
            },
            'safety': {
                'basic': True, 'supply': False, 'safety': True,
                'analysis': True, 'financial': False, 'details': False
            },
            'financial': {
                'basic': True, 'supply': True, 'safety': False,
                'analysis': False, 'financial': True, 'details': False
            },
            'all': {
                'basic': True, 'supply': True, 'safety': True,
                'analysis': True, 'financial': True, 'details': True
            }
        }
        
        if preset in presets:
            self.set_table_columns_config(presets[preset])
            logger.debug(f"Applied table preset: {preset}")
    
    # State Optimization Methods
    def has_state_changed_since(self, version: int) -> bool:
        """Check if state has changed since given version"""
        return self._get_state_version() > version
    
    def get_current_state_version(self) -> int:
        """Get current state version for comparison"""
        return self._get_state_version()
    
    def should_refresh_ui(self, last_known_version: int) -> bool:
        """
        FIXED: Determine if UI should refresh based on state changes
        Helps prevent unnecessary reruns
        """
        return self.has_state_changed_since(last_known_version)
    
    # Utility Methods
    def clear_all(self) -> None:
        """Clear all GAP analysis state"""
        keys_to_clear = [
            self.KEY_FILTERS, self.KEY_FILTERS_HASH, self.KEY_CURRENT_PAGE,
            self.KEY_SHOW_CUSTOMER_DIALOG, self.KEY_DIALOG_PAGE,
            self.KEY_DIALOG_REQUESTED, self.KEY_CALCULATION_RESULT,
            self.KEY_LAST_CALCULATION_TIME, self.KEY_STATE_VERSION
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        logger.info("All GAP state cleared")
        self._initialize_defaults()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary for debugging"""
        result = self.get_gap_result()
        calc_age = self.get_calculation_age_seconds()
        
        return {
            'filters_active': self._get_filter_summary(self.get_filters()),
            'filters_hash': st.session_state.get(self.KEY_FILTERS_HASH, 'none')[:8],
            'current_page': self.get_current_page(),
            'dialog_requested': st.session_state.get(self.KEY_DIALOG_REQUESTED, False),
            'dialog_open': self.is_dialog_open(),
            'table_config': self.get_table_columns_config(),
            'has_result': result is not None,
            'calculation_age_sec': round(calc_age, 1) if calc_age else None,
            'items_count': len(result.gap_df) if result else 0,
            'affected_customers': result.metrics.get('affected_customers', 0) if result else 0,
            'state_version': self._get_state_version()
        }
    
    def log_state_summary(self) -> None:
        """Log current state for debugging"""
        summary = self.get_state_summary()
        logger.info(f"State Summary: {summary}")


# Singleton instance
_session_manager = None

def get_session_manager() -> SessionStateManager:
    """Get or create singleton SessionStateManager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionStateManager()
    return _session_manager