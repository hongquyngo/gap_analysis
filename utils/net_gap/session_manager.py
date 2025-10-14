# utils/net_gap/session_manager.py

"""
Session State Manager - Version 3.0 REFACTORED
- Stores GAPCalculationResult (single object)
- Auto-resets pagination on filter changes
- Simplified dialog state management
- Removed temporary state keys
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SessionStateManager:
    """Centralized session state manager for GAP analysis"""
    
    # State keys
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
    
    # Calculation result key (single object)
    KEY_CALCULATION_RESULT = 'gap_calculation_result'
    
    def __init__(self):
        self._initialize_defaults()
    
    def _initialize_defaults(self) -> None:
        """Initialize default values"""
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
            self.KEY_CALCULATION_RESULT: None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _get_default_filters(self) -> Dict[str, Any]:
        """Get default filter configuration"""
        return {
            'entity': None,
            'products': [],
            'brands': [],
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
        """Reset filters to defaults"""
        st.session_state[self.KEY_FILTERS] = self._get_default_filters()
        logger.info("Filters reset to defaults")
    
    def _get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """Generate summary of active filters"""
        active = []
        if filters.get('entity'):
            active.append(f"entity={filters['entity']}")
        if filters.get('products'):
            active.append(f"products={len(filters['products'])}")
        if filters.get('brands'):
            active.append(f"brands={len(filters['brands'])}")
        return ", ".join(active) if active else "no filters"
    
    # GAP Calculation Result Management (Single Object)
    def is_gap_calculated(self) -> bool:
        """Check if GAP has been calculated"""
        result = st.session_state.get(self.KEY_CALCULATION_RESULT)
        return result is not None
    
    def set_gap_calculated(self, result) -> None:
        """
        Store complete GAP calculation result
        
        Args:
            result: GAPCalculationResult object
        """
        st.session_state[self.KEY_CALCULATION_RESULT] = result
        logger.info(f"GAP calculation stored: {result.to_summary_dict()}")
    
    def get_gap_result(self):
        """
        Get stored GAP calculation result
        
        Returns:
            GAPCalculationResult or None
        """
        return st.session_state.get(self.KEY_CALCULATION_RESULT)
    
    def clear_gap_calculation(self) -> None:
        """Clear GAP calculation (force recalculation)"""
        st.session_state[self.KEY_CALCULATION_RESULT] = None
        self.reset_pagination()
        logger.info("GAP calculation cleared")
    
    def should_recalculate(self, current_filters: Dict[str, Any]) -> bool:
        """
        Check if recalculation needed based on filter changes
        Auto-resets pagination if filters changed
        """
        result = self.get_gap_result()
        if result is None:
            return True
        
        current_hash = self._hash_filters(current_filters)
        stored_hash = result.get_filter_hash()
        
        if current_hash != stored_hash:
            logger.info("Filters changed, recalculation needed")
            self.reset_pagination()  # Auto-reset pagination
            return True
        
        return False
    
    def _hash_filters(self, filters: Dict[str, Any]) -> str:
        """Create hash of filters for comparison"""
        key_parts = [
            str(filters.get('entity')),
            str(sorted(filters.get('products', []))),
            str(sorted(filters.get('brands', []))),
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
        """Set current page with validation"""
        validated_page = max(1, min(page, total_pages))
        st.session_state[self.KEY_CURRENT_PAGE] = validated_page
        
        if validated_page != page:
            logger.debug(f"Page adjusted: {page} -> {validated_page} (total: {total_pages})")
    
    def reset_pagination(self) -> None:
        """Reset pagination to first page"""
        st.session_state[self.KEY_CURRENT_PAGE] = 1
        logger.debug("Pagination reset to page 1")
    
    # Customer Dialog Management (Simplified)
    def show_customer_dialog(self) -> bool:
        """Check if customer dialog should be shown"""
        return st.session_state.get(self.KEY_SHOW_CUSTOMER_DIALOG, False)
    
    def open_customer_dialog(self) -> None:
        """Open customer dialog (data from result object)"""
        st.session_state[self.KEY_SHOW_CUSTOMER_DIALOG] = True
        st.session_state[self.KEY_DIALOG_PAGE] = 1
        logger.info("Customer dialog opened")
    
    def close_customer_dialog(self) -> None:
        """Close customer dialog"""
        st.session_state[self.KEY_SHOW_CUSTOMER_DIALOG] = False
        st.session_state[self.KEY_DIALOG_PAGE] = 1
        logger.info("Customer dialog closed")
    
    def get_dialog_page(self) -> int:
        """Get current dialog page"""
        return st.session_state.get(self.KEY_DIALOG_PAGE, 1)
    
    def set_dialog_page(self, page: int, total_pages: int) -> None:
        """Set dialog page with validation"""
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
        """Set table column configuration"""
        st.session_state[self.KEY_TABLE_COL_BASIC] = config.get('basic', True)
        st.session_state[self.KEY_TABLE_COL_SUPPLY] = config.get('supply', True)
        st.session_state[self.KEY_TABLE_COL_SAFETY] = config.get('safety', True)
        st.session_state[self.KEY_TABLE_COL_ANALYSIS] = config.get('analysis', True)
        st.session_state[self.KEY_TABLE_COL_FINANCIAL] = config.get('financial', False)
        st.session_state[self.KEY_TABLE_COL_DETAILS] = config.get('details', False)
    
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
    
    # Utility Methods
    def clear_all(self) -> None:
        """Clear all GAP analysis state"""
        keys_to_clear = [
            self.KEY_FILTERS, self.KEY_CURRENT_PAGE,
            self.KEY_SHOW_CUSTOMER_DIALOG, self.KEY_DIALOG_PAGE,
            self.KEY_CALCULATION_RESULT
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        logger.info("All GAP state cleared")
        self._initialize_defaults()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary for debugging"""
        result = self.get_gap_result()
        
        return {
            'filters_active': self._get_filter_summary(self.get_filters()),
            'current_page': self.get_current_page(),
            'dialog_open': self.show_customer_dialog(),
            'table_config': self.get_table_columns_config(),
            'has_result': result is not None,
            'calculation_time': result.timestamp.isoformat() if result else None,
            'items_count': len(result.gap_df) if result else 0,
            'affected_customers': result.metrics.get('affected_customers', 0) if result else 0
        }


# Singleton instance
_session_manager = None

def get_session_manager() -> SessionStateManager:
    """Get or create singleton SessionStateManager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionStateManager()
    return _session_manager