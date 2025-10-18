# utils/net_gap/state.py

"""
Simplified state management for GAP Analysis
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GAPState:
    """Simple state manager - no over-engineering"""
    
    # State keys
    KEY_FILTERS = 'gap_filters'
    KEY_RESULT = 'gap_result'
    KEY_PAGE = 'current_page'
    KEY_DIALOG_PAGE = 'dialog_page'
    KEY_TABLE_PRESET = 'table_preset'
    
    def __init__(self):
        self._init_defaults()
    
    def _init_defaults(self):
        """Initialize default values"""
        if self.KEY_FILTERS not in st.session_state:
            st.session_state[self.KEY_FILTERS] = self.get_default_filters()
        if self.KEY_RESULT not in st.session_state:
            st.session_state[self.KEY_RESULT] = None
        if self.KEY_PAGE not in st.session_state:
            st.session_state[self.KEY_PAGE] = 1
        if self.KEY_DIALOG_PAGE not in st.session_state:
            st.session_state[self.KEY_DIALOG_PAGE] = 1
        if self.KEY_TABLE_PRESET not in st.session_state:
            st.session_state[self.KEY_TABLE_PRESET] = 'standard'
    
    @staticmethod
    def get_default_filters() -> Dict[str, Any]:
        """Get default filter configuration"""
        return {
            'entity': None,
            'exclude_entity': False,
            'products': [],
            'exclude_products': False,
            'brands': [],
            'exclude_brands': False,
            'exclude_expired': True,
            'group_by': 'product',
            'supply_sources': ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER'],
            'demand_sources': ['OC_PENDING'],
            'include_safety': True
        }
    
    # Filters
    def get_filters(self) -> Dict[str, Any]:
        """Get current filters"""
        return st.session_state.get(self.KEY_FILTERS, self.get_default_filters())
    
    def set_filters(self, filters: Dict[str, Any]):
        """Set filters and check if changed"""
        current = self.get_filters()
        
        # Check if filters actually changed (simple comparison)
        changed = False
        for key in ['entity', 'products', 'brands', 'supply_sources', 'demand_sources', 
                   'include_safety', 'group_by', 'exclude_expired']:
            if filters.get(key) != current.get(key):
                changed = True
                break
        
        st.session_state[self.KEY_FILTERS] = filters
        
        # Clear result if filters changed
        if changed:
            st.session_state[self.KEY_RESULT] = None
            st.session_state[self.KEY_PAGE] = 1
            logger.info("Filters changed, cleared result")
    
    def reset_filters(self):
        """Reset to default filters"""
        st.session_state[self.KEY_FILTERS] = self.get_default_filters()
        st.session_state[self.KEY_RESULT] = None
        st.session_state[self.KEY_PAGE] = 1
    
    # GAP Result
    def get_result(self):
        """Get calculation result"""
        return st.session_state.get(self.KEY_RESULT)
    
    def set_result(self, result):
        """Store calculation result"""
        st.session_state[self.KEY_RESULT] = result
        logger.info(f"Stored result: {len(result.gap_df)} items")
    
    def has_result(self) -> bool:
        """Check if result exists"""
        return st.session_state.get(self.KEY_RESULT) is not None
    
    def should_recalculate(self) -> bool:
        """Check if recalculation needed"""
        return st.session_state.get(self.KEY_RESULT) is None
    
    # Pagination
    def get_page(self) -> int:
        """Get current page"""
        return st.session_state.get(self.KEY_PAGE, 1)
    
    def set_page(self, page: int, max_page: int):
        """Set page with validation"""
        page = max(1, min(page, max_page))
        st.session_state[self.KEY_PAGE] = page
    
    def get_dialog_page(self) -> int:
        """Get dialog page"""
        return st.session_state.get(self.KEY_DIALOG_PAGE, 1)
    
    def set_dialog_page(self, page: int, max_page: int):
        """Set dialog page"""
        page = max(1, min(page, max_page))
        st.session_state[self.KEY_DIALOG_PAGE] = page
    
    # Table Preset
    def get_table_preset(self) -> str:
        """Get current table preset"""
        return st.session_state.get(self.KEY_TABLE_PRESET, 'standard')
    
    def set_table_preset(self, preset: str):
        """Set table preset"""
        st.session_state[self.KEY_TABLE_PRESET] = preset
    
    # Utility
    def clear_all(self):
        """Clear all state"""
        for key in [self.KEY_FILTERS, self.KEY_RESULT, self.KEY_PAGE, 
                   self.KEY_DIALOG_PAGE, self.KEY_TABLE_PRESET]:
            if key in st.session_state:
                del st.session_state[key]
        self._init_defaults()
        logger.info("All state cleared")


# Singleton instance
_state = None

def get_state() -> GAPState:
    """Get or create state manager"""
    global _state
    if _state is None:
        _state = GAPState()
    return _state