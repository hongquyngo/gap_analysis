# utils/period_gap/session_state.py
"""
Complete Session State Management for Period GAP Analysis
Includes cross-page data persistence
"""

import streamlit as st
from typing import Any
from datetime import datetime

def initialize_session_state():
    """Initialize all required session state variables for Period GAP"""
    
    # Core session state defaults
    defaults = {
        # Data loading status
        'period_gap_data_loaded': False,
        'period_gap_load_time': None,
        
        # Analysis status
        'period_gap_analysis_ran': False,
        'period_gap_analysis_data': None,
        'period_gap_result': None,
        
        # Cross-page data sharing (CRITICAL)
        'gap_analysis_result': None,
        'demand_filtered': None,
        'supply_filtered': None,
        'last_gap_analysis': None,
        'last_analysis_time': None,
        
        # Filter cache
        'period_gap_filter_entities': [],
        'period_gap_filter_products': [],
        'period_gap_filter_brands': [],
        'period_gap_filter_customers': [],
        
        # Filter data initialization
        'pgap_filter_data': None,
        'pgap_temp_demand': None,
        'pgap_temp_supply': None,
        
        # Calculation options cache
        'period_gap_period_type': 'Weekly',
        'period_gap_track_backlog': True,
        'period_gap_exclude_missing_dates': True,
        
        # Display options cache
        'period_gap_show_matched': True,
        'period_gap_show_demand_only': True,
        'period_gap_show_supply_only': True,
        'period_gap_period_filter': 'All',
        'period_gap_enable_row_highlighting': False,
        
        # GAP calculation cache
        'pgap_gap_df': None,
        'pgap_result_cache_key': None,
        
        # Debug mode
        'period_gap_debug_mode': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Safely get value from session state
    
    Args:
        key: Session state key
        default: Default value if key not found
    
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any):
    """
    Safely set value in session state
    
    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def clear_period_gap_cache():
    """Clear Period GAP analysis cache but preserve filter data"""
    cache_keys = [
        'period_gap_data_loaded',
        'period_gap_load_time',
        'period_gap_analysis_ran',
        'period_gap_analysis_data',
        'period_gap_result',
        'pgap_gap_df',
        'pgap_result_cache_key'
    ]
    
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]


def save_period_gap_state(data: dict):
    """
    Save Period GAP analysis state for display and cross-page access
    
    Args:
        data: Dictionary with analysis results
    """
    st.session_state['period_gap_analysis_data'] = data
    st.session_state['period_gap_analysis_ran'] = True
    st.session_state['period_gap_load_time'] = datetime.now()
    
    # Also save for cross-page access
    if 'demand' in data:
        st.session_state['demand_filtered'] = data['demand']
    if 'supply' in data:
        st.session_state['supply_filtered'] = data['supply']


def get_period_gap_state() -> dict:
    """
    Get Period GAP analysis state
    
    Returns:
        Dictionary with cached analysis data or empty dict
    """
    if st.session_state.get('period_gap_analysis_ran', False):
        return st.session_state.get('period_gap_analysis_data', {})
    return {}


def update_filter_cache(entities: list, products: list, brands: list, customers: list):
    """
    Update filter options cache for dropdowns
    
    Args:
        entities: List of entities
        products: List of products
        brands: List of brands
        customers: List of customers
    """
    st.session_state['period_gap_filter_entities'] = entities or []
    st.session_state['period_gap_filter_products'] = products or []
    st.session_state['period_gap_filter_brands'] = brands or []
    st.session_state['period_gap_filter_customers'] = customers or []


def get_filter_cache() -> dict:
    """
    Get cached filter options
    
    Returns:
        Dictionary with filter options
    """
    return {
        'entities': st.session_state.get('period_gap_filter_entities', []),
        'products': st.session_state.get('period_gap_filter_products', []),
        'brands': st.session_state.get('period_gap_filter_brands', []),
        'customers': st.session_state.get('period_gap_filter_customers', [])
    }


def is_gap_analysis_available() -> bool:
    """
    Check if GAP analysis results are available for other pages
    
    Returns:
        True if GAP analysis has been run and results are available
    """
    return (
        st.session_state.get('gap_analysis_result') is not None and
        st.session_state.get('demand_filtered') is not None and
        st.session_state.get('supply_filtered') is not None
    )


def get_gap_analysis_for_allocation() -> dict:
    """
    Get GAP analysis data formatted for Allocation Plan page
    
    Returns:
        Dictionary with gap_df, demand_df, supply_df or None values
    """
    if is_gap_analysis_available():
        return {
            'gap_df': st.session_state.get('gap_analysis_result'),
            'demand_df': st.session_state.get('demand_filtered'),
            'supply_df': st.session_state.get('supply_filtered'),
            'period_type': st.session_state.get('period_gap_period_type', 'Weekly'),
            'analysis_time': st.session_state.get('last_analysis_time')
        }
    return {
        'gap_df': None,
        'demand_df': None,
        'supply_df': None,
        'period_type': 'Weekly',
        'analysis_time': None
    }


def get_gap_analysis_for_po_suggestions() -> dict:
    """
    Get GAP analysis data formatted for PO Suggestions page
    
    Returns:
        Dictionary with shortage products and analysis metadata
    """
    gap_df = st.session_state.get('gap_analysis_result')
    
    if gap_df is not None and not gap_df.empty:
        # Get shortage products
        shortage_df = gap_df[gap_df['gap_quantity'] < 0].copy()
        
        # Aggregate by product
        if not shortage_df.empty:
            shortage_summary = shortage_df.groupby(['pt_code', 'product_name']).agg({
                'gap_quantity': lambda x: x.abs().sum(),
                'period': 'count'
            }).reset_index()
            shortage_summary.columns = ['pt_code', 'product_name', 'shortage_quantity', 'affected_periods']
            
            return {
                'shortage_products': shortage_summary,
                'period_type': st.session_state.get('period_gap_period_type', 'Weekly'),
                'analysis_time': st.session_state.get('last_analysis_time')
            }
    
    return {
        'shortage_products': None,
        'period_type': 'Weekly',
        'analysis_time': None
    }


def clear_all_gap_data():
    """Clear all GAP analysis related data (for logout or reset)"""
    gap_keys = [k for k in st.session_state.keys() if any(
        pattern in k.lower() for pattern in ['gap', 'pgap', 'period_gap']
    )]
    
    for key in gap_keys:
        del st.session_state[key]
    
    # Also clear cross-page data
    cross_page_keys = [
        'gap_analysis_result',
        'demand_filtered',
        'supply_filtered',
        'last_gap_analysis',
        'last_analysis_time'
    ]
    
    for key in cross_page_keys:
        if key in st.session_state:
            del st.session_state[key]