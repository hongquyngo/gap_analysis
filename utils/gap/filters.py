# utils/gap/filters.py

"""
Filter components module for GAP Analysis System
Provides reusable filter UI components for sidebar
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GAPFilters:
    """Manages filter UI components for GAP analysis"""
    
    def __init__(self, data_loader):
        """
        Initialize filters with data loader
        
        Args:
            data_loader: Instance of GAPDataLoader for fetching reference data
        """
        self.data_loader = data_loader
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state for filter persistence"""
        if 'gap_filters' not in st.session_state:
            st.session_state.gap_filters = {
                'entity': None,
                'date_range': (date.today(), date.today() + timedelta(days=30)),
                'products': [],
                'brands': [],
                'customers': [],
                'quick_filter': 'all',
                'group_by': 'product'
            }
    
    def render_sidebar_filters(self) -> Dict[str, Any]:
        """
        Render all filter components in sidebar
        
        Returns:
            Dictionary containing all filter selections
        """
        filters = {}
        
        st.sidebar.header("🔍 Filters")
        
        # Quick filter presets
        filters['quick_filter'] = self._render_quick_filters()
        
        # Entity selection
        filters['entity'] = self._render_entity_filter()
        
        # Date range
        filters['date_range'] = self._render_date_filter()
        
        # Product filters
        with st.sidebar.expander("📦 Product Filters", expanded=False):
            filters['products'] = self._render_product_filter(filters['entity'])
            filters['brands'] = self._render_brand_filter(filters['entity'])
        
        # Customer filter
        filters['customers'] = self._render_customer_filter(filters['entity'])
        
        # Grouping option
        filters['group_by'] = self._render_grouping_option()
        
        # Save to session state
        st.session_state.gap_filters = filters
        
        # Filter actions
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                self._reset_filters()
                st.rerun()
        
        with col2:
            if st.button("📊 Apply", type="primary", use_container_width=True):
                st.rerun()
        
        return filters
    
    def _render_quick_filters(self) -> str:
        """Render quick filter preset buttons"""
        st.sidebar.subheader("Quick Filters")
        
        quick_options = {
            'all': '📋 All Items',
            'shortage': '⚠️ Shortage Only',
            'critical': '🚨 Critical Only',
            'surplus': '📦 Surplus Only',
            'overdue': '⏰ Overdue Demand'
        }
        
        selected = st.sidebar.radio(
            "Select preset",
            options=list(quick_options.keys()),
            format_func=lambda x: quick_options[x],
            index=list(quick_options.keys()).index(st.session_state.gap_filters.get('quick_filter', 'all')),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        return selected
    
    def _render_entity_filter(self) -> Optional[str]:
        """Render entity selection filter"""
        entities = self.data_loader.get_entities()
        
        if not entities:
            st.sidebar.warning("No entities available")
            return None
        
        # Add "All Entities" option
        entity_options = ["All Entities"] + entities
        
        selected_entity = st.sidebar.selectbox(
            "🏢 Entity",
            options=entity_options,
            index=0 if st.session_state.gap_filters.get('entity') is None 
                   else entity_options.index(st.session_state.gap_filters.get('entity', 'All Entities')),
            help="Select entity to analyze"
        )
        
        return None if selected_entity == "All Entities" else selected_entity
    
    def _render_date_filter(self) -> Tuple[date, date]:
        """Render date range filter"""
        st.sidebar.subheader("📅 Date Range")
        
        col1, col2 = st.sidebar.columns(2)
        
        default_from, default_to = st.session_state.gap_filters.get(
            'date_range', 
            (date.today(), date.today() + timedelta(days=30))
        )
        
        with col1:
            date_from = st.date_input(
                "From",
                value=default_from,
                max_value=date.today() + timedelta(days=365),
                help="Start date for analysis"
            )
        
        with col2:
            date_to = st.date_input(
                "To",
                value=default_to,
                min_value=date_from,
                max_value=date.today() + timedelta(days=365),
                help="End date for analysis"
            )
        
        # Date range presets
        preset_ranges = {
            "Today": 0,
            "Next 7 Days": 7,
            "Next 14 Days": 14,
            "Next 30 Days": 30,
            "Next 60 Days": 60,
            "Next 90 Days": 90
        }
        
        preset_cols = st.sidebar.columns(3)
        for idx, (label, days) in enumerate(preset_ranges.items()):
            col_idx = idx % 3
            with preset_cols[col_idx]:
                if st.button(label, key=f"date_preset_{days}", use_container_width=True):
                    st.session_state.gap_filters['date_range'] = (
                        date.today(),
                        date.today() + timedelta(days=days)
                    )
                    st.rerun()
        
        return (date_from, date_to)
    
    def _render_product_filter(self, entity: Optional[str]) -> List[int]:
        """Render product selection filter"""
        products_df = self.data_loader.get_products(entity)
        
        if products_df.empty:
            st.warning("No products available")
            return []
        
        # Create display strings for products
        products_df['display'] = products_df.apply(
            lambda x: f"{x['pt_code']} - {x['product_name'][:30]}", 
            axis=1
        )
        
        # Search box for products
        search_term = st.text_input(
            "🔍 Search Products",
            placeholder="Enter PT code or product name...",
            help="Search for specific products"
        )
        
        # Filter products based on search
        if search_term:
            mask = (
                products_df['pt_code'].str.contains(search_term, case=False, na=False) |
                products_df['product_name'].str.contains(search_term, case=False, na=False)
            )
            filtered_products = products_df[mask]
        else:
            filtered_products = products_df.head(100)  # Show first 100 if no search
        
        # Multi-select for products
        selected_products = st.multiselect(
            "Select Products",
            options=filtered_products['product_id'].tolist(),
            format_func=lambda x: filtered_products[
                filtered_products['product_id'] == x
            ]['display'].iloc[0],
            default=st.session_state.gap_filters.get('products', []),
            help="Leave empty to include all products"
        )
        
        return selected_products
    
    def _render_brand_filter(self, entity: Optional[str]) -> List[str]:
        """Render brand selection filter"""
        brands = self.data_loader.get_brands(entity)
        
        if not brands:
            return []
        
        selected_brands = st.multiselect(
            "🏷️ Brands",
            options=brands,
            default=st.session_state.gap_filters.get('brands', []),
            help="Select brands to filter (leave empty for all)"
        )
        
        return selected_brands
    
    def _render_customer_filter(self, entity: Optional[str]) -> List[str]:
        """Render customer selection filter"""
        customers = self.data_loader.get_customers(entity)
        
        if not customers:
            return []
        
        # Only show if not too many customers
        if len(customers) > 100:
            # Use search box for large customer lists
            search_customer = st.sidebar.text_input(
                "🏢 Search Customer",
                placeholder="Type customer name...",
                help="Search for specific customers"
            )
            
            if search_customer:
                filtered_customers = [
                    c for c in customers 
                    if search_customer.lower() in c.lower()
                ]
                
                selected_customers = st.sidebar.multiselect(
                    "Select Customers",
                    options=filtered_customers[:20],  # Limit display
                    default=st.session_state.gap_filters.get('customers', []),
                    help="Select customers from search results"
                )
            else:
                selected_customers = []
        else:
            selected_customers = st.sidebar.multiselect(
                "🏢 Customers",
                options=customers,
                default=st.session_state.gap_filters.get('customers', []),
                help="Select customers to filter (leave empty for all)"
            )
        
        return selected_customers
    
    def _render_grouping_option(self) -> str:
        """Render grouping/aggregation option"""
        st.sidebar.subheader("📊 Grouping")
        
        group_options = {
            'product': '📦 By Product',
            'brand': '🏷️ By Brand',
            'category': '📂 By Category'
        }
        
        selected_group = st.sidebar.radio(
            "Group results by",
            options=list(group_options.keys()),
            format_func=lambda x: group_options[x],
            index=list(group_options.keys()).index(
                st.session_state.gap_filters.get('group_by', 'product')
            ),
            help="Choose aggregation level for analysis"
        )
        
        return selected_group
    
    def _reset_filters(self):
        """Reset all filters to default values"""
        st.session_state.gap_filters = {
            'entity': None,
            'date_range': (date.today(), date.today() + timedelta(days=30)),
            'products': [],
            'brands': [],
            'customers': [],
            'quick_filter': 'all',
            'group_by': 'product'
        }
    
    def apply_quick_filter(self, gap_df: pd.DataFrame, quick_filter: str) -> pd.DataFrame:
        """
        Apply quick filter preset to GAP dataframe
        
        Args:
            gap_df: DataFrame with GAP calculations
            quick_filter: Quick filter type
            
        Returns:
            Filtered DataFrame
        """
        if quick_filter == 'all':
            return gap_df
        elif quick_filter == 'shortage':
            return gap_df[gap_df['net_gap'] < 0]
        elif quick_filter == 'critical':
            return gap_df[gap_df['gap_status'].isin(['severe_shortage', 'high_shortage'])]
        elif quick_filter == 'surplus':
            return gap_df[gap_df['net_gap'] > 0]
        elif quick_filter == 'overdue':
            if 'avg_days_to_required' in gap_df.columns:
                return gap_df[gap_df['avg_days_to_required'] < 0]
            else:
                return gap_df[gap_df['net_gap'] < 0]  # Fallback to shortage
        else:
            return gap_df
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """
        Generate a summary string of active filters
        
        Args:
            filters: Dictionary of filter values
            
        Returns:
            Summary string for display
        """
        summary_parts = []
        
        if filters.get('entity'):
            summary_parts.append(f"Entity: {filters['entity']}")
        
        if filters.get('date_range'):
            date_from, date_to = filters['date_range']
            summary_parts.append(f"Dates: {date_from} to {date_to}")
        
        if filters.get('brands'):
            summary_parts.append(f"Brands: {len(filters['brands'])} selected")
        
        if filters.get('products'):
            summary_parts.append(f"Products: {len(filters['products'])} selected")
        
        if filters.get('customers'):
            summary_parts.append(f"Customers: {len(filters['customers'])} selected")
        
        if filters.get('quick_filter') != 'all':
            summary_parts.append(f"Filter: {filters['quick_filter'].title()}")
        
        return " | ".join(summary_parts) if summary_parts else "No filters applied"