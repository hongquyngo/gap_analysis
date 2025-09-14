# utils/gap/formatters.py

"""
Formatting module for GAP Analysis System
Handles number formatting, date formatting, and tooltip generation
"""

import pandas as pd
import streamlit as st
from datetime import datetime, date
from typing import Optional, Union, Dict, List, Any
import numpy as np


class GAPFormatter:
    """Handles all formatting and tooltip generation for GAP analysis"""
    
    # Status emoji and colors
    STATUS_CONFIG = {
        'severe_shortage': {'emoji': '🔴', 'label': 'Severe Shortage', 'color': '#FF4444'},
        'high_shortage': {'emoji': '🟠', 'label': 'High Shortage', 'color': '#FF8800'},
        'low_shortage': {'emoji': '🟡', 'label': 'Low Shortage', 'color': '#FFAA00'},
        'balanced': {'emoji': '✅', 'label': 'Balanced', 'color': '#00AA00'},
        'surplus': {'emoji': '🔵', 'label': 'Surplus', 'color': '#0088FF'},
        'high_surplus': {'emoji': '🟣', 'label': 'High Surplus', 'color': '#8800FF'}
    }
    
    @staticmethod
    def format_number(value: Union[float, int], decimals: int = 0, 
                      show_sign: bool = False) -> str:
        """
        Format number with thousand separators
        
        Args:
            value: Number to format
            decimals: Number of decimal places
            show_sign: Whether to show + sign for positive numbers
            
        Returns:
            Formatted string
        """
        if pd.isna(value):
            return "N/A"
        
        if decimals == 0:
            formatted = f"{int(value):,}"
        else:
            formatted = f"{value:,.{decimals}f}"
        
        if show_sign and value > 0:
            formatted = f"+{formatted}"
        
        return formatted
    
    @staticmethod
    def format_currency(value: Union[float, int], currency: str = "USD") -> str:
        """Format value as currency"""
        if pd.isna(value):
            return "N/A"
        
        if currency == "USD":
            return f"${value:,.2f}"
        elif currency == "VND":
            return f"₫{value:,.0f}"
        else:
            return f"{value:,.2f} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format value as percentage"""
        if pd.isna(value):
            return "N/A"
        
        return f"{value:.{decimals}%}"
    
    @staticmethod
    def format_date(date_value: Union[date, datetime, pd.Timestamp], 
                   format_str: str = "%Y-%m-%d") -> str:
        """Format date for display"""
        if pd.isna(date_value):
            return "N/A"
        
        if isinstance(date_value, pd.Timestamp):
            date_value = date_value.to_pydatetime()
        
        return date_value.strftime(format_str)
    
    @staticmethod
    def get_date_tooltip(date_type: str, date_value: Union[date, datetime], 
                        days_value: int = None) -> str:
        """
        Generate tooltip for date fields
        
        Args:
            date_type: Type of date ('required', 'availability', 'expiry')
            date_value: The date value
            days_value: Days to/from date (optional)
            
        Returns:
            Tooltip text
        """
        tooltips = {
            'required': {
                'base': "📅 Required Date (ETD/ETA)",
                'description': "Date when demand needs to be fulfilled",
                'overdue': "⚠️ OVERDUE by {} days - Immediate action required",
                'upcoming': "📍 Due in {} days",
                'future': "📆 Scheduled for {} days from now"
            },
            'availability': {
                'base': "📦 Availability Date",
                'description': "Expected date when supply becomes available",
                'immediate': "✅ Available immediately (in stock)",
                'pending': "⏳ Expected in {} days ({})",
                'source_info': {
                    'INVENTORY': "Already in warehouse",
                    'CAN_PENDING': "Awaiting stock-in after arrival",
                    'WAREHOUSE_TRANSFER': "In transit between warehouses",
                    'PURCHASE_ORDER': "On order from supplier"
                }
            },
            'expiry': {
                'base': "⏰ Expiry Date",
                'description': "Product expiration date",
                'expired': "❌ EXPIRED {} days ago",
                'expiring_soon': "⚠️ Expiring in {} days - Priority to use",
                'good': "✅ {} days remaining shelf life"
            }
        }
        
        if date_type not in tooltips:
            return ""
        
        tooltip_config = tooltips[date_type]
        tooltip_parts = [tooltip_config['base']]
        
        if pd.notna(date_value):
            formatted_date = GAPFormatter.format_date(date_value)
            tooltip_parts.append(f"Date: {formatted_date}")
            
            if days_value is not None:
                if date_type == 'required':
                    if days_value < 0:
                        tooltip_parts.append(tooltip_config['overdue'].format(abs(days_value)))
                    elif days_value <= 7:
                        tooltip_parts.append(tooltip_config['upcoming'].format(days_value))
                    else:
                        tooltip_parts.append(tooltip_config['future'].format(days_value))
                
                elif date_type == 'availability':
                    if days_value == 0:
                        tooltip_parts.append(tooltip_config['immediate'])
                    else:
                        tooltip_parts.append(tooltip_config['pending'].format(days_value, "processing"))
                
                elif date_type == 'expiry':
                    if days_value < 0:
                        tooltip_parts.append(tooltip_config['expired'].format(abs(days_value)))
                    elif days_value <= 30:
                        tooltip_parts.append(tooltip_config['expiring_soon'].format(days_value))
                    else:
                        tooltip_parts.append(tooltip_config['good'].format(days_value))
        else:
            tooltip_parts.append("No date specified")
        
        tooltip_parts.append(f"\n{tooltip_config['description']}")
        
        return "\n".join(tooltip_parts)
    
    @staticmethod
    def get_quantity_tooltip(quantity_type: str, value: float, 
                           breakdown: Dict[str, float] = None,
                           allocation_info: Dict[str, Any] = None) -> str:
        """
        Generate tooltip for quantity fields
        
        Args:
            quantity_type: Type of quantity ('supply', 'demand', 'gap')
            value: The quantity value
            breakdown: Breakdown by source/customer
            allocation_info: Allocation details
            
        Returns:
            Tooltip text
        """
        tooltip_parts = []
        
        if quantity_type == 'supply':
            tooltip_parts.append("📦 Total Supply Quantity")
            tooltip_parts.append(f"Total: {GAPFormatter.format_number(value)} units")
            
            if breakdown:
                tooltip_parts.append("\n📊 Breakdown by Source:")
                for source, qty in breakdown.items():
                    if qty > 0:
                        pct = (qty / value * 100) if value > 0 else 0
                        tooltip_parts.append(f"  • {source}: {GAPFormatter.format_number(qty)} ({pct:.1f}%)")
        
        elif quantity_type == 'demand':
            tooltip_parts.append("📋 Total Demand Quantity")
            tooltip_parts.append(f"Total: {GAPFormatter.format_number(value)} units")
            
            if allocation_info:
                if allocation_info.get('is_allocated'):
                    coverage = allocation_info.get('allocation_coverage_percent', 0)
                    allocated = allocation_info.get('allocated_quantity', 0)
                    unallocated = allocation_info.get('unallocated_quantity', 0)
                    
                    tooltip_parts.append("\n🔗 Allocation Status:")
                    tooltip_parts.append(f"  • Coverage: {coverage:.1f}%")
                    tooltip_parts.append(f"  • Allocated: {GAPFormatter.format_number(allocated)}")
                    tooltip_parts.append(f"  • Unallocated: {GAPFormatter.format_number(unallocated)}")
                    
                    if allocation_info.get('is_over_committed'):
                        over_qty = allocation_info.get('over_committed_qty_standard', 0)
                        tooltip_parts.append(f"  ⚠️ Over-committed: {GAPFormatter.format_number(over_qty)}")
            
            if breakdown:
                tooltip_parts.append(f"\n👥 Customer Count: {len(breakdown)}")
                if len(breakdown) <= 5:
                    tooltip_parts.append("Customers:")
                    for customer, qty in list(breakdown.items())[:5]:
                        tooltip_parts.append(f"  • {customer}: {GAPFormatter.format_number(qty)}")
        
        elif quantity_type == 'gap':
            tooltip_parts.append("📊 Supply-Demand GAP")
            tooltip_parts.append(f"GAP: {GAPFormatter.format_number(value, show_sign=True)} units")
            
            if value < 0:
                tooltip_parts.append("⚠️ Shortage - Additional supply needed")
            elif value > 0:
                tooltip_parts.append("✅ Surplus - Consider reallocation")
            else:
                tooltip_parts.append("✅ Perfectly balanced")
        
        return "\n".join(tooltip_parts)
    
    @staticmethod
    def get_gap_status(gap_value: float, demand_value: float) -> Dict[str, str]:
        """
        Determine GAP status based on gap value and percentage
        
        Args:
            gap_value: Net GAP value (supply - demand)
            demand_value: Total demand value
            
        Returns:
            Dictionary with status key, emoji, label, and color
        """
        if demand_value == 0:
            if gap_value > 0:
                return GAPFormatter.STATUS_CONFIG['high_surplus']
            else:
                return GAPFormatter.STATUS_CONFIG['balanced']
        
        gap_percent = gap_value / demand_value
        
        if gap_percent < -0.5:  # < -50%
            return GAPFormatter.STATUS_CONFIG['severe_shortage']
        elif gap_percent < -0.2:  # -50% to -20%
            return GAPFormatter.STATUS_CONFIG['high_shortage']
        elif gap_percent < -0.05:  # -20% to -5%
            return GAPFormatter.STATUS_CONFIG['low_shortage']
        elif gap_percent <= 0.1:  # -5% to +10%
            return GAPFormatter.STATUS_CONFIG['balanced']
        elif gap_percent <= 0.5:  # +10% to +50%
            return GAPFormatter.STATUS_CONFIG['surplus']
        else:  # > +50%
            return GAPFormatter.STATUS_CONFIG['high_surplus']
    
    @staticmethod
    def format_status_badge(status_dict: Dict[str, str]) -> str:
        """Format status as HTML badge for display"""
        emoji = status_dict['emoji']
        label = status_dict['label']
        color = status_dict['color']
        
        return f"""
        <span style="
            background-color: {color}20; 
            color: {color}; 
            padding: 2px 8px; 
            border-radius: 4px;
            font-weight: 500;
        ">
            {emoji} {label}
        </span>
        """
    
    @staticmethod
    def get_suggested_action(gap_value: float, demand_value: float, 
                            days_to_required: float = None) -> str:
        """
        Generate suggested action based on GAP analysis
        
        Args:
            gap_value: Net GAP value
            demand_value: Total demand value
            days_to_required: Average days to required date
            
        Returns:
            Suggested action text
        """
        if demand_value == 0:
            return "No demand - Monitor for changes"
        
        gap_percent = gap_value / demand_value
        
        if gap_percent < -0.5:
            action = "🚨 URGENT: Create emergency PO"
            if days_to_required and days_to_required < 7:
                action += " + expedite shipping"
        elif gap_percent < -0.2:
            action = "⚠️ Create PO within 2 days"
        elif gap_percent < -0.05:
            action = "📋 Plan PO for next week"
        elif gap_percent <= 0.1:
            action = "✅ Monitor stock levels"
        elif gap_percent <= 0.5:
            action = "📊 Consider demand forecast adjustment"
        else:
            action = "🔄 Evaluate for redistribution"
        
        return action
    
    @staticmethod
    def format_dataframe_with_tooltips(df: pd.DataFrame, 
                                      tooltip_columns: Dict[str, str]) -> pd.DataFrame:
        """
        Add tooltip columns to dataframe
        
        Args:
            df: Input dataframe
            tooltip_columns: Mapping of column names to tooltip types
            
        Returns:
            DataFrame with tooltip columns added
        """
        df_with_tooltips = df.copy()
        
        for col, tooltip_type in tooltip_columns.items():
            if col in df.columns:
                tooltip_col_name = f"{col}_tooltip"
                
                if 'date' in tooltip_type:
                    # Handle date tooltips
                    days_col = col.replace('_date', '_days')
                    if days_col in df.columns:
                        df_with_tooltips[tooltip_col_name] = df.apply(
                            lambda row: GAPFormatter.get_date_tooltip(
                                tooltip_type, row[col], row.get(days_col)
                            ), axis=1
                        )
                
                elif 'quantity' in tooltip_type:
                    # Handle quantity tooltips
                    df_with_tooltips[tooltip_col_name] = df[col].apply(
                        lambda x: GAPFormatter.get_quantity_tooltip(tooltip_type, x)
                    )
        
        return df_with_tooltips