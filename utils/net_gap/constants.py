# utils/net_gap/constants.py

"""
Constants for GAP Analysis System - Updated Version
Optimized chart heights and configurations
"""

# GAP Status Categories - Simplified from 11 to 5
GAP_CATEGORIES = {
    'SHORTAGE': {
        'statuses': ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE', 
                    'BELOW_SAFETY', 'CRITICAL_BREACH'],
        'color': '#DC2626',
        'label': 'Shortage',
        'icon': '🔴'
    },
    'OPTIMAL': {
        'statuses': ['BALANCED', 'LIGHT_SURPLUS'],
        'color': '#10B981',
        'label': 'Optimal',
        'icon': '✅'
    },
    'SURPLUS': {
        'statuses': ['MODERATE_SURPLUS', 'HIGH_SURPLUS', 'SEVERE_SURPLUS'],
        'color': '#3B82F6',
        'label': 'Surplus',
        'icon': '📦'
    },
    'INACTIVE': {
        'statuses': ['NO_DEMAND', 'NO_DEMAND_INCOMING'],
        'color': '#9CA3AF',
        'label': 'Inactive',
        'icon': '⭕'
    },
    'EXPIRED': {
        'statuses': ['HAS_EXPIRED', 'EXPIRY_RISK'],
        'color': '#F59E0B',
        'label': 'Expired/Risk',
        'icon': '⚠️'
    }
}

# Coverage Thresholds
THRESHOLDS = {
    'coverage': {
        'severe_shortage': 0.5,
        'high_shortage': 0.7,
        'moderate_shortage': 0.9,
        'balanced_low': 0.9,
        'balanced_high': 1.1,
        'light_surplus': 1.5,
        'moderate_surplus': 2.0,
        'high_surplus': 3.0
    },
    'safety': {
        'critical_breach': 0.5,
        'below_safety': 1.0,
        'at_reorder': 1.0
    },
    'priority': {
        'critical': 1,
        'high': 2,
        'medium': 3,
        'low': 4,
        'ok': 99
    }
}

# Supply Sources Configuration
SUPPLY_SOURCES = {
    'INVENTORY': {
        'name': 'Inventory',
        'icon': '📦',
        'priority': 1,
        'lead_days': '0'
    },
    'CAN_PENDING': {
        'name': 'CAN Pending',
        'icon': '📋',
        'priority': 2,
        'lead_days': '1-3'
    },
    'WAREHOUSE_TRANSFER': {
        'name': 'Transfer',
        'icon': '🚛',
        'priority': 3,
        'lead_days': '2-5'
    },
    'PURCHASE_ORDER': {
        'name': 'Purchase Order',
        'icon': '📝',
        'priority': 4,
        'lead_days': '7-30'
    }
}

# Demand Sources Configuration
DEMAND_SOURCES = {
    'OC_PENDING': {
        'name': 'Confirmed Orders',
        'icon': '✔',
        'priority': 1
    },
    'FORECAST': {
        'name': 'Forecast',
        'icon': '📊',
        'priority': 2
    }
}

# Field Tooltips - Enhanced with formula explanations
FIELD_TOOLTIPS = {
    'pt_code': 'Product code identifier',
    'Total Supply': 'Total Supply = Inventory + Pending + Transfer + PO',
    'Total Demand': 'Total Demand = Orders + Forecast',
    'Net GAP': 'Available Supply - Demand (considers safety stock if enabled)',
    'True GAP': 'Total Supply - Demand (ignores safety stock)',
    'Coverage %': '(Supply ÷ Demand) × 100%',
    'Safety Stock': 'Minimum required inventory level',
    'Available Supply': 'Total Supply - Safety Stock',
    'At Risk Value': 'Revenue at risk from shortage = Shortage × Selling Price',
    'GAP Value': 'Inventory value of gap = GAP × Unit Cost',
    'Reorder Point': 'Stock level that triggers reorder',
    'Below Reorder': 'Indicates if stock is below reorder point',
    'Safety Coverage': 'Current Inventory ÷ Safety Stock requirement',
    'Unit Cost': 'Average landed cost per unit',
    'Sell Price': 'Average selling price per unit',
    'Customers': 'Number of unique customers affected'
}

# Export Configuration
EXPORT_CONFIG = {
    'max_rows': 10000,
    'include_formulas': True,
    'include_cost_breakdown': True,
    'sheets': ['Summary', 'GAP Details', 'Cost Analysis', 'Calculation Guide']
}

# UI Configuration - Optimized heights
UI_CONFIG = {
    'items_per_page_options': [10, 25, 50, 100],
    'default_items_per_page': 25,
    'max_chart_items': 20,
    'chart_height': 400,  # Original height
    'chart_height_compact': 300,  # Reduced height for donut charts
    'chart_height_min': 250,  # Minimum for bar charts
    'chart_height_max': 400,  # Maximum for bar charts
    'chart_margin_compact': 30,  # Reduced margins
    'table_row_height': 35  # Height per row in tables
}

# Status Icons
STATUS_ICONS = {
    'SHORTAGE': '🔴',
    'OPTIMAL': '✅',
    'SURPLUS': '📦',
    'INACTIVE': '⭕',
    'WARNING': '⚠️',
    'CRITICAL': '🚨'
}

# Formula Constants
FORMULA_INFO = {
    'net_gap': {
        'formula': 'Available Supply - Total Demand',
        'description': 'When safety enabled: Available = Supply - Safety Stock'
    },
    'true_gap': {
        'formula': 'Total Supply - Total Demand',
        'description': 'Always ignores safety stock'
    },
    'coverage_ratio': {
        'formula': '(Supply ÷ Demand) × 100%',
        'description': 'Supply as percentage of demand'
    },
    'at_risk_value': {
        'formula': '|Shortage Qty| × Selling Price',
        'description': 'Potential revenue loss'
    },
    'gap_value': {
        'formula': 'Net GAP × Unit Cost',
        'description': 'Inventory value of the gap'
    },
    'safety_impact': {
        'formula': 'Net GAP - True GAP',
        'description': 'How safety stock affects the gap'
    }
}