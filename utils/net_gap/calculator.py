# utils/net_gap/calculator.py

"""
Calculator module for GAP Analysis System - Clean Version
Handles all GAP calculations and aggregations
Aligned with SQL view logic from net_gap_view_with_fc
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Constants for GAP thresholds
COVERAGE_THRESHOLDS = {
    'SEVERE_SURPLUS': 3.0,      # > 300%
    'HIGH_SURPLUS': 2.0,         # 200-300%
    'MODERATE_SURPLUS': 1.5,     # 150-200%
    'LIGHT_SURPLUS': 1.1,        # 110-150%
    'BALANCED_HIGH': 1.1,        # 90-110%
    'BALANCED_LOW': 0.9,         # 90-110%
    'MODERATE_SHORTAGE': 0.7,    # 70-90%
    'HIGH_SHORTAGE': 0.5,        # 50-70%
    'SEVERE_SHORTAGE': 0.0       # < 50%
}

PRIORITY_LEVELS = {
    'CRITICAL': 1,
    'HIGH': 2,
    'MEDIUM': 3,
    'LOW': 4,
    'OK': 99
}


class GAPCalculator:
    """Handles GAP calculations for supply-demand analysis"""
    
    def __init__(self):
        """Initialize calculator with default settings"""
        self.supply_sources = ['INVENTORY', 'CAN_PENDING', 'WAREHOUSE_TRANSFER', 'PURCHASE_ORDER']
        self.demand_sources = ['OC_PENDING', 'FORECAST']
    
    def calculate_net_gap(
        self,
        supply_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        group_by: str = 'product',
        selected_supply_sources: Optional[List[str]] = None,
        selected_demand_sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate simple net GAP (Supply - Demand) without time dimension
        
        Args:
            supply_df: Supply data from unified_supply_view
            demand_df: Demand data from unified_demand_view
            group_by: Aggregation level ('product', 'brand', 'category')
            selected_supply_sources: Supply sources to include
            selected_demand_sources: Demand sources to include
            
        Returns:
            DataFrame with GAP calculations
        """
        try:
            # Filter by selected sources
            if selected_supply_sources:
                supply_df = supply_df[supply_df['supply_source'].isin(selected_supply_sources)].copy()
            
            if selected_demand_sources:
                demand_df = demand_df[demand_df['demand_source'].isin(selected_demand_sources)].copy()
            
            # Get group columns
            group_cols = self._get_group_columns(group_by, supply_df, demand_df)
            
            # Aggregate supply and demand
            supply_agg = self._aggregate_supply(supply_df, group_cols)
            demand_agg = self._aggregate_demand(demand_df, group_cols)
            
            # Merge supply and demand
            gap_df = pd.merge(
                supply_agg,
                demand_agg,
                on=group_cols,
                how='outer',
                suffixes=('_supply', '_demand')
            )
            
            # Fill NaN values
            numeric_cols = gap_df.select_dtypes(include=[np.number]).columns
            gap_df[numeric_cols] = gap_df[numeric_cols].fillna(0)
            
            # Calculate GAP metrics
            gap_df = self._calculate_gap_metrics(gap_df)
            
            # Sort by priority
            gap_df = gap_df.sort_values(
                by=['priority', 'net_gap'],
                ascending=[True, True]
            )
            
            logger.info(f"Calculated GAP for {len(gap_df)} {group_by} groups")
            return gap_df
            
        except Exception as e:
            logger.error(f"Error calculating net GAP: {e}", exc_info=True)
            raise
    
    def _get_group_columns(self, group_by: str, supply_df: pd.DataFrame, 
                          demand_df: pd.DataFrame) -> List[str]:
        """Get grouping columns based on aggregation level"""
        if group_by == 'product':
            return ['product_id', 'product_name', 'pt_code', 'brand', 'standard_uom']
        elif group_by == 'brand':
            return ['brand']
        else:  # category
            # Add category column based on PT code prefix
            if 'pt_code' in supply_df.columns:
                supply_df['category'] = supply_df['pt_code'].str[:2].fillna('Unknown')
            if 'pt_code' in demand_df.columns:
                demand_df['category'] = demand_df['pt_code'].str[:2].fillna('Unknown')
            return ['category']
    
    def _aggregate_supply(self, supply_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Aggregate supply data by specified columns"""
        if supply_df.empty:
            return pd.DataFrame(columns=group_cols + ['total_supply'])
        
        # Filter out expired items
        today = pd.Timestamp.now().date()
        if 'expiry_date' in supply_df.columns:
            supply_df = supply_df[
                (supply_df['expiry_date'].isna()) | 
                (pd.to_datetime(supply_df['expiry_date']).dt.date > today)
            ]
        
        # Create aggregation dictionary
        agg_dict = {
            'available_quantity': 'sum',
            'total_value_usd': 'sum'
        }
        
        # Add source-specific columns for breakdown
        for source in self.supply_sources:
            col_name = f'supply_{source.lower()}'
            supply_df[col_name] = np.where(
                supply_df['supply_source'] == source,
                supply_df['available_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # Add weighted average cost calculation
        if 'unit_cost_usd' in supply_df.columns:
            supply_df['cost_x_qty'] = supply_df['unit_cost_usd'] * supply_df['available_quantity']
            agg_dict['cost_x_qty'] = 'sum'
        
        # Perform aggregation
        supply_agg = supply_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        supply_agg.rename(columns={
            'available_quantity': 'total_supply',
            'total_value_usd': 'supply_value_usd'
        }, inplace=True)
        
        # Calculate weighted average unit cost
        if 'cost_x_qty' in supply_agg.columns:
            supply_agg['avg_unit_cost_usd'] = np.where(
                supply_agg['total_supply'] > 0,
                supply_agg['cost_x_qty'] / supply_agg['total_supply'],
                0
            )
            supply_agg.drop(columns=['cost_x_qty'], inplace=True)
        
        return supply_agg
    
    def _aggregate_demand(self, demand_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Aggregate demand data by specified columns"""
        if demand_df.empty:
            return pd.DataFrame(columns=group_cols + ['total_demand'])
        
        # Create aggregation dictionary
        agg_dict = {
            'required_quantity': 'sum',
            'total_value_usd': 'sum',
            'customer': 'nunique'
        }
        
        # Add optional columns if they exist
        optional_aggs = {
            'allocated_quantity': 'sum',
            'unallocated_quantity': 'sum',
            'over_committed_qty_standard': 'sum',
            'days_to_required': 'mean'
        }
        
        for col, agg_func in optional_aggs.items():
            if col in demand_df.columns:
                agg_dict[col] = agg_func
        
        # Add source-specific columns
        for source in self.demand_sources:
            col_name = f'demand_{source.lower()}'
            demand_df[col_name] = np.where(
                demand_df['demand_source'] == source,
                demand_df['required_quantity'],
                0
            )
            agg_dict[col_name] = 'sum'
        
        # Add weighted average price calculation
        if 'selling_unit_price' in demand_df.columns:
            demand_df['price_x_qty'] = demand_df['selling_unit_price'] * demand_df['required_quantity']
            agg_dict['price_x_qty'] = 'sum'
        
        # Count urgency levels
        if 'urgency_level' in demand_df.columns:
            for urgency in ['OVERDUE', 'URGENT']:
                col_name = f'count_{urgency.lower()}'
                demand_df[col_name] = (demand_df['urgency_level'] == urgency).astype(int)
                agg_dict[col_name] = 'sum'
        
        # Perform aggregation
        demand_agg = demand_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        # Rename columns
        demand_agg.rename(columns={
            'required_quantity': 'total_demand',
            'total_value_usd': 'demand_value_usd',
            'customer': 'customer_count',
            'days_to_required': 'avg_days_to_required'
        }, inplace=True)
        
        # Calculate weighted average selling price
        if 'price_x_qty' in demand_agg.columns:
            demand_agg['avg_selling_price_usd'] = np.where(
                demand_agg['total_demand'] > 0,
                demand_agg['price_x_qty'] / demand_agg['total_demand'],
                0
            )
            demand_agg.drop(columns=['price_x_qty'], inplace=True)
        
        # Calculate allocation coverage
        if 'allocated_quantity' in demand_agg.columns:
            demand_agg['allocation_coverage'] = np.where(
                demand_agg['total_demand'] > 0,
                (demand_agg['allocated_quantity'] / demand_agg['total_demand']) * 100,
                0
            )
        
        return demand_agg
    
    def _calculate_gap_metrics(self, gap_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all GAP-related metrics"""
        # Basic GAP calculations
        gap_df['net_gap'] = gap_df['total_supply'] - gap_df['total_demand']
        
        # Coverage ratio
        gap_df['coverage_ratio'] = np.where(
            gap_df['total_demand'] > 0,
            gap_df['total_supply'] / gap_df['total_demand'],
            np.where(gap_df['total_supply'] > 0, 999, 0)
        )
        
        # GAP percentage
        gap_df['gap_percentage'] = np.where(
            gap_df['total_demand'] > 0,
            (gap_df['net_gap'] / gap_df['total_demand']) * 100,
            np.where(gap_df['total_supply'] > 0, 100, 0)
        )
        
        # GAP type
        gap_df['gap_type'] = np.where(
            gap_df['net_gap'] > 0, 'SURPLUS',
            np.where(gap_df['net_gap'] < 0, 'SHORTAGE', 'BALANCED')
        )
        
        # GAP status
        gap_df['gap_status'] = gap_df.apply(
            lambda row: self._classify_gap_status(
                row['coverage_ratio'],
                row['total_demand'],
                row.get('supply_inventory', 0),
                row.get('supply_purchase_order', 0)
            ), axis=1
        )
        
        # Priority
        gap_df['priority'] = gap_df['coverage_ratio'].apply(self._calculate_priority)
        
        # Suggested action
        gap_df['suggested_action'] = gap_df.apply(
            lambda row: self._get_suggested_action(
                row['coverage_ratio'],
                row['net_gap'],
                row['total_demand']
            ), axis=1
        )
        
        # Value calculations
        if 'avg_unit_cost_usd' in gap_df.columns:
            gap_df['gap_value_usd'] = gap_df['net_gap'] * gap_df['avg_unit_cost_usd']
            
            # At-risk value for shortages
            gap_df['at_risk_value_usd'] = np.where(
                gap_df['net_gap'] < 0,
                abs(gap_df['net_gap']) * gap_df.get('avg_selling_price_usd', gap_df['avg_unit_cost_usd']),
                0
            )
        
        return gap_df
    
    def _classify_gap_status(self, coverage: float, demand: float, 
                            inventory: float = 0, po_qty: float = 0) -> str:
        """Classify GAP status based on coverage ratio"""
        # Special cases
        if demand == 0:
            if inventory > 0:
                return 'NO_DEMAND'
            elif po_qty > 0:
                return 'NO_DEMAND_INCOMING'
            return 'NO_DEMAND'
        
        # Coverage-based classification
        if coverage > COVERAGE_THRESHOLDS['SEVERE_SURPLUS']:
            return 'SEVERE_SURPLUS'
        elif coverage > COVERAGE_THRESHOLDS['HIGH_SURPLUS']:
            return 'HIGH_SURPLUS'
        elif coverage > COVERAGE_THRESHOLDS['MODERATE_SURPLUS']:
            return 'MODERATE_SURPLUS'
        elif coverage > COVERAGE_THRESHOLDS['LIGHT_SURPLUS']:
            return 'LIGHT_SURPLUS'
        elif coverage >= COVERAGE_THRESHOLDS['BALANCED_LOW']:
            return 'BALANCED'
        elif coverage >= COVERAGE_THRESHOLDS['MODERATE_SHORTAGE']:
            return 'MODERATE_SHORTAGE'
        elif coverage >= COVERAGE_THRESHOLDS['HIGH_SHORTAGE']:
            return 'HIGH_SHORTAGE'
        else:
            return 'SEVERE_SHORTAGE'
    
    def _calculate_priority(self, coverage: float) -> int:
        """Calculate action priority based on coverage"""
        if coverage < COVERAGE_THRESHOLDS['HIGH_SHORTAGE']:
            return PRIORITY_LEVELS['CRITICAL']
        elif coverage > COVERAGE_THRESHOLDS['SEVERE_SURPLUS']:
            return PRIORITY_LEVELS['HIGH']
        elif coverage < COVERAGE_THRESHOLDS['MODERATE_SHORTAGE']:
            return PRIORITY_LEVELS['HIGH']
        elif coverage > COVERAGE_THRESHOLDS['HIGH_SURPLUS']:
            return PRIORITY_LEVELS['MEDIUM']
        elif coverage < COVERAGE_THRESHOLDS['BALANCED_LOW']:
            return PRIORITY_LEVELS['MEDIUM']
        elif coverage > COVERAGE_THRESHOLDS['MODERATE_SURPLUS']:
            return PRIORITY_LEVELS['LOW']
        else:
            return PRIORITY_LEVELS['OK']
    
    def _get_suggested_action(self, coverage: float, gap: float, demand: float) -> str:
        """Generate suggested action based on GAP analysis"""
        if demand == 0:
            if gap > 0:
                return f"NO DEMAND: {abs(gap):.0f} units need liquidation"
            return "No demand - monitor for changes"
        
        if coverage < COVERAGE_THRESHOLDS['HIGH_SHORTAGE']:
            return f"URGENT: Need {abs(gap):.0f} units immediately. Create emergency PO!"
        elif coverage < COVERAGE_THRESHOLDS['MODERATE_SHORTAGE']:
            return f"HIGH: Need {abs(gap):.0f} units. Create PO within 2 days"
        elif coverage < COVERAGE_THRESHOLDS['BALANCED_LOW']:
            return f"Need {abs(gap):.0f} units. Plan replenishment"
        elif coverage <= COVERAGE_THRESHOLDS['BALANCED_HIGH']:
            return "Supply-demand balanced. Monitor levels"
        elif coverage <= COVERAGE_THRESHOLDS['MODERATE_SURPLUS']:
            return f"Minor surplus ({gap:.0f} units). Monitor closely"
        elif coverage <= COVERAGE_THRESHOLDS['HIGH_SURPLUS']:
            return f"Surplus {gap:.0f} units. Review ordering patterns"
        elif coverage <= COVERAGE_THRESHOLDS['SEVERE_SURPLUS']:
            return f"High surplus {gap:.0f} units. Reduce orders & plan promotions"
        else:
            return f"SEVERE SURPLUS {gap:.0f} units. Stop ordering immediately!"
    
    def get_summary_metrics(self, gap_df: pd.DataFrame) -> Dict[str, any]:
        """Calculate summary metrics from GAP analysis"""
        # Define status groups
        shortage_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE', 'MODERATE_SHORTAGE']
        critical_statuses = ['SEVERE_SHORTAGE', 'HIGH_SHORTAGE']
        surplus_statuses = ['SEVERE_SURPLUS', 'HIGH_SURPLUS', 'MODERATE_SURPLUS', 'LIGHT_SURPLUS']
        
        # Calculate metrics
        metrics = {
            'total_products': len(gap_df),
            'total_supply': gap_df['total_supply'].sum(),
            'total_demand': gap_df['total_demand'].sum(),
            'net_gap': gap_df['net_gap'].sum(),
            
            # Shortage metrics
            'shortage_items': len(gap_df[gap_df['gap_status'].isin(shortage_statuses)]),
            'total_shortage': abs(gap_df[gap_df['net_gap'] < 0]['net_gap'].sum()),
            
            # Surplus metrics
            'surplus_items': len(gap_df[gap_df['gap_status'].isin(surplus_statuses)]),
            'total_surplus': gap_df[gap_df['net_gap'] > 0]['net_gap'].sum(),
            
            # Critical items
            'critical_items': len(gap_df[gap_df['gap_status'].isin(critical_statuses)]),
            
            # Coverage
            'overall_coverage': self._calculate_overall_coverage(gap_df),
            
            # Value metrics
            'at_risk_value_usd': gap_df.get('at_risk_value_usd', pd.Series([0])).sum(),
            'gap_value_usd': gap_df.get('gap_value_usd', pd.Series([0])).sum(),
            
            # Customer impact
            'affected_customers': int(gap_df[gap_df['net_gap'] < 0].get('customer_count', pd.Series([0])).sum()),
            
            # Urgency counts
            'overdue_items': int(gap_df.get('count_overdue', pd.Series([0])).sum()),
            'urgent_items': int(gap_df.get('count_urgent', pd.Series([0])).sum())
        }
        
        return metrics
    
    def _calculate_overall_coverage(self, gap_df: pd.DataFrame) -> float:
        """Calculate overall coverage percentage"""
        total_supply = gap_df['total_supply'].sum()
        total_demand = gap_df['total_demand'].sum()
        
        if total_demand > 0:
            return (total_supply / total_demand) * 100
        return 100.0 if total_supply == 0 else 999.0