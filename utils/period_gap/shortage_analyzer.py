# utils/period_gap/shortage_analyzer.py
"""
Shortage Analyzer Module
Categorizes shortage types into Net Shortage vs Timing Gap
"""

import pandas as pd
from typing import Dict, Set
import logging

logger = logging.getLogger(__name__)


def categorize_shortage_type(gap_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Categorize products based on shortage type:
    - Net Shortage: Total supply < Total demand (need new orders)
    - Timing Gap: Total supply >= Total demand but timing mismatch (need expedite/reschedule)
    
    Args:
        gap_df: GAP analysis dataframe with columns:
                - pt_code: Product code
                - gap_quantity: GAP amount for each period
                - total_demand_qty: Demand quantity per period
                - supply_in_period: Supply quantity per period
                - backlog_to_next (optional): Backlog carried to next period
    
    Returns:
        Dictionary with two keys:
        - 'net_shortage': Set of product codes with net shortage
        - 'timing_gap': Set of product codes with timing gaps only
    """
    
    if gap_df.empty:
        return {'net_shortage': set(), 'timing_gap': set()}
    
    net_shortage_products = set()
    timing_gap_products = set()
    
    # Group by product
    for pt_code in gap_df['pt_code'].unique():
        product_df = gap_df[gap_df['pt_code'] == pt_code].copy()
        
        # Calculate totals
        total_demand = product_df['total_demand_qty'].sum()
        total_supply = product_df['supply_in_period'].sum()
        
        # Check for any shortage periods
        has_shortage = (product_df['gap_quantity'] < 0).any()
        
        # Determine shortage type
        if total_supply < total_demand:
            # Net shortage - insufficient total supply
            net_shortage_products.add(pt_code)
        elif has_shortage:
            # Has shortage in some periods but total supply is sufficient
            # This is a timing gap issue
            timing_gap_products.add(pt_code)
        
        # Alternative check using backlog if available
        if 'backlog_to_next' in product_df.columns:
            # Check final backlog
            final_backlog = product_df['backlog_to_next'].iloc[-1] if not product_df.empty else 0
            if final_backlog > 0:
                # If there's backlog at the end, it's a net shortage
                net_shortage_products.add(pt_code)
                # Remove from timing gap if it was there
                timing_gap_products.discard(pt_code)
    
    logger.info(f"Categorization complete: {len(net_shortage_products)} net shortage, "
                f"{len(timing_gap_products)} timing gap products")
    
    return {
        'net_shortage': net_shortage_products,
        'timing_gap': timing_gap_products
    }


def get_shortage_summary(gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of shortage categorization with actionable insights
    
    Args:
        gap_df: GAP analysis dataframe
    
    Returns:
        Summary dataframe with shortage categorization and recommended actions
    """
    
    if gap_df.empty:
        return pd.DataFrame()
    
    categorization = categorize_shortage_type(gap_df)
    summary_data = []
    
    for pt_code in gap_df['pt_code'].unique():
        product_df = gap_df[gap_df['pt_code'] == pt_code]
        
        # Basic info
        product_name = product_df['product_name'].iloc[0] if 'product_name' in product_df.columns else ''
        brand = product_df['brand'].iloc[0] if 'brand' in product_df.columns else ''
        
        # Calculate metrics
        total_demand = product_df['total_demand_qty'].sum()
        total_supply = product_df['supply_in_period'].sum()
        net_position = total_supply - total_demand
        
        # Count shortage periods
        shortage_periods = (product_df['gap_quantity'] < 0).sum()
        total_periods = len(product_df)
        
        # Maximum shortage in any period
        max_shortage = abs(product_df[product_df['gap_quantity'] < 0]['gap_quantity'].min()) if shortage_periods > 0 else 0
        
        # Determine category and action
        if pt_code in categorization['net_shortage']:
            category = "Net Shortage"
            action = "Place New Order"
            priority = "High"
        elif pt_code in categorization['timing_gap']:
            category = "Timing Gap"
            action = "Expedite/Reschedule"
            priority = "Medium"
        else:
            category = "Sufficient"
            action = "Monitor"
            priority = "Low"
        
        summary_data.append({
            'pt_code': pt_code,
            'product_name': product_name,
            'brand': brand,
            'category': category,
            'total_demand': total_demand,
            'total_supply': total_supply,
            'net_position': net_position,
            'shortage_periods': shortage_periods,
            'total_periods': total_periods,
            'max_shortage': max_shortage,
            'recommended_action': action,
            'priority': priority
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by priority and net position
    priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
    summary_df['priority_sort'] = summary_df['priority'].map(priority_order)
    summary_df = summary_df.sort_values(['priority_sort', 'net_position'])
    summary_df = summary_df.drop(columns=['priority_sort'])
    
    return summary_df


def identify_expedite_candidates(gap_df: pd.DataFrame, 
                                supply_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Identify which supply orders could be expedited to resolve timing gaps
    
    Args:
        gap_df: GAP analysis dataframe
        supply_df: Optional supply dataframe with order details
    
    Returns:
        Dataframe of expedite candidates with recommended actions
    """
    
    categorization = categorize_shortage_type(gap_df)
    timing_gap_products = categorization['timing_gap']
    
    if not timing_gap_products or supply_df is None or supply_df.empty:
        return pd.DataFrame()
    
    expedite_candidates = []
    
    for pt_code in timing_gap_products:
        # Get product GAP data
        product_gap = gap_df[gap_df['pt_code'] == pt_code].copy()
        
        # Find first shortage period
        shortage_periods = product_gap[product_gap['gap_quantity'] < 0]
        if shortage_periods.empty:
            continue
            
        first_shortage_period = shortage_periods.iloc[0]['period']
        shortage_amount = abs(shortage_periods.iloc[0]['gap_quantity'])
        
        # Find supply that could be expedited
        product_supply = supply_df[supply_df['pt_code'] == pt_code].copy()
        
        # Look for supply arriving after the shortage period
        # This is simplified - in practice you'd need to parse periods properly
        future_supply = product_supply[product_supply['source_type'].isin(['Pending PO', 'Pending CAN'])]
        
        if not future_supply.empty:
            for _, supply_row in future_supply.iterrows():
                expedite_candidates.append({
                    'pt_code': pt_code,
                    'product_name': product_gap['product_name'].iloc[0] if 'product_name' in product_gap.columns else '',
                    'shortage_period': first_shortage_period,
                    'shortage_qty': shortage_amount,
                    'supply_source': supply_row.get('source_type', ''),
                    'supply_number': supply_row.get('supply_number', ''),
                    'supply_qty': supply_row.get('quantity', 0),
                    'current_eta': supply_row.get('date_ref', ''),
                    'action': 'Expedite delivery to before ' + str(first_shortage_period)
                })
    
    return pd.DataFrame(expedite_candidates)


def calculate_order_requirements(gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate new order requirements for products with net shortage
    
    Args:
        gap_df: GAP analysis dataframe
    
    Returns:
        Dataframe with order requirements for each product
    """
    
    categorization = categorize_shortage_type(gap_df)
    net_shortage_products = categorization['net_shortage']
    
    if not net_shortage_products:
        return pd.DataFrame()
    
    order_requirements = []
    
    for pt_code in net_shortage_products:
        product_gap = gap_df[gap_df['pt_code'] == pt_code]
        
        # Calculate total shortage
        total_demand = product_gap['total_demand_qty'].sum()
        total_supply = product_gap['supply_in_period'].sum()
        net_shortage = total_demand - total_supply
        
        # Find when shortage starts
        shortage_periods = product_gap[product_gap['gap_quantity'] < 0]
        first_shortage_period = shortage_periods.iloc[0]['period'] if not shortage_periods.empty else None
        
        # Check if using backlog tracking
        if 'backlog_to_next' in product_gap.columns:
            final_backlog = product_gap['backlog_to_next'].iloc[-1] if not product_gap.empty else 0
            order_qty = max(net_shortage, final_backlog)
        else:
            order_qty = net_shortage
        
        order_requirements.append({
            'pt_code': pt_code,
            'product_name': product_gap['product_name'].iloc[0] if 'product_name' in product_gap.columns else '',
            'brand': product_gap['brand'].iloc[0] if 'brand' in product_gap.columns else '',
            'package_size': product_gap['package_size'].iloc[0] if 'package_size' in product_gap.columns else '',
            'standard_uom': product_gap['standard_uom'].iloc[0] if 'standard_uom' in product_gap.columns else '',
            'order_quantity': order_qty,
            'first_shortage_period': first_shortage_period,
            'total_demand': total_demand,
            'total_supply': total_supply,
            'coverage_periods': len(product_gap),
            'urgency': 'Immediate' if first_shortage_period else 'Plan'
        })
    
    order_df = pd.DataFrame(order_requirements)
    
    # Sort by order quantity descending
    if not order_df.empty:
        order_df = order_df.sort_values('order_quantity', ascending=False)
    
    return order_df


def get_action_summary(gap_df: pd.DataFrame, supply_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
    """
    Get comprehensive action summary for all shortage types
    
    Args:
        gap_df: GAP analysis dataframe
        supply_df: Optional supply dataframe
    
    Returns:
        Dictionary with action summaries:
        - 'overview': Overall categorization summary
        - 'order_requirements': New orders needed
        - 'expedite_candidates': Orders to expedite/reschedule
    """
    
    return {
        'overview': get_shortage_summary(gap_df),
        'order_requirements': calculate_order_requirements(gap_df),
        'expedite_candidates': identify_expedite_candidates(gap_df, supply_df)
    }