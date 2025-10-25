"""
Debug Script to Find Row Count Differences
Run this to identify where the 2-product difference comes from
"""

import streamlit as st
import pandas as pd
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_gap_calculation():
    """
    Run GAP calculation with detailed logging to find duplicates
    """
    
    from utils.net_gap.data_loader import GAPDataLoader
    from utils.net_gap.calculator import GAPCalculator
    
    # Initialize
    data_loader = GAPDataLoader()
    calculator = GAPCalculator()
    
    # Test filters (adjust as needed)
    filter_values = {
        'entity': None,
        'exclude_entity': False,
        'products_tuple': None,
        'brands_tuple': None,
        'exclude_products': False,
        'exclude_brands': False,
        'exclude_expired': False,  # Include expired to test
        'group_by': 'product',
        'supply_sources': ['INVENTORY', 'CAN_PENDING', 'TRANSFER', 'PURCHASE_ORDER'],
        'demand_sources': ['OC_PENDING', 'FORECAST'],
        'include_safety': True
    }
    
    print("=" * 60)
    print("STARTING DEBUG RUN")
    print("=" * 60)
    
    # Load data
    print("\n1. LOADING SUPPLY DATA...")
    supply_df = data_loader.load_supply_data(
        entity_name=filter_values.get('entity'),
        exclude_entity=filter_values.get('exclude_entity', False),
        product_ids=filter_values.get('products_tuple'),
        brands=filter_values.get('brands_tuple'),
        exclude_products=filter_values.get('exclude_products', False),
        exclude_brands=filter_values.get('exclude_brands', False),
        exclude_expired=filter_values.get('exclude_expired', True)
    )
    print(f"   Supply rows: {len(supply_df)}")
    if 'product_id' in supply_df.columns:
        print(f"   Unique products: {supply_df['product_id'].nunique()}")
        
        # Check for inconsistent product data
        product_info = supply_df.groupby('product_id').agg({
            'product_name': lambda x: len(x.unique()),
            'brand': lambda x: len(x.unique()) if 'brand' in supply_df.columns else 0,
            'pt_code': lambda x: len(x.unique()) if 'pt_code' in supply_df.columns else 0
        }).rename(columns={
            'product_name': 'unique_names',
            'brand': 'unique_brands',
            'pt_code': 'unique_pt_codes'
        })
        
        inconsistent = product_info[
            (product_info['unique_names'] > 1) | 
            (product_info['unique_brands'] > 1) |
            (product_info['unique_pt_codes'] > 1)
        ]
        
        if not inconsistent.empty:
            print(f"   ⚠️ INCONSISTENT DATA for {len(inconsistent)} products:")
            print(inconsistent.head())
    
    print("\n2. LOADING EXPIRED INVENTORY...")
    expired_df = data_loader.load_expired_inventory_details(
        entity_name=filter_values.get('entity'),
        exclude_entity=filter_values.get('exclude_entity', False),
        product_ids=filter_values.get('products_tuple'),
        brands=filter_values.get('brands_tuple'),
        exclude_products=filter_values.get('exclude_products', False),
        exclude_brands=filter_values.get('exclude_brands', False)
    )
    print(f"   Expired inventory rows: {len(expired_df)}")
    if not expired_df.empty and 'product_id' in expired_df.columns:
        print(f"   Unique products with expired: {expired_df['product_id'].nunique()}")
        
        # Check for duplicates
        duplicates = expired_df[expired_df['product_id'].duplicated()]
        if not duplicates.empty:
            print(f"   🔴 DUPLICATE product_ids in expired: {len(duplicates)}")
            print(duplicates[['product_id', 'product_name']].head())
    
    print("\n3. LOADING DEMAND DATA...")
    demand_df = data_loader.load_demand_data(
        entity_name=filter_values.get('entity'),
        exclude_entity=filter_values.get('exclude_entity', False),
        product_ids=filter_values.get('products_tuple'),
        brands=filter_values.get('brands_tuple'),
        exclude_products=filter_values.get('exclude_products', False),
        exclude_brands=filter_values.get('exclude_brands', False)
    )
    print(f"   Demand rows: {len(demand_df)}")
    if 'product_id' in demand_df.columns:
        print(f"   Unique products: {demand_df['product_id'].nunique()}")
    
    print("\n4. LOADING SAFETY STOCK...")
    safety_df = data_loader.load_safety_stock_data(
        entity_name=filter_values.get('entity'),
        exclude_entity=filter_values.get('exclude_entity', False),
        product_ids=filter_values.get('products_tuple')
    )
    print(f"   Safety stock rows: {len(safety_df)}")
    if not safety_df.empty and 'product_id' in safety_df.columns:
        print(f"   Unique products: {safety_df['product_id'].nunique()}")
        
        # Check for duplicates
        duplicates = safety_df[safety_df['product_id'].duplicated()]
        if not duplicates.empty:
            print(f"   🔴 DUPLICATE product_ids in safety: {len(duplicates)}")
    
    print("\n5. CALCULATING GAP...")
    result = calculator.calculate_net_gap(
        supply_df=supply_df,
        demand_df=demand_df,
        safety_stock_df=safety_df,
        expired_inventory_df=expired_df,
        group_by=filter_values.get('group_by', 'product'),
        selected_supply_sources=filter_values.get('supply_sources'),
        selected_demand_sources=filter_values.get('demand_sources'),
        include_safety_stock=filter_values.get('include_safety', False)
    )
    
    print(f"\n6. FINAL RESULTS:")
    print(f"   Total products in result: {result.metrics['total_products']}")
    print(f"   Shortage items: {result.metrics['shortage_items']}")
    print(f"   Critical items: {result.metrics['critical_items']}")
    print(f"   At Risk Value: ${result.metrics['at_risk_value_usd']:,.2f}")
    
    # Check for duplicates in final result
    if 'product_id' in result.gap_df.columns:
        duplicates = result.gap_df[result.gap_df['product_id'].duplicated()]
        if not duplicates.empty:
            print(f"\n   🔴 DUPLICATE product_ids in FINAL RESULT: {len(duplicates)}")
            print(duplicates[['product_id', 'product_name', 'net_gap']].head())
    
    print("\n" + "=" * 60)
    print("DEBUG RUN COMPLETE")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    debug_gap_calculation()