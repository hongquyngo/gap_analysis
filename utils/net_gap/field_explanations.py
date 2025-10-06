# utils/net_gap/field_explanations.py

"""
Field explanations and calculation formulas for GAP Analysis System
Focus on calculation logic and formulas for each field
"""

import streamlit as st
from typing import Dict, Optional


# Field tooltips dictionary with calculation formulas
FIELD_TOOLTIPS = {
    # Basic fields
    'pt_code': 'Product code identifier in the system',
    'product_name': 'Full name/description of the product',
    'brand': 'Brand or manufacturer of the product',
    
    # Supply fields
    'Supply': 'Total Supply = Inventory + CAN Pending + Warehouse Transfer + Purchase Order',
    'supply_inventory': 'Current stock in warehouse (available immediately)',
    'supply_can_pending': 'Goods arrived awaiting stock-in (1-3 days to available)',
    'supply_warehouse_transfer': 'Goods being transferred between warehouses (2-5 days)',
    'supply_purchase_order': 'Incoming goods from purchase orders (7-30 days)',
    
    # Demand fields
    'Demand': 'Total Demand = OC Pending + Forecast',
    'demand_oc_pending': 'Confirmed customer orders pending delivery',
    'demand_forecast': 'Customer forecasted demand not yet converted to orders',
    
    # GAP calculations
    'Net GAP': 'Net GAP = Supply - Demand (or Available Supply - Demand if safety enabled)',
    'Coverage': 'Coverage = (Supply ÷ Demand) × 100%',
    'GAP %': 'GAP % = (Net GAP ÷ Demand) × 100%',
    'Status': 'Auto-classification based on coverage ratio and thresholds',
    'Action': 'System recommendation based on GAP status and coverage',
    
    # Safety stock fields
    'Safety Stock': 'Min inventory = Lead Time × Daily Demand + Safety Buffer',
    'Available': 'Available = Max(0, Total Supply - Safety Stock)',
    'True GAP': 'True GAP = Total Supply - Demand (ignoring safety stock)',
    'Safety Cov': 'Safety Coverage = Current Inventory ÷ Safety Stock',
    'Reorder': 'Reorder Flag = (Current Inventory ≤ Reorder Point)',
    
    # Financial fields
    'At Risk Value': 'At Risk = |Shortage Qty| × Selling Price (when GAP < 0)',
    'GAP Value': 'GAP Value = Net GAP × Unit Cost',
    
    # Additional metrics
    'priority': 'Priority level: 1=Critical, 2=High, 3=Medium, 4=Low, 99=OK',
    'customer_count': 'Number of unique customers with demand',
    'avg_days_to_required': 'Average days until demand due date',
}


def get_field_tooltip(field_name: str) -> Optional[str]:
    """
    Get tooltip text for a specific field
    
    Args:
        field_name: Name of the field
        
    Returns:
        Tooltip text with formula or None if not found
    """
    # Try exact match first
    if field_name in FIELD_TOOLTIPS:
        return FIELD_TOOLTIPS[field_name]
    
    # Try lowercase match
    field_lower = field_name.lower()
    for key, value in FIELD_TOOLTIPS.items():
        if key.lower() == field_lower:
            return value
    
    return None


def show_field_explanations(include_safety: bool = False):
    """
    Display calculation formulas and field explanations
    Focus on mathematical formulas and calculation logic
    
    Args:
        include_safety: Whether safety stock is included in the analysis
    """
    st.markdown("### 📐 Calculation Formulas & Field Logic")
    
    # Core GAP Calculations
    st.markdown("#### **1. Core GAP Calculations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Standard Mode (No Safety Stock):**")
        st.code("""
# Basic Formulas
Total Supply = Inventory + CAN_Pending + 
               Warehouse_Transfer + Purchase_Order

Total Demand = OC_Pending + Forecast

Net GAP = Total Supply - Total Demand

Coverage = (Total Supply ÷ Total Demand) × 100%
         = 0% if no supply
         = "No Demand" if demand = 0

GAP % = (Net GAP ÷ Total Demand) × 100%
      = Shows relative size of gap
        """, language='text')
    
    with col2:
        if include_safety:
            st.markdown("**With Safety Stock Mode:**")
            st.code("""
# Safety-Adjusted Formulas
Safety Stock = Lead_Time_Days × Avg_Daily_Demand 
             + Safety_Buffer

Available Supply = Max(0, Total Supply - Safety Stock)

Net GAP = Available Supply - Total Demand

True GAP = Total Supply - Total Demand
         (ignores safety for comparison)

Safety Coverage = Current Inventory ÷ Safety Stock
            """, language='text')
        else:
            st.markdown("**Coverage Interpretation:**")
            st.code("""
Coverage Range → Status
0%           → No Supply (Critical)
<50%         → Severe Shortage
50-70%       → High Shortage  
70-90%       → Moderate Shortage
90-110%      → Balanced ✓
110-150%     → Light Surplus
150-300%     → Moderate Surplus
>300%        → Excessive Surplus
            """, language='text')
    
    st.divider()
    
    # Supply Priority & Lead Times
    st.markdown("#### **2. Supply Sources Priority & Availability**")
    
    supply_calc = """
    | Source | Priority | Lead Time | Calculation |
    |--------|----------|-----------|-------------|
    | Inventory | P1 | 0 days | `remaining_quantity` from inventory table |
    | CAN Pending | P2 | 1-3 days | `pending_quantity` awaiting stock-in |
    | Warehouse Transfer | P3 | 2-5 days | `transfer_quantity` in transit |
    | Purchase Order | P4 | 7-30 days | `pending_arrival_quantity` from PO |
    
    **Aggregation Formula:**
    ```
    For each product:
      Supply[product] = SUM(quantity WHERE supply_source IN selected_sources)
    ```
    """
    st.markdown(supply_calc)
    
    st.divider()
    
    # Status Classification Logic
    st.markdown("#### **3. Status Classification Algorithm**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Standard Status Logic:**")
        st.code("""
IF demand = 0:
    RETURN "NO_DEMAND"
ELIF coverage < 50%:
    RETURN "SEVERE_SHORTAGE"
ELIF coverage < 70%:
    RETURN "HIGH_SHORTAGE"
ELIF coverage < 90%:
    RETURN "MODERATE_SHORTAGE"
ELIF coverage <= 110%:
    RETURN "BALANCED"
ELIF coverage <= 150%:
    RETURN "LIGHT_SURPLUS"
ELIF coverage <= 300%:
    RETURN "MODERATE_SURPLUS"
ELSE:
    RETURN "SEVERE_SURPLUS"
        """, language='text')
    
    with col2:
        if include_safety:
            st.markdown("**Safety Stock Status (Priority Override):**")
            st.code("""
# Checked BEFORE standard logic
IF inventory < safety_stock × 0.5:
    RETURN "CRITICAL_BREACH" (P1)
ELIF inventory < safety_stock:
    RETURN "BELOW_SAFETY" (P1)
ELIF inventory <= reorder_point:
    RETURN "AT_REORDER" (P2)
ELIF has_expired_items:
    RETURN "HAS_EXPIRED" (P1)
ELSE:
    → Continue to standard logic
            """, language='text')
        else:
            st.markdown("**Priority Assignment:**")
            st.code("""
Priority = CASE gap_status:
    CRITICAL_BREACH, 
    SEVERE_SHORTAGE → Priority 1
    
    HIGH_SHORTAGE,
    AT_REORDER     → Priority 2
    
    MODERATE_SHORTAGE,
    MODERATE_SURPLUS → Priority 3
    
    LIGHT_SURPLUS   → Priority 4
    
    BALANCED        → Priority 99
            """, language='text')
    
    st.divider()
    
    # Financial Calculations
    st.markdown("#### **4. Financial Impact Calculations**")
    
    financial_formulas = """
    ```python
    # At Risk Value (Revenue Impact)
    IF net_gap < 0:  # Shortage case
        at_risk_value = ABS(net_gap) × selling_price_per_unit
    ELSE:
        at_risk_value = 0
    
    # GAP Value (Inventory Value)
    gap_value = net_gap × unit_cost
              # Positive = excess inventory value
              # Negative = shortage inventory value
    
    # Supply/Demand Values
    supply_value = total_supply × avg_unit_cost
    demand_value = total_demand × avg_selling_price
    
    # For aggregated view:
    weighted_avg_cost = SUM(cost × qty) ÷ SUM(qty)
    ```
    """
    st.markdown(financial_formulas)
    
    st.divider()
    
    # Action Generation Logic
    st.markdown("#### **5. Suggested Action Algorithm**")
    
    action_logic = """
    ```python
    def get_suggested_action(status, gap, coverage):
        actions = {
            # Critical Actions (P1)
            'CRITICAL_BREACH': f"⚠️ CRITICAL: Below safety minimum. EXPEDITE NOW!",
            'SEVERE_SHORTAGE': f"🚨 URGENT: Need {abs(gap)} units. Expedite orders!",
            
            # High Priority (P2)  
            'HIGH_SHORTAGE': f"⚡ Need {abs(gap)} units. Create PO within 2 days",
            'AT_REORDER': f"🔄 At reorder point. Order {reorder_qty} units",
            
            # Medium Priority (P3)
            'MODERATE_SHORTAGE': f"📋 Need {abs(gap)} units. Plan replenishment",
            'MODERATE_SURPLUS': f"📦 Surplus {gap} units. Reduce orders",
            
            # Low Priority (P4)
            'LIGHT_SURPLUS': f"📈 Minor surplus ({gap} units). Monitor",
            
            # No Action (P99)
            'BALANCED': "✅ Supply-demand balanced"
        }
        return actions.get(status, "Review manually")
    ```
    """
    st.markdown(action_logic)
    
    # Quick Reference Table
    st.markdown("#### **6. Quick Reference - Key Thresholds**")
    
    threshold_table = """
    | Metric | Critical | Warning | OK | Good |
    |--------|----------|---------|-----|------|
    | Coverage | <50% | 50-90% | 90-110% | 90-110% |
    | Safety Coverage | <0.5x | 0.5-1.0x | 1.0-1.5x | >1.5x |
    | Days of Supply | <3 days | 3-7 days | 7-30 days | 15-45 days |
    | GAP % | <-50% | -50% to -10% | -10% to +10% | ±10% |
    | At Risk Value | >$100K | $10K-100K | $1K-10K | <$1K |
    """
    st.markdown(threshold_table)
    
    # Special Cases
    st.markdown("#### **7. Special Cases & Edge Conditions**")
    
    st.info("""
    **Edge Case Handling:**
    
    • **Division by Zero**: Coverage = 999 if demand = 0 but supply > 0
    • **No Activity**: Coverage = 0 if both supply and demand = 0  
    • **Expired Stock**: Excluded from available supply in calculations
    • **Negative Values**: Net GAP can be negative (shortage), but quantities are always ≥ 0
    • **Maximum Display**: Coverage capped at 999% for display (actual value preserved internally)
    • **Reorder Point**: Only checked if safety stock is configured and > 0
    • **Customer Count**: DISTINCT count to avoid duplication across multiple orders
    """)
    
    # Example Calculations
    st.markdown("#### **8. Example Calculations**")
    
    tab1, tab2, tab3 = st.tabs(["Shortage Example", "Surplus Example", "Safety Stock Example"])
    
    with tab1:
        st.code("""
Product: ABC-123
Supply: Inventory=50, CAN=0, Transfer=20, PO=30
Demand: OC=150, Forecast=50

Calculations:
Total Supply = 50 + 0 + 20 + 30 = 100 units
Total Demand = 150 + 50 = 200 units
Net GAP = 100 - 200 = -100 units (SHORTAGE)
Coverage = (100 ÷ 200) × 100% = 50%
GAP % = (-100 ÷ 200) × 100% = -50%
Status = SEVERE_SHORTAGE (coverage < 50%)
At Risk = 100 × $10 = $1,000
Action = "🚨 URGENT: Need 100 units. Expedite orders!"
        """, language='text')
    
    with tab2:
        st.code("""
Product: XYZ-789
Supply: Inventory=500, CAN=100, Transfer=0, PO=0
Demand: OC=150, Forecast=50

Calculations:
Total Supply = 500 + 100 + 0 + 0 = 600 units
Total Demand = 150 + 50 = 200 units
Net GAP = 600 - 200 = +400 units (SURPLUS)
Coverage = (600 ÷ 200) × 100% = 300%
GAP % = (400 ÷ 200) × 100% = +200%
Status = MODERATE_SURPLUS (150% < coverage ≤ 300%)
At Risk = 0 (no shortage)
Action = "📦 Surplus 400 units. Reduce orders"
        """, language='text')
    
    with tab3:
        if include_safety:
            st.code("""
Product: DEF-456
Supply: Inventory=100, CAN=0, Transfer=0, PO=200
Demand: OC=180, Forecast=20
Safety Stock: 80 units, Reorder Point: 120 units

Calculations:
Total Supply = 100 + 0 + 0 + 200 = 300 units
Total Demand = 180 + 20 = 200 units
Safety Stock = 80 units

Available Supply = Max(0, 300 - 80) = 220 units
Net GAP = 220 - 200 = +20 units (SLIGHT SURPLUS)
True GAP = 300 - 200 = +100 units (actual surplus)

Current Inventory = 100 units
Safety Coverage = 100 ÷ 80 = 1.25x (OK)
Below Reorder = 100 ≤ 120 = TRUE

Status = AT_REORDER (inventory ≤ reorder point)
Action = "🔄 At reorder point. Order 160 units"
            """, language='text')
        else:
            st.info("Enable Safety Stock in filters to see this example")


def get_coverage_interpretation(coverage_percentage: float) -> Dict[str, str]:
    """
    Get interpretation for a coverage percentage value
    
    Args:
        coverage_percentage: Coverage value as percentage
        
    Returns:
        Dictionary with status, formula, and action
    """
    if coverage_percentage == 0:
        return {
            'status': '🔴 No Supply',
            'formula': 'Supply = 0, Demand > 0',
            'action': 'Expedite all orders immediately'
        }
    elif coverage_percentage < 50:
        return {
            'status': '🔴 Severe Shortage',
            'formula': f'Supply ÷ Demand = {coverage_percentage:.0f}%',
            'action': 'Highest priority - expedite orders'
        }
    elif coverage_percentage < 70:
        return {
            'status': '🟠 High Shortage',
            'formula': f'Supply ÷ Demand = {coverage_percentage:.0f}%',
            'action': 'Create PO within 1-2 days'
        }
    elif coverage_percentage < 90:
        return {
            'status': '🟡 Moderate Shortage',
            'formula': f'Supply ÷ Demand = {coverage_percentage:.0f}%',
            'action': 'Plan replenishment soon'
        }
    elif coverage_percentage <= 110:
        return {
            'status': '🟢 Balanced',
            'formula': f'Supply ÷ Demand = {coverage_percentage:.0f}%',
            'action': 'No immediate action needed'
        }
    elif coverage_percentage <= 150:
        return {
            'status': '🔵 Light Surplus',
            'formula': f'Supply ÷ Demand = {coverage_percentage:.0f}%',
            'action': 'Acceptable - monitor levels'
        }
    elif coverage_percentage <= 300:
        return {
            'status': '🟣 Moderate Surplus',
            'formula': f'Supply ÷ Demand = {coverage_percentage:.0f}%',
            'action': 'Review and reduce orders'
        }
    else:
        return {
            'status': '⚪ Excessive Surplus',
            'formula': f'Supply ÷ Demand = {coverage_percentage:.0f}%',
            'action': 'Stop ordering immediately'
        }