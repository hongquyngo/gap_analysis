# utils/gap/formatters.py

"""
Formatting module for GAP Analysis System - Minimal Version
Contains only actively used formatting functions
"""

import pandas as pd
from typing import Union, Optional

# Formatting configuration
DEFAULT_DECIMAL_PLACES = 0

# Currency symbols
CURRENCY_SYMBOLS = {
    'USD': '$',
    'VND': '₫',
    'EUR': '€',
    'GBP': '£'
}

# Number abbreviations for large numbers
NUMBER_ABBREVIATIONS = [
    (1e9, 'B'),  # Billion
    (1e6, 'M'),  # Million
    (1e3, 'K'),  # Thousand
]


class GAPFormatter:
    """Handles formatting for GAP analysis display"""
    
    @staticmethod
    def format_number(
        value: Union[float, int, None], 
        decimals: int = DEFAULT_DECIMAL_PLACES,
        show_sign: bool = False,
        abbreviate: bool = False
    ) -> str:
        """
        Format number with thousand separators and optional features
        
        Args:
            value: Number to format
            decimals: Number of decimal places
            show_sign: Whether to show + sign for positive numbers
            abbreviate: Whether to abbreviate large numbers (1.5M instead of 1,500,000)
            
        Returns:
            Formatted string
        
        Used by:
            - charts.py: create_kpi_cards(), create_top_shortage_bar_chart(), 
                        create_supply_demand_comparison()
            - main page: prepare_display_dataframe()
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        # Handle abbreviation for large numbers
        if abbreviate and abs(value) >= 1000:
            for threshold, suffix in NUMBER_ABBREVIATIONS:
                if abs(value) >= threshold:
                    abbreviated_value = value / threshold
                    if decimals == 0 and abbreviated_value.is_integer():
                        formatted = f"{int(abbreviated_value)}{suffix}"
                    else:
                        formatted = f"{abbreviated_value:.{max(1, decimals)}f}{suffix}"
                    return f"+{formatted}" if show_sign and value > 0 else formatted
        
        # Standard formatting
        if decimals == 0:
            formatted = f"{int(value):,}"
        else:
            formatted = f"{value:,.{decimals}f}"
        
        if show_sign and value > 0:
            formatted = f"+{formatted}"
        
        return formatted
    
    @staticmethod
    def format_currency(
        value: Union[float, int, None],
        currency: str = "USD",
        decimals: Optional[int] = None,
        abbreviate: bool = False
    ) -> str:
        """
        Format value as currency
        
        Args:
            value: Amount to format
            currency: Currency code (USD, VND, EUR, etc.)
            decimals: Decimal places (None = auto based on currency)
            abbreviate: Whether to abbreviate large amounts
            
        Returns:
            Formatted currency string
            
        Used by:
            - charts.py: create_kpi_cards() for "At Risk Value" metric
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        # Determine decimal places based on currency
        if decimals is None:
            decimals = 0 if currency == 'VND' else 2
        
        # Get currency symbol
        symbol = CURRENCY_SYMBOLS.get(currency, '')
        
        # Format the number part
        if abbreviate and abs(value) >= 1000:
            for threshold, suffix in NUMBER_ABBREVIATIONS:
                if abs(value) >= threshold:
                    abbreviated_value = value / threshold
                    if symbol:
                        return f"{symbol}{abbreviated_value:.{max(1, decimals)}f}{suffix}"
                    else:
                        return f"{abbreviated_value:.{max(1, decimals)}f}{suffix} {currency}"
        
        # Standard currency formatting
        if symbol:
            if currency == 'VND':
                return f"{symbol}{value:,.0f}"
            else:
                return f"{symbol}{value:,.{decimals}f}"
        else:
            return f"{value:,.{decimals}f} {currency}"