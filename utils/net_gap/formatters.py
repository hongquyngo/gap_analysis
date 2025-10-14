# utils/net_gap/formatters.py

"""
Formatting Module - Version 3.0 ENHANCED
- Added export formatting (fix 999 values)
- Support for multiple format modes
- Clean handling of special cases
"""

import pandas as pd
from typing import Union, Optional, Any

# Configuration
DEFAULT_DECIMAL_PLACES = 0

CURRENCY_SYMBOLS = {
    'USD': '$',
    'VND': '₫',
    'EUR': '€',
    'GBP': '£'
}

NUMBER_ABBREVIATIONS = [
    (1e9, 'B'),
    (1e6, 'M'),
    (1e3, 'K'),
]


class GAPFormatter:
    """Handles formatting for GAP analysis display and export"""
    
    @staticmethod
    def format_number(
        value: Union[float, int, None], 
        decimals: int = DEFAULT_DECIMAL_PLACES,
        show_sign: bool = False,
        abbreviate: bool = False
    ) -> str:
        """
        Format number with thousand separators
        
        Args:
            value: Number to format
            decimals: Decimal places
            show_sign: Show + for positive
            abbreviate: Use K/M/B notation
            
        Returns:
            Formatted string
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        # Abbreviation for large numbers
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
            currency: Currency code
            decimals: Decimal places (auto if None)
            abbreviate: Use K/M/B notation
            
        Returns:
            Formatted currency string
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        # Determine decimals
        if decimals is None:
            decimals = 0 if currency == 'VND' else 2
        
        # Get currency symbol
        symbol = CURRENCY_SYMBOLS.get(currency, '')
        
        # Abbreviation
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
    
    @staticmethod
    def format_percentage(
        value: Union[float, int, None],
        decimals: int = 1,
        show_sign: bool = False
    ) -> str:
        """
        Format value as percentage
        
        Args:
            value: Value to format (0-100 scale)
            decimals: Decimal places
            show_sign: Show + for positive
            
        Returns:
            Formatted percentage string
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        if decimals == 0:
            formatted = f"{int(value)}%"
        else:
            formatted = f"{value:.{decimals}f}%"
        
        if show_sign and value > 0:
            formatted = f"+{formatted}"
        
        return formatted
    
    @staticmethod
    def format_for_export(
        value: Any, 
        field_type: str,
        max_value_threshold: float = 999
    ) -> Any:
        """
        Format value for Excel export
        CRITICAL: Handles special values like 999
        
        Args:
            value: Value to format
            field_type: Type of field ('coverage', 'days', 'ratio', 'number', 'currency')
            max_value_threshold: Values >= this become None
            
        Returns:
            Excel-friendly value (None for blanks)
        """
        if pd.isna(value) or value is None:
            return None
        
        # Handle special cases based on field type
        if field_type == 'coverage':
            # Safety coverage, coverage ratio, etc.
            if value >= max_value_threshold:
                return None  # Excel will show as blank
            return round(value, 2)
        
        elif field_type == 'days':
            # Days of supply, days to expiry, etc.
            if value >= max_value_threshold:
                return None  # Excel will show as blank
            return round(value, 1)
        
        elif field_type == 'ratio':
            # Ratios that can be excessive
            if value > 10:  # >1000% is excessive
                return None
            return round(value, 2)
        
        elif field_type == 'percentage':
            # Regular percentages
            if value >= max_value_threshold:
                return None
            return round(value, 1)
        
        elif field_type == 'number':
            # Regular numbers
            if isinstance(value, float):
                return round(value, 2)
            return value
        
        elif field_type == 'currency':
            # Currency values
            return round(value, 2) if isinstance(value, float) else value
        
        else:
            # Default: return as-is for numeric, None for extreme values
            if isinstance(value, (int, float)):
                if abs(value) >= max_value_threshold:
                    return None
                return round(value, 2) if isinstance(value, float) else value
            return value
    
    @staticmethod
    def format_for_display(
        value: Any,
        field_name: str,
        formatter_instance = None
    ) -> str:
        """
        Format value for UI display
        
        Args:
            value: Value to format
            field_name: Name of the field
            formatter_instance: GAPFormatter instance (for recursive calls)
            
        Returns:
            Formatted display string
        """
        if pd.isna(value) or value is None:
            return "—"
        
        if formatter_instance is None:
            formatter_instance = GAPFormatter()
        
        # Map field names to format types
        if field_name in ['safety_coverage', 'safety_cov']:
            if value >= 999:
                return "N/A"
            return f"{value:.1f}x"
        
        elif field_name in ['days_of_supply', 'days_to_expiry']:
            if value >= 999:
                return ">1 year"
            if value >= 365:
                return f"{value/365:.1f} years"
            return f"{value:.0f} days"
        
        elif field_name in ['coverage_ratio', 'coverage']:
            if value > 10:
                return "Excess"
            return f"{value*100:.0f}%"
        
        elif field_name in ['gap_percentage', 'gap_%']:
            return formatter_instance.format_percentage(value, show_sign=True)
        
        elif 'value' in field_name.lower() or 'amount' in field_name.lower():
            return formatter_instance.format_currency(value, abbreviate=True)
        
        elif 'quantity' in field_name.lower() or 'qty' in field_name.lower():
            return formatter_instance.format_number(value)
        
        else:
            # Default formatting
            if isinstance(value, float):
                return f"{value:.2f}"
            elif isinstance(value, int):
                return formatter_instance.format_number(value)
            return str(value)