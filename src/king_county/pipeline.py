"""Reusable preprocessing pipeline for King County house prices."""

import pandas as pd

from src.king_county.cleaning import clean_data
from src.king_county.features import (
    calculate_center_distance,
    calculate_last_known_change,
    calculate_sqft_basement,
    calculate_sqft_price,
    calculate_water_distance,
)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw house data and calculate engineered features."""
    processed = clean_data(data)

    processed = calculate_sqft_basement(processed)
    processed = calculate_last_known_change(processed)
    processed = calculate_sqft_price(processed)
    processed = calculate_center_distance(processed)
    processed = calculate_water_distance(processed)

    return processed
