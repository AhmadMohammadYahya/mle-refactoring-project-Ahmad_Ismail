"""Cleaning functions for the King County housing dataset."""

import pandas as pd


def remove_implausible_houses(data: pd.DataFrame) -> pd.DataFrame:
    """Remove houses with an implausible bedroom count.

    Returns a cleaned copy without modifying the caller's DataFrame.
    """
    cleaned = data.copy()
    return cleaned.loc[cleaned["bedrooms"] != 33].copy()


def fill_missing_view(data: pd.DataFrame) -> pd.DataFrame:
    """Replace missing view values with zero."""
    cleaned = data.copy()
    cleaned["view"] = cleaned["view"].fillna(0)
    return cleaned


def fill_missing_waterfront(data: pd.DataFrame) -> pd.DataFrame:
    """Replace missing waterfront values with zero."""
    cleaned = data.copy()
    cleaned["waterfront"] = cleaned["waterfront"].fillna(0)
    return cleaned


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the dataset's cleaning steps in sequence."""
    cleaned = remove_implausible_houses(data)
    cleaned = fill_missing_view(cleaned)
    cleaned = fill_missing_waterfront(cleaned)
    return cleaned
