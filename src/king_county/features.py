"""Feature-engineering functions for the King County housing dataset."""

import numpy as np
import pandas as pd


def calculate_sqft_basement(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate basement area from total and above-ground living area."""
    features = data.copy()
    features["sqft_basement"] = (
        features["sqft_living"] - features["sqft_above"]
    )
    return features


def calculate_last_known_change(data: pd.DataFrame) -> pd.DataFrame:
    """Use the renovation year when available; otherwise use the build year."""
    features = data.copy()

    renovation_year = features["yr_renovated"].fillna(0)

    features["last_known_change"] = np.where(
        renovation_year == 0,
        features["yr_built"],
        renovation_year,
    ).astype(int)

    features = features.drop(columns=["yr_renovated", "yr_built"])
    return features


def calculate_sqft_price(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the price feature using the original notebook formula."""
    features = data.copy()

    features["sqft_price"] = (
        features["price"]
        / (features["sqft_living"] + features["sqft_lot"])
    ).round(2)

    return features


def calculate_center_distance(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate approximate distance from the notebook's reference center."""
    features = data.copy()

    features["delta_lat"] = np.absolute(47.62774 - features["lat"])
    features["delta_long"] = np.absolute(-122.24194 - features["long"])

    features["center_distance"] = (
        (
            (
                features["delta_long"]
                * np.cos(np.radians(47.6219))
            ) ** 2
            + features["delta_lat"] ** 2
        ) ** (1 / 2)
        * 2
        * np.pi
        * 6378
        / 360
    )

    return features


def dist(
    long: float,
    lat: float,
    ref_long: float,
    ref_lat: float,
) -> float:
    """Calculate approximate distance in kilometres to a reference point."""
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))

    return (
        np.sqrt(delta_long_corr**2 + delta_lat**2)
        * 2
        * np.pi
        * 6378
        / 360
    )


def calculate_water_distance(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance to the nearest house marked as waterfront."""
    features = data.copy()

    waterfront_houses = features.loc[
        features["waterfront"] == 1, ["long", "lat"]
    ].to_numpy()

    if len(waterfront_houses) == 0:
        raise ValueError(
            "Cannot calculate water_distance: no waterfront houses found."
        )

    property_coordinates = features[["long", "lat"]].to_numpy()

    distances = np.array([
        [
            dist(long, lat, ref_long, ref_lat)
            for ref_long, ref_lat in waterfront_houses
        ]
        for long, lat in property_coordinates
    ])

    features["water_distance"] = distances.min(axis=1)
    return features
