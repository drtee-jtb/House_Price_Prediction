from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(row_count: int = 500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    lot_area = rng.integers(4500, 18001, size=row_count)
    overall_qual = rng.integers(4, 11, size=row_count)
    overall_cond = rng.integers(4, 10, size=row_count)
    year_built = rng.integers(1960, 2023, size=row_count)
    year_remod = np.clip(year_built + rng.integers(0, 16, size=row_count), 1970, 2024)
    gr_liv_area = rng.integers(900, 3201, size=row_count)
    full_bath = rng.integers(1, 4, size=row_count)
    half_bath = rng.integers(0, 3, size=row_count)
    bedrooms = rng.integers(2, 7, size=row_count)
    total_rooms = np.maximum(bedrooms + rng.integers(2, 6, size=row_count), 4)
    fireplaces = rng.integers(0, 3, size=row_count)
    garage_cars = rng.integers(0, 4, size=row_count)
    garage_area = garage_cars * rng.integers(180, 320, size=row_count)
    neighborhoods = rng.choice(["CollgCr", "NAmes", "OldTown", "Edwards", "Somerst"], size=row_count)
    house_styles = rng.choice(["1Story", "2Story", "SLvl"], size=row_count)

    neighborhood_bonus = pd.Series(neighborhoods).map(
        {"CollgCr": 30000, "NAmes": 18000, "OldTown": 8000, "Edwards": 12000, "Somerst": 35000}
    )
    house_style_bonus = pd.Series(house_styles).map({"1Story": 6000, "2Story": 12000, "SLvl": 9000})
    noise = rng.normal(0, 12000, size=row_count)

    sale_price = (
        45000
        + (lot_area * 2.4)
        + (overall_qual * 19000)
        + (overall_cond * 2500)
        + ((year_built - 1950) * 650)
        + ((year_remod - year_built) * 450)
        + (gr_liv_area * 95)
        + (full_bath * 5500)
        + (half_bath * 2200)
        + (bedrooms * 1800)
        + (total_rooms * 1400)
        + (fireplaces * 4200)
        + (garage_cars * 8500)
        + (garage_area * 18)
        + neighborhood_bonus.to_numpy()
        + house_style_bonus.to_numpy()
        + noise
    )

    df = pd.DataFrame(
        {
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "GrLivArea": gr_liv_area,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BedroomAbvGr": bedrooms,
            "TotRmsAbvGrd": total_rooms,
            "Fireplaces": fireplaces,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "Neighborhood": neighborhoods,
            "HouseStyle": house_styles,
            "SalePrice": sale_price.round(0).astype(int),
        }
    )
    return df


if __name__ == "__main__":
    output_path = Path("data/raw/housing.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_csv(output_path, index=False)
    print(f"Synthetic training dataset written to: {output_path}")
    print(f"Rows: {len(dataset)}")