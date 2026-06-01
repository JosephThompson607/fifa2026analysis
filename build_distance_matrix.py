import pandas as pd
from openpyxl import load_workbook


def build_and_save(csv_path="flight_data_analysis/stadium_flight_distances.csv",
                   xlsx_path="data_fifa.xlsx",
                   sheet_name="stadium_distances"):

    df = pd.read_csv(csv_path)[["origin_stadium", "destination_stadium", "min_path_distance_km"]]

    # Pivot to square matrix
    matrix = df.pivot(index="origin_stadium", columns="destination_stadium", values="min_path_distance_km")

    # Add missing stadiums on both axes (self-distances) and fill diagonal with 0
    all_stadiums = sorted(set(df["origin_stadium"]) | set(df["destination_stadium"]))
    matrix = matrix.reindex(index=all_stadiums, columns=all_stadiums, fill_value=0)
    matrix = matrix.fillna(0)

    print(f"Distance matrix: {matrix.shape[0]} × {matrix.shape[1]} stadiums")
    print(matrix.iloc[:4, :4])

    # Write to a new sheet in data_fifa.xlsx (replace if exists)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        matrix.to_excel(writer, sheet_name=sheet_name)

    print(f"\nSaved to '{sheet_name}' tab in {xlsx_path}")
    return matrix


if __name__ == "__main__":
    build_and_save()
