import pandas as pd

train = pd.read_csv("data/raw/train.csv")

affected = train[
    (train["wing"] == "New Wing") &
    (train["avg_consult_min"] > 30)
].copy()

affected["converted_value"] = affected["avg_consult_min"] / 60

print("Affected rows:", len(affected))

print("\nOriginal values and values after dividing by 60:")
print(
    affected[["avg_consult_min", "converted_value"]]
    .head(20)
    .to_string(index=False)
)

print("\nConverted value range:")
print(affected["converted_value"].describe())

print("\nAre all converted values within 4 to 20?")
print(
    affected["converted_value"].between(4, 20).all()
)