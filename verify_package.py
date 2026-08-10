from pathlib import Path

from predictor import PortableHousePriceModel

ROOT = Path(__file__).resolve().parent

model = PortableHousePriceModel(
    ROOT / "xgboost_booster.ubj",
    ROOT / "deployment_config.json",
)

record = {
    "built_up_sqft": 1200.0,
    "land_area_sqft": 0.0,
    "total_rooms": 3.0,
    "additional_rooms": 0.0,
    "is_studio": 0.0,
    "bathrooms": 2.0,
    "car_parks": 1.0,
    "location": "Mont Kiara, Kuala Lumpur",
    "size_type": "Built-up",
    "property_type_main": "Condominium",
    "property_subtype": "None",
    "furnishing": "Partly Furnished",
}

prediction = model.predict(record)
expected = 1056953.3350032477

assert abs(prediction - expected) < 1.0

print("PACKAGE SMOKE TEST: PASS")
print(f"Prediction: RM {prediction:,.2f}")
