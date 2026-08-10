[README.md](https://github.com/user-attachments/files/30903330/README.md)
# KL House Price Predictor

This Streamlit application predicts residential house prices in Kuala Lumpur using a trained machine learning model.

## Model

Final model: Refined XGBoost selected through training-only grouped cross-validation.

The model was trained using cleaned Kuala Lumpur housing listing data and evaluated using a group-based train-test split to reduce feature-level overlap between training and testing data.

## Final Test Performance

- R2: 0.8934
- MAPE: 14.70%
- MAE: RM 286,596.10
- RMSE: RM 661,168.89

## Input Features

The application accepts the following user inputs:

- Location
- Property type
- Furnishing status
- Size in square feet
- Number of rooms
- Number of bathrooms
- Number of car parks
- Whether car park information is missing

## Files

- app.py: Streamlit application script
- requirements.txt: Python dependencies
- final_xgboost_deployment_pipeline.joblib: Final trained model pipeline
- deployment_metadata.joblib: Final deployment metadata and metrics

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

## Notes

The prediction range is based on the final model's MAPE. Actual house prices may vary due to unobserved factors such as renovation quality, building age, floor level, view, tenure, and micro-location.
