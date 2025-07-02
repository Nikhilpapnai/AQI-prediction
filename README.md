# Comparison of Machine Learning Models for AQI Prediction

## Objective

This project focuses on evaluating and comparing multiple machine learning models to determine the most accurate method for predicting the Air Quality Index (AQI) using environmental data.

## Models Implemented

The following regression algorithms were applied and tested:

- Multiple Linear Regression (MLR)
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)

All models were trained using Python libraries including NumPy, Pandas, and Scikit-learn, within the Spyder IDE (Anaconda distribution).

## Evaluation Metrics

To assess and compare model performance, the following error metrics were used:

- R² Score (Coefficient of Determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSLE (Root Mean Squared Logarithmic Error)

## Dataset Source

The dataset was sourced from the [Open Government Data (OGD) Platform India](https://data.gov.in/resources/real-time-air-quality-index-various-locations), which provides hourly AQI readings across multiple Indian cities. Parameters include concentrations of pollutants like:

- PM2.5
- PM10
- NO₂
- SO₂
- CO
- O₃

## Results

### Training Set Performance

| Model            | R²     | RMSE   | MAE    | RMSLE  |
|------------------|--------|--------|--------|--------|
| MLR              | 0.9965 | 5.9334 | 3.2952 | 0.0595 |
| Decision Tree    | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest    | 0.9996 | 2.0237 | 0.7106 | 0.0195 |
| SVR              | 0.9494 | 22.628 | 16.076 | 0.1423 |
| Polynomial Reg.  | 1.0000 | 0.0900 | 0.0180 | 0.0012 |

### Testing Set Performance

| Model            | R²     | RMSE     | MAE     | RMSLE  |
|------------------|--------|----------|---------|--------|
| MLR              | 0.9965 | 5.4973   | 3.4796  | 0.0517 |
| Decision Tree    | 0.9955 | 6.2370   | 2.3540  | 0.0563 |
| Random Forest    | 0.9982 | 3.8577   | 1.7016  | 0.0422 |
| SVR              | 0.9164 | 27.0025  | 19.0722 | 0.1686 |
| Polynomial Reg.  | -4.1417| 211.8759 | 81.5855 | 0.4638 |

## Sample Predictions

Predicted vs Actual AQI values for various models:

![MLR Prediction](imgs/mlr_p1.png)  
![Decision Tree Prediction](imgs/dtr_p1.png)  
![Random Forest Prediction](imgs/rfr_p1.png)  
![SVR Prediction](imgs/svr_p1.png)  
![Polynomial Regression Prediction](imgs/pr_p1.png)

## Conclusion

Among all models tested, the Random Forest Regressor achieved the highest accuracy and lowest error on both training and testing datasets. It proved to be the most reliable model for AQI forecasting in this study.

## AQI Standards Reference

Below is the AQI classification table as per CPCB guidelines:

![AQI Reference Table](https://w.ndtvimg.com/sites/3/2019/12/18122322/air_quality_index_standards_CPCB_650.jpg)
