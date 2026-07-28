# Gold-Price-Indicator
Gold Price Predictors utilizing python and 5 years of historical data to accurately predict prices.
Language: Python
Data Extraction: 'yfinance' API
Data Manipulation: 'pandas', 'numpy'
Machine Learning: 'scikit-learn' using random forest regressor
visualization: 'plotlib'
Overall predicting prices can lead to overfitting in machine learning models and to combat this and have a more robust engine This project uses targeting price adjustments, rolling technical indicators, and preventing data leakage
Targeting Price adjustments: The model predicts next-day change. Which is then reconstructed post-prediction which prevents the model echoing the previous day's close.
Rolling technical indicator: Engineered through "MA_10" technical indicators and "MA_50" both moving averages to capture broader market trends and provide it historical context. 
Preventing Data Leakage: This model has strict chronological train-test splitting. in time-series forecasting, shuffling training data allows the future events to leak into the training set, ruining the mode's real-world validity.
To evaluate the model the Mean Absolute Error and Root mean squared error is used to evaluate performance.
To instal dependicies 
pip install yfinance pandas numpy scikit-learn matplotlib
To execute script
python model.py
