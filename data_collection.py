#the changes made in this file are as follows:-the ds column in the holidays have been converted into proper format. 
#in the end the final_data=read_csv..... this has been changed for the final_data.to_write_csv . 

import pandas as pd
from prophet import Prophet
import numpy as np
import os
# Add this function near the top of your file
def get_data_path(relative_path):
    """Get absolute path to data file, works in both local and cloud environments"""
    # Get the directory where the current file is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the path using os.path.join which handles platform-specific separators
    return os.path.join(base_dir, relative_path)

# Then whenever loading the data file, use:
# Instead of: 'data\robyn_sample.csv'
# Use: get_data_path('data/robyn_sample.csv')  # Note the forward slashes
class DataCollection:
    def data_preparation(self):
        data = pd.read_csv(get_data_path('data/robyn_sample.csv'),parse_dates = ["DATE"])
        data.columns = [c.lower() if c in ["DATE"] else c for c in data.columns]
        holidays = pd.read_csv(get_data_path('data/robyn_holidays.csv'),parse_dates = ["ds"])
        holidays["ds"] = pd.to_datetime(holidays["ds"], format="%d-%m-%Y")
        holidays["begin_week"] = holidays["ds"].dt.to_period('W').dt.start_time
        #combine same week holidays into one holiday
        holidays_weekly = holidays.groupby(["begin_week", "country","year"], as_index = False).agg({'holiday':'#'.join, 'country':'first', 'year': 'first'}).rename(columns = {'begin_week': 'ds'})
        holidays_weekly_de = holidays_weekly.query("(country =='DE')").copy()

        # Prophet Decomposition
        prophet_data = data.rename(columns = {'revenue': 'y', 'date': 'ds'})
        #add categorical into prophet
        prophet_data = pd.concat([prophet_data,pd.get_dummies(prophet_data["events"], drop_first = True, prefix ="events")], axis = 1)
        prophet = Prophet(yearly_seasonality=True,holidays=holidays_weekly_de)
        prophet.add_regressor(name = "events_event2")
        prophet.add_regressor(name = "events_na")
        prophet.fit(prophet_data[["ds", "y", "events_event2", "events_na"]])
        prophet_predict = prophet.predict(prophet_data[["ds", "y","events_event2", "events_na"]])
        #######################
        #Let’s extract the seasonality, trend, holidays, and events components:
        prophet_columns = [col for col in prophet_predict.columns if (col.endswith("upper") == False) & (col.endswith("lower") == False)]
        events_numeric = prophet_predict[prophet_columns].filter(like = "events_").sum(axis = 1)
        final_data = data.copy()
        final_data["trend"] = prophet_predict["trend"]
        final_data["season"] = prophet_predict["yearly"]
        final_data["holiday"] = prophet_predict["holidays"]
        final_data["events"] = (events_numeric - np.min(events_numeric)).values
        # final_data = pd.read_csv("data/data.csv", parse_dates = ["date"])
        final_data.to_csv(get_data_path('data/data.csv'), index=False)

        return final_data, prophet