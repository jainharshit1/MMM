# changes made: - target has been set to revenue, earlier target='revenue' now target=["revenue"]
# best_trial = optimizer.optuna_optimize(trials=10) this line was commented out and then the below line was written.best_trial, carryover_models, saturation_models, saturation_params = optimizer.optuna_optimize(trials=10)
# y->    .mul(y.values.flatten(), axis=0) as y is  now a dataframe and not a series
# a change was made at the saturationa and the carryove pipeline , the change was that instead of applying them over the entire delayed channels , they were applied oly on yhe media_channels

import pdb
from budgetoptimizer import BudgetAllocation
from budgetoptimisation import BudgetOptimization
# from historical_pattern_dist import BudgetDistribution
from helper_transformation_pipeline import get_transformers
# from allocator import model_based_optimization_solution, hist_pattern_allocation
from scipy import optimize
from data_collection import DataCollection
import numpy as np
from hyperparameter_tuning_optuna import OptunaTuning
from model import Model
import pandas as pd
# from target_scaling import AutoTargetScaler
import matplotlib.pyplot as plt

# Initialize data collection and prepare the data
data_obj = DataCollection()
data, prophet = data_obj.data_preparation()

transform_variables = ["trend", "season", "holiday", "competitor_sales_B",
                       "events", "tv_S", "ooh_S", "print_S",  "facebook_S", "search_S", "newsletter"]
media_channels = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
organic_channels = ['newsletter']
control_variables = ["trend", "season",
                     "holiday", "competitor_sales_B", "events"]
# control_variables = []
delayed_channels = media_channels + organic_channels
delay_control=delayed_channels+control_variables
target = ["revenue"]
X = data[delayed_channels + control_variables]
y = data[target]

# pdb.set_trace()
dates = pd.to_datetime(data['date'])

# Initialize Optuna tuning
optimizer = OptunaTuning(X, y, delayed_channels, control_variables)
# best_trial = optimizer.optuna_optimize(trials=10)
best_trial, carryover_models, saturation_models, saturation_params = optimizer.optuna_optimize(trials=100)
# Print the results of the best trial
# pdb.set_trace()
# print("Best trial RMSE:", np.sqrt(-best_trial.value))
# print("Best parameters:", best_trial.params)

# Initialize the model and build the final model
model_obj = Model()
final_model = model_obj.build_final_model(
    best_trial, delayed_channels, control_variables)
# Fit the final model
final_model.fit(X, y)

###### ACTUAL VS PREDICTION ######################################
y_pred = final_model.predict(X)
# print(f"RMSE: {np.sqrt(np.mean((y - y_pred)**2))}")
# print(f"MAPE: {np.mean(np.abs((y - y_pred) / y))}")
# print(f"NRMSE: {nrmse(y, y_pred)}")
# # Plot Predicted vs Actual according to the date column
plt.figure(figsize=(12, 6))
plt.plot(dates, y, label='Actual', color='blue', linestyle='-', marker='o')
plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Predicted vs Actual Revenue')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()  # Adjust layout to fit labels
plt.show()
##################################################################
# Extract and calculate contributions
adstock_data = pd.DataFrame(
    final_model.named_steps['adstock'].transform(X),
    columns=X.columns,
    index=X.index
)
print(adstock_data["tv_S"].min())
# pdb.set_trace()
weights = pd.Series(
    final_model.named_steps['regression'].regressor_.coef_,
    index=X.columns
)
base = final_model.named_steps['regression'].regressor_.intercept_

# Compute the unadjusted contributions
scaler = final_model.named_steps['regression'].transformer_
unadj_contributions = scaler.inverse_transform(
    adstock_data.mul(weights).assign(Base=base))
# print("Difference:", (np.array(unadj_contributions.sum(axis=1)).sum() - y_pred.sum()))

# unadj_contributions[unadj_contributions < 0] = 0
unadj_contributions_df = pd.DataFrame(unadj_contributions, columns=adstock_data.columns.tolist()+ ['Base'])
adj_contributions = (
    unadj_contributions_df
    .div(unadj_contributions_df.sum(axis=1), axis=0)
    .mul(y.values.flatten(), axis=0)  # Ensure y.values is flattened to match the shape
)
#########################################################
model = final_model.named_steps['regression']
optimization_period = 8
future_date = prophet.make_future_dataframe(
    periods=optimization_period, freq='W-SUN', include_history=False)
future_data = future_date.copy()
future_data['events_na'] = 1  # Assumed event for forecasting
future_data['events_event2'] = 0  # Assumed event for forecasting
forecast_df = prophet.predict(future_data)
prophet_columns = [col for col in forecast_df.columns if not col.endswith(
    "upper") and not col.endswith("lower")]
events_numeric = forecast_df[prophet_columns].filter(
    like="events_").sum(axis=1)
forecast_df = forecast_df[['ds', 'trend', 'yearly', 'holidays']]
forecast_df.rename(
    columns={'ds': 'date', 'yearly': 'season', 'holidays': 'holiday'}, inplace=True)
forecast_df["events"] = (events_numeric - np.min(events_numeric)).values
forecast_df["competitor_sales_B"] = data["competitor_sales_B"].mean()
forecast_df["newsletter"] = data["newsletter"].mean()
for col in media_channels:
    if col not in forecast_df.columns:
        forecast_df[col] = 10000

forecast_adstock = pd.DataFrame(
    final_model.named_steps['adstock'].transform(
        forecast_df[delayed_channels + control_variables]),
    columns=forecast_df[delayed_channels + control_variables].columns,
    index=forecast_df[delayed_channels + control_variables].index
)

additional_inputs = forecast_adstock[organic_channels + control_variables]

saturation_pipeline = get_transformers(
    best_trial=best_trial, delayed_channels=media_channels, transformer_model="saturation")
carry_over_pipeline = get_transformers(
    best_trial=best_trial, delayed_channels=media_channels, transformer_model="carryover")

carryover_data = pd.DataFrame(
    carry_over_pipeline.fit_transform(X),
    columns=X.columns,
    index=X.index
)
 
#media_channels = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
media_min_max_ranges = [(carryover_data[media_channel].min(), carryover_data[media_channel].max()) for media_channel in media_channels]
# pdb.set_trace()

saturated_data = pd.DataFrame(
    saturation_pipeline.fit_transform(carryover_data),
    columns=carryover_data.columns,
    index=carryover_data.index
)

# saturation_pipeline.fit(X)


model_features = X.columns
optimization_percentage = 0.2
budget_obj = BudgetOptimization(saturated_data, media_channels, model_features, prophet)
channel_spend_dist = budget_obj.historical_response(optimization_period)
# pdb.set_trace()
media_channel_average_spend = np.array([channel_spend_dist])

hist_response = budget_obj.hist_pattern_allocation(
    media_channel_average_spend,  # Baseline allocation
    model=final_model,  # Trained model
    # Additional inputs (e.g., seasonality, events)
    additional_inputs=additional_inputs,
    # saturation_pipeline=saturation_pipeline,  # Saturation pipeline
    media_min_max_ranges=media_min_max_ranges,
    delayed_channels=delay_control  # Min/max spend ranges
)

lower_bound = media_channel_average_spend * \
    np.ones(len(media_channels))*(1-optimization_percentage)
upper_bound = media_channel_average_spend * \
    np.ones(len(media_channels))*(1+optimization_percentage)
boundaries = optimize.Bounds(lb=lower_bound, ub=upper_bound)
# print(boundaries.lb.shape)
# print(boundaries.lb)
# print(media_channel_average_spend)
# print(boundaries.ub)
# print(f"total budget: {np.sum(media_channel_average_spend)}")


media_coefficients = [best_trial.params[f"coef_{channel}"] for channel in media_channels]
media_hill_slopes = [best_trial.params[f"hill_slope_{channel}"] for channel in media_channels]
media_hill_half_saturations = [best_trial.params[f"hill_half_saturation_{channel}"] for channel in media_channels]
# pdb.set_trace()

optimized_media_spend = budget_obj.optimize_budget(
    hist_response,  # Use adjusted allocation as baseline
    media_coefficients,  # Use media coefficients from best_trial
    media_hill_slopes,  # Use hill slopes from best_trial
    media_hill_half_saturations,  # Use half-saturation values from best_trial
    media_min_max_ranges,
    additional_inputs
)

# Create the result object
result = budget_obj.create_result(
    model=final_model,
    model_features=model_features,
    optimization_period=optimization_period,
    additional_inputs=additional_inputs,
    saturation_pipeline=saturation_pipeline,
    media_min_max_ranges=media_min_max_ranges,
    media_channels=media_channels,
    media_inputs=media_channel_average_spend,
    boundaries=boundaries,
    optimized_media_spend=optimized_media_spend
)

# print("Result Object:", result)
# pdb.set_trace()
#############
#########
budget_allocation = BudgetAllocation(data=data, media_channels=media_channels, model_features=delayed_channels +
                                     control_variables, best_trial=best_trial, model=final_model, prophet=prophet, adjusted_allocation=hist_response)
historical_spend_df = budget_allocation.historical_response()
solution = budget_allocation.optimized_response(solution=optimized_media_spend)
# pdb.set_trace()
########################################################################################
#########################################################################################################################################################
# Plotting the results
ax = (
    adj_contributions[adj_contributions.columns.tolist()[::-1]]
    .plot.area(
        figsize=(16, 10),
        linewidth=1,
        title='Predicted Sales and Breakdown',
        ylabel='Sales',
        xlabel='Date'
    )
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1], labels[::-1],
    title='Channels', loc="center left",
    bbox_to_anchor=(1.01, 0.5)
)

plt.show()

# Share of Spend vs Share of Effect
response_df = pd.DataFrame()
for media_channel in media_channels:
    # Calculate the total effect for each media channel
    # pdb.set_trace()
    response_total = adj_contributions[media_channel].sum()
    response_df = pd.concat([response_df, pd.DataFrame(
        {'media': [media_channel], 'total_effect': [response_total]})]).reset_index(drop=True)

response_df["effect_share"] = (
    response_df["total_effect"] / response_df["total_effect"].sum())*100

spend_df = pd.DataFrame()
for media_channel in media_channels:
    spends_total = X[media_channel].sum()
    spend_df = pd.concat([spend_df, pd.DataFrame(
        {'media': [media_channel], 'total_spend': [spends_total]})]).reset_index(drop=True)

spend_df["spend_share"] = (spend_df["total_spend"] /
                           spend_df["total_spend"].sum())*100
# pdb.set_trace()
# Plotting
fig, ax = plt.subplots(figsize=(14, 10))

# Set bar width and index positions
bar_width = 0.35
index = range(len(response_df))

# Plot horizontal bars for share of spend, offset by bar_width
bars1 = ax.barh([i + bar_width for i in index], spend_df['spend_share'],
                bar_width, label='Share of Spend', color='aqua', align='center')

# Plot horizontal bars for share of effect
bars2 = ax.barh(index, response_df['effect_share'], bar_width,
                label='Share of Effect', color='red', align='center')

# Labeling
ax.set_xlabel('Percentage (%)')
ax.set_ylabel('Media Channels')
ax.set_title('Share of Effect Vs Share of Spend')
ax.set_yticks([i + bar_width / 2 for i in index])
ax.set_yticklabels(response_df['media'])
ax.legend()

# Add percentage values on the bars
for bar in bars1:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%',
            va='center', ha='left', color='blue')

for bar in bars2:
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%',
            va='center', ha='left', color='red')

# Display the plot
plt.tight_layout()
plt.show()



# Plot Actual Spend vs Optimal Spend
actual_spend = X[media_channels].sum()
optimal_spend = pd.Series(optimized_media_spend, index=media_channels)

scaling_factor = actual_spend.sum() / optimal_spend.sum()
optimal_spend_scaled = optimal_spend * scaling_factor

print("Actual Spend:", actual_spend)
print("Optimal Spend:", optimal_spend)

fig, ax = plt.subplots(figsize=(14, 10))

bar_width = 0.35
index = range(len(media_channels))

bars1 = ax.bar(index, actual_spend, bar_width, label='Actual Spend', color='blue')
bars2 = ax.bar([i + bar_width for i in index], optimal_spend_scaled, bar_width, label='Optimal Spend', color='green')

ax.set_xlabel('Media Channels')
ax.set_ylabel('Spend')
ax.set_title('Actual Spend vs Optimal Spend')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(media_channels)

ax.legend()

# Add spend values on the bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', va='bottom', ha='center', color='blue')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', va='bottom', ha='center', color='green')

# plt.tight_layout()
# plt.show()