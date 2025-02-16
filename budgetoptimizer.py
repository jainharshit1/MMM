import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from carry_over import AdstockGeometric, AdstockWeibull
from saturation import HillSaturation, ExponentialSaturation # Replace with the actual module name

class BudgetAllocation:
    def __init__(self, data, media_channels, model_features, best_trial, model, prophet, adjusted_allocation):
        """
        Initialize the BudgetAllocation class.

        :param data: Historical data containing media spend and other variables.
        :param media_channels: List of media channels to optimize.
        :param model_features: List of features used in the model.
        :param best_trial: Best trial from Optuna tuning.
        :param model: Trained model.
        :param prophet: Prophet model for forecasting.
        :param adjusted_allocation: Adjusted historical allocation from hist_pattern_allocation.
        """
        self.data = data
        self.media_channels = media_channels
        self.model_features = model_features
        self.best_trial = best_trial
        self.model = model
        self.prophet = prophet
        self.adjusted_allocation = adjusted_allocation

    def historical_response(self):
        """
        Calculate the historical spend and response for each media channel using the adjusted allocation.

        :return: DataFrame containing historical spend and response for each channel.
        """
        historical_spend_df = pd.DataFrame()

        for i, media_channel in enumerate(self.media_channels):
            # Extract adstock, hill slope, and half-saturation values from best_trial
            carryover_model_name = self.best_trial.params[f"carryover_model_{media_channel}"]
            saturation_model_name = self.best_trial.params[f"saturation_model_{media_channel}"]
            coef = self.best_trial.params[f"coef_{media_channel}"]

            # Select and apply the adstock model
            if carryover_model_name == 'AdstockGeometric':
                theta = self.best_trial.params[f"theta_{media_channel}"]
                adstock_model = AdstockGeometric(theta=theta)
            elif carryover_model_name == 'AdstockWeibull':
                shape = self.best_trial.params[f"shape_{media_channel}"]
                scale = self.best_trial.params[f"scale_{media_channel}"]
                adstock_type = self.best_trial.params[f"adstock_type_{media_channel}"]
                adstock_model = AdstockWeibull(shape=shape, scale=scale, adstock_type=adstock_type)

            # Use the adjusted allocation for spend
            # pdb.set_trace()
            adjusted_allocation = np.atleast_2d(self.adjusted_allocation[i])

            spendings_adstocked = adstock_model.fit_transform(adjusted_allocation)

                    # Select and apply the saturation model
            if saturation_model_name == 'HillSaturation':
                hill_slope = self.best_trial.params[f"hill_slope_{media_channel}"]
                hill_half_saturation = self.best_trial.params[f"hill_half_saturation_{media_channel}"]
                saturation_model = HillSaturation(alpha=hill_slope, gamma=hill_half_saturation)
            elif saturation_model_name == 'ExponentialSaturation':
                exp_a = self.best_trial.params[f"a_{media_channel}"]
                saturation_model = ExponentialSaturation(a=exp_a)

            # Calculate saturated response
            spendings_saturated = saturation_model.fit_transform(spendings_adstocked)
            response = spendings_saturated * coef

            # Store results in DataFrame
            temp_df = pd.DataFrame({
                'media_channel': [media_channel] * len(spendings_adstocked),
                'spend': spendings_adstocked.flatten(),
                'response': response.flatten()
            })
            historical_spend_df = pd.concat([historical_spend_df, temp_df], ignore_index=True)

        return historical_spend_df
    def optimized_response(self, solution):
        """
        Calculate the optimized spend and response for each media channel.

        :param solution: Optimized spend values from the optimization process.
        :return: DataFrame containing optimized spend and response for each channel.
        """
        optimized_response_df = pd.DataFrame()

        for i, media_channel in enumerate(self.media_channels):
            # Extract adstock, hill slope, and half-saturation values from best_trial
            carryover_model_name = self.best_trial.params[f"carryover_model_{media_channel}"]
            saturation_model_name = self.best_trial.params[f"saturation_model_{media_channel}"]
            coef = self.best_trial.params[f"coef_{media_channel}"]

            # Select and apply the adstock model
            if carryover_model_name == 'AdstockGeometric':
                theta = self.best_trial.params[f"theta_{media_channel}"]
                adstock_model = AdstockGeometric(theta=theta)
            elif carryover_model_name == 'AdstockWeibull':
                shape = self.best_trial.params[f"shape_{media_channel}"]
                scale = self.best_trial.params[f"scale_{media_channel}"]
                adstock_type = self.best_trial.params[f"adstock_type_{media_channel}"]
                adstock_model = AdstockWeibull(shape=shape, scale=scale, adstock_type=adstock_type)

            # Use the adjusted allocation for spend
            adjusted_allocation = np.atleast_2d(self.adjusted_allocation[i])

            # Apply the adstock model
            spendings_adstocked = adstock_model.fit_transform(adjusted_allocation)

            # Select and apply the saturation model
            if saturation_model_name == 'HillSaturation':
                hill_slope = self.best_trial.params[f"hill_slope_{media_channel}"]
                hill_half_saturation = self.best_trial.params[f"hill_half_saturation_{media_channel}"]
                saturation_model = HillSaturation(alpha=hill_slope, gamma=hill_half_saturation)
            elif saturation_model_name == 'ExponentialSaturation':
                exp_a = self.best_trial.params[f"a_{media_channel}"]
                saturation_model = ExponentialSaturation(a=exp_a)


            saturation_model.fit(spendings_adstocked)
            # Calculate optimized response
            optimized_spend = solution[i]





            #             hill_applied = HillSaturation(alpha=hill_slope, gamma=hill_half_saturation)
            # hill_fit=hill_applied.fit(min_max)
            # hill_saturation = hill_fit.transform(X=min_max, x_point=media_input)
            # 
            #  
            optimized_response = coef * saturation_model.transform(X=spendings_adstocked, x_point=optimized_spend)
            # pdb.set_trace()
            # Store results in DataFrame
            temp_df = pd.DataFrame({
                'media_channel': [media_channel],
                'optimized_spend': [optimized_spend],
                'optimized_response': [optimized_response]
            })
            optimized_response_df = pd.concat([optimized_response_df, temp_df], ignore_index=True)

        return optimized_response_df

    def plot_spend_response_curve(self, media_channel, spend_response_df, average_spend, average_response, optimized_spend, optimized_response):
        """
        Plot the spend-response curve for a media channel.

        :param media_channel: Name of the media channel.
        :param spend_response_df: DataFrame containing spend and response data.
        :param average_spend: Average spend for the channel.
        :param average_response: Average response for the channel.
        :param optimized_spend: Optimized spend for the channel.
        :param optimized_response: Optimized response for the channel.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(spend_response_df['spend'], spend_response_df['response'], label='Spend-Response Curve')
        plt.scatter(average_spend, average_response, color='red', label='Average Spend/Response')
        plt.scatter(optimized_spend, optimized_response, color='green', label='Optimized Spend/Response')
        plt.title(f'Spend-Response Curve for {media_channel}')
        plt.xlabel('Spend')
        plt.ylabel('Response')
        plt.legend()
        plt.grid(True)
        plt.show()