import pdb
import numpy as np
import pandas as pd
from scipy import optimize
from functools import partial
from saturation import HillSaturation, ExponentialSaturation

class BudgetOptimization:
    def __init__(self,raw_data, data, media_channels, model_features, prophet, optimization_percentage=0.2):
        """
        Initialize the BudgetOptimization class.

        :param data: Historical data containing media spend and other variables.
        :param media_channels: List of media channels to optimize.
        :param model_features: List of features used in the model.
        :param prophet: Prophet model for forecasting.
        :param optimization_percentage: Percentage to vary the budget during optimization.
        """
        self.raw_data=raw_data
        self.data = data
        self.media_channels = media_channels
        self.model_features = model_features
        self.prophet = prophet
        self.optimization_percentage = optimization_percentage

    def historical_response(self, optimization_period):
        """
        Calculate the historical pattern allocation based on historical average spend.

        :param optimization_period: Number of periods to project the historical spend.
        :return: Historical pattern allocation for each media channel.
        """
        # Calculate the historical average spend for each media channel
        media_channel_average_spend = self.raw_data[self.media_channels].mean(axis=0).values
        # Scale the average spend by the optimization period
        historical_allocation = optimization_period * media_channel_average_spend
        return historical_allocation

    def hist_pattern_allocation(self, media_channel_average_spend, model, additional_inputs, media_min_max_ranges,delayed_channels):
        """
        Adjust the baseline allocation to account for adstock, additional inputs, and media constraints.

        :param media_channel_average_spend: Baseline media spend for each channel.
        :param model: Trained model to calculate contributions.
        :param additional_inputs: Additional inputs (e.g., seasonality, events).
        :param media_min_max_ranges: Min and max spend ranges for each channel.
        :return: Adjusted historical pattern allocation.
        """
        # Step 1: Adjust for adstock effects (if applicable)
        if hasattr(model, 'named_steps') and 'adstock' in model.named_steps:
            adstock_transform = model.named_steps['adstock']
            media_channel_average_spend = media_channel_average_spend.reshape(1, -1)
            zeroes = np.zeros((1, 6))
            media_channel_average_spend = np.concatenate((media_channel_average_spend, zeroes), axis=1)
            media_channel_average_spend_df = pd.DataFrame(media_channel_average_spend, columns=delayed_channels)
            media_channel_average_spend = adstock_transform.transform(media_channel_average_spend_df).flatten()[:5]
            # pdb.set_trace()    
        # Step 2: Incorporate additional inputs (e.g., seasonality, events)
        if additional_inputs is not None:
            media_channel_average_spend += additional_inputs.sum(axis=1)[:5]
        # pdb.set_trace()
        # Step 3: Respect media constraints (min/max ranges)
        # for i, (min_spend, max_spend) in enumerate(media_min_max_ranges):
        #     media_channel_average_spend[i] = np.clip(media_channel_average_spend[i], min_spend, max_spend)
        # pdb.set_trace()
        return media_channel_average_spend

    def budget_constraint(self, media_spend, budget):
        """
        Constraint function to ensure the total media spend equals the budget.

        :param media_spend: Array of media spends for each channel.
        :param budget: Total budget to be allocated.
        :return: Difference between total spend and budget.
        """
        return np.sum(media_spend) - budget

    def saturation_objective_function(self, coefficients, hill_slopes, hill_half_saturations, media_min_max_dictionary, media_inputs):
        """
        Objective function to maximize the response based on saturation curves.

        :param coefficients: Coefficients for each media channel.
        :param hill_slopes: Hill slopes for saturation curves.
        :param hill_half_saturations: Half-saturation points for saturation curves.
        :param media_min_max_dictionary: Min and max spend ranges for each channel.
        :param media_inputs: Media spend inputs for each channel.
        :return: Negative total response (to be minimized).
        """
        responses = []
        for i in range(len(coefficients)):
            coef = coefficients[i]
            hill_slope = hill_slopes[i]
            hill_half_saturation = hill_half_saturations[i]
            
            min_max = np.array(media_min_max_dictionary[i]).reshape(-1, 1)
            media_input = media_inputs[i]
            
            # Apply saturation transformation
            # hill_saturation = HillSaturation(alpha=hill_slope, gamma=hill_half_saturation).transform(X=min_max, x_point=media_input)
            hill_applied = HillSaturation(alpha=hill_slope, gamma=hill_half_saturation)
            hill_fit=hill_applied.fit(min_max)
            hill_saturation = hill_fit.transform(X=min_max, x_point=media_input)         
            response = coef * hill_saturation
            responses.append(response)
            
        responses = np.array(responses)
        responses_total = np.sum(responses)
        return -responses_total  # Negative for minimization

    def optimize_budget(self, media_channel_average_spend, media_coefficients, media_hill_slopes, media_hill_half_saturations, media_min_max_ranges, additional_inputs,delayed_channels,model,optimisation_period):
        """
        Optimize the media budget allocation.

        :param media_channel_average_spend: Baseline media spend for each channel.
        :param media_coefficients: Coefficients for each media channel.
        :param media_hill_slopes: Hill slopes for saturation curves.
        :param media_hill_half_saturations: Half-saturation points for saturation curves.
        :param media_min_max_ranges: Min and max spend ranges for each channel.
        :param additional_inputs: Additional inputs (e.g., seasonality, events).
        :return: Optimized media spend allocation.
        """
        # Calculate bounds for optimization
        lower_bound = media_channel_average_spend * (1 - self.optimization_percentage)
        upper_bound = media_channel_average_spend * (1 + self.optimization_percentage)
        # pdb.set_trace()
        # Ensure bounds respect media constraints
        transformed_media_min_max_ranges = []
        for i, (min_spend, max_spend) in enumerate(media_min_max_ranges):
            adstock_transform = model.named_steps['adstock']
            # Create a small dataframe with min and max values
            min_df = pd.DataFrame(np.zeros((1, len(delayed_channels))), columns=delayed_channels)
            max_df = pd.DataFrame(np.zeros((1, len(delayed_channels))), columns=delayed_channels)
            
            # Set the specific channel values
            min_df[delayed_channels[i]] = min_spend
            max_df[delayed_channels[i]] = max_spend
            
            # Transform with the same adstock/saturation pipeline
            transformed_min = adstock_transform.transform(min_df).flatten()[i]
            transformed_max = adstock_transform.transform(max_df).flatten()[i]
            
            transformed_media_min_max_ranges.append((transformed_min, transformed_max))
        for i, (min_spend, max_spend) in enumerate(transformed_media_min_max_ranges):
            lower_bound[i] = max(lower_bound[i], min_spend)
            upper_bound[i] = min(upper_bound[i], 2*optimisation_period*max_spend)

        boundaries = optimize.Bounds(lb=lower_bound, ub=upper_bound)
        print("Initial media_channel_average_spend:", media_channel_average_spend)
        print("Lower bounds:", lower_bound)
        print("Upper bounds:", upper_bound)
        # pdb.set_trace()
        # Partial function for the objective
        partial_saturation_objective_function = partial(
            self.saturation_objective_function,
            media_coefficients,
            media_hill_slopes,
            media_hill_half_saturations,
            media_min_max_ranges
        )

        # Optimization settings
        max_iterations = 100
        solver_func_tolerance = 1.0e-10

        # Perform optimization
        solution = optimize.minimize(
            fun=partial_saturation_objective_function,
            x0=media_channel_average_spend,
            bounds=boundaries,
            method="SLSQP",
            jac="3-point",
            options={
                "maxiter": max_iterations,
                "disp": True,
                "ftol": solver_func_tolerance,
            },
            constraints={
                "type": "eq",
                "fun": self.budget_constraint,
                "args": (np.sum(media_channel_average_spend), )
            }
        )
        optimal_spend = solution.x    
        # Calculate the raw actual spend (not transformed)
        actual_spend = self.raw_data[self.media_channels].mean(axis=0).values
        total_actual_spend = np.sum(actual_spend)        
        # Calculate optimal proportions and rescale
        optimal_proportions = optimal_spend / np.sum(optimal_spend)
        rescaled_optimal_spend = optimal_proportions * total_actual_spend
        
        print("Original Solution:", solution.x)
        print("Optimal proportions:", optimal_proportions)
        print("Rescaled optimal spend:", rescaled_optimal_spend)
        print("Total actual spend:", total_actual_spend)
        print("Total rescaled spend:", np.sum(rescaled_optimal_spend))
        # Return the optimized media spend    
        return rescaled_optimal_spend  # Return the optimized media spend
    
    def create_result(self, model, model_features, optimization_period, additional_inputs, saturation_pipeline, media_min_max_ranges, media_channels, media_inputs, boundaries, optimized_media_spend):
        """
        Create the result object in the same format as model_based_optimization_solution.

        :param model: Trained model.
        :param model_features: List of features used in the model.
        :param optimization_period: Number of periods to optimize.
        :param additional_inputs: Additional inputs (e.g., seasonality, events).
        :param saturation_pipeline: Pipeline for saturation effects.
        :param media_min_max_ranges: Min and max spend ranges for each channel.
        :param media_channels: List of media channels.
        :param media_inputs: Baseline media spend for each channel.
        :param boundaries: Bounds for optimization.
        :param optimized_media_spend: Optimized media spend allocation.
        :return: Result object containing all relevant information.
        """
        result = {
            "optimized_media_spend": optimized_media_spend,
            "boundaries": boundaries,
            "media_min_max_ranges": media_min_max_ranges,
            "model": model,
            "model_features": model_features,
            "optimization_period": optimization_period,
            "additional_inputs": additional_inputs,
            "saturation_pipeline": saturation_pipeline,
            "media_channels": media_channels,
            "media_inputs": media_inputs
        }
        return result
    