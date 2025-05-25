import pdb
import optuna
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from functools import partial
from carry_over import AdstockGeometric, AdstockWeibull
from saturation import HillSaturation, ExponentialSaturation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class OptunaTuning:
    def __init__(self, X, y, delayed_channels, control_variables):
        self.X = X
        self.y = y
        self.delayed_channels = delayed_channels
        self.control_variables = control_variables

    def optuna_trial(self, trial, data, target, features, adstock_features, adstock_features_params, hill_slopes_params, hill_half_saturations_params, tscv):
        data_temp = data.copy()
        adstock_alphas = {}
        adstock_types = {}
        hill_slopes = {}
        hill_half_saturations = {}
        carryover_models = {}
        saturation_models = {}
        saturation_params = {}
        # coefficients = {}

        for feature in adstock_features:
            carryover_model_name = trial.suggest_categorical(f"carryover_model_{feature}", ["AdstockGeometric", "AdstockWeibull"])
            carryover_models[feature] = carryover_model_name

            saturation_model_name = trial.suggest_categorical(f"saturation_model_{feature}", ["HillSaturation"])
            saturation_models[feature] = saturation_model_name

            if carryover_model_name == 'AdstockGeometric':
                adstock_alpha = trial.suggest_float(f"theta_{feature}", 0.1, 0.8)
                adstock_alphas[feature] = adstock_alpha
            elif carryover_model_name == 'AdstockWeibull':
                adstock_shape = trial.suggest_float(f"shape_{feature}", 0.1, 1.0)
                adstock_scale = trial.suggest_float(f"scale_{feature}", 0.1, 1.0)
                adstock_type = trial.suggest_categorical(f"adstock_type_{feature}", ["cdf", "pdf"])
                adstock_alphas[feature] = (adstock_shape, adstock_scale)
                adstock_types[feature] = adstock_type
                
            if saturation_model_name == 'HillSaturation':
                hill_alpha = trial.suggest_float(f"hill_slope_{feature}", 0.5, 3.0)
                hill_gamma = trial.suggest_float(f"hill_half_saturation_{feature}", 0.3, 1.0)
                saturation_params[feature] = (hill_alpha, hill_gamma)
            elif saturation_model_name == 'ExponentialSaturation':
                exp_a = trial.suggest_float(f"a_{feature}", 0.01, 1.0)
                saturation_params[feature] = exp_a
            
            # hill_slope_param = f"{feature}_hill_slope"
            # min_, max_ = hill_slopes_params[hill_slope_param]
            # hill_slope = trial.suggest_float(f"hill_slope_{feature}", min_, max_)
            # hill_slopes[feature] = hill_slope

            # hill_half_saturation_param = f"{feature}_hill_half_saturation"
            # min_, max_ = hill_half_saturations_params[hill_half_saturation_param]
            # hill_half_saturation = trial.suggest_float(f"hill_half_saturation_{feature}", min_, max_)
            # hill_half_saturations[feature] = hill_half_saturation

            # Adstock transformation
            x_feature = data[feature].values.reshape(-1, 1)
            if carryover_model_name == 'AdstockGeometric':
                temp_adstock = AdstockGeometric(theta=adstock_alpha).fit_transform(x_feature)
            elif carryover_model_name == 'AdstockWeibull':
                shape, scale = adstock_alphas[feature]
                temp_adstock = AdstockWeibull(shape=shape, scale=scale, adstock_type=adstock_type).fit_transform(x_feature)

            # Hill saturation transformation
            if saturation_model_name == 'HillSaturation':
                temp_hill_saturation = HillSaturation(alpha=hill_alpha, gamma=hill_gamma).fit_transform(temp_adstock)
            elif saturation_model_name == 'ExponentialSaturation':
                temp_hill_saturation = ExponentialSaturation(a=exp_a).fit_transform(temp_adstock)

            data_temp[feature] = temp_hill_saturation

        regression_model_name = trial.suggest_categorical("regression_model", ["Ridge", "ElasticNet"])
        # Ridge parameters
        if regression_model_name == "Ridge":
            ridge_alpha = trial.suggest_float("alpha_ridge", 0.01, 10000)
            base_model = linear_model.Ridge(alpha=ridge_alpha, random_state=0,positive=True)
             # wrap in a small pipeline that scales your inputs
            regression_model = Pipeline([
                 ("scaler", StandardScaler()),
                 ("regression_model_name",   base_model)
            ])

            params = {"alpha_ridge": ridge_alpha}
        elif regression_model_name == "ElasticNet":
            alpha_enet = trial.suggest_float("alpha_enet", 0.01, 10000)
            l1_ratio_enet = trial.suggest_float("l1_ratio_enet", 0.0, 1.0)
            base_model = linear_model.ElasticNet(alpha=alpha_enet, l1_ratio=l1_ratio_enet, random_state=0,positive=True)
            regression_model = Pipeline([
                ("scaler", StandardScaler()),
                ("regression_model_name",base_model)
            ])
            params = {"alpha_enet": alpha_enet, "l1_ratio_enet": l1_ratio_enet}
        scores = []

        # Cross-validation
        for train_index, test_index in tscv.split(data_temp):
            x_train = data_temp.iloc[train_index][features]
            y_train = target[train_index]

            x_test = data_temp.iloc[test_index][features]
            y_test = target[test_index]

            regression_model.fit(x_train, y_train)
            prediction = regression_model.predict(x_test)

            rmse = root_mean_squared_error(y_true=y_test, y_pred=prediction)
            scores.append(rmse)
        
        # pdb.set_trace()
        print(regression_model.named_steps)
        for i, feature in enumerate(features):
            params[f"coef_{feature}"] = regression_model.named_steps["regression_model_name"].coef_[i]

        trial.set_user_attr("scores", scores)
        trial.set_user_attr("params", params)
        trial.set_user_attr("adstock_alphas", adstock_alphas)
        trial.set_user_attr("adstock_types", adstock_types)
        trial.set_user_attr("hill_slopes", hill_slopes)
        trial.set_user_attr("hill_half_saturations", hill_half_saturations)
        trial.set_user_attr("carryover_models", carryover_models)
        trial.set_user_attr("saturation_models", saturation_models)
        trial.set_user_attr("saturation_params", saturation_params)

        # Average of all scores
        return np.mean(scores)

    def optuna_optimize(self, trials, seed=42):
        print(f"Data size: {len(self.X)}")
        print(f"Adstock features: {self.delayed_channels}")
        print(f"Features: {self.delayed_channels + self.control_variables}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_mmm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))

        optimization_function = partial(self.optuna_trial,
                                       data=self.X,
                                       target=self.y.values.flatten(),  # Ensure target is a 1D array
                                       features=self.delayed_channels + self.control_variables,
                                       adstock_features=self.delayed_channels,
                                       adstock_features_params={
                                           f"{feature}_adstock": (0.1, 0.8) for feature in self.delayed_channels
                                       },
                                       hill_slopes_params={
                                           f"{feature}_hill_slope": (0.1, 5.0) for feature in self.delayed_channels
                                       },
                                       hill_half_saturations_params={
                                           f"{feature}_hill_half_saturation": (0.1, 1.0) for feature in self.delayed_channels
                                       },
                                       tscv=TimeSeriesSplit(n_splits=2, test_size=5))

        study_mmm.optimize(optimization_function, n_trials=trials, show_progress_bar=True)

        best_trial = study_mmm.best_trial

        # Extract the best carryover and saturation models
        carryover_models = {feature: best_trial.params[f"carryover_model_{feature}"] for feature in self.delayed_channels}
        saturation_models = {feature: best_trial.params[f"saturation_model_{feature}"] for feature in self.delayed_channels}
        saturation_params = {feature: best_trial.user_attrs["saturation_params"][feature] for feature in self.delayed_channels}

        best_trial_params = best_trial.user_attrs["params"]
        for key, value in best_trial_params.items():
            best_trial.params[key] = value
            
        return best_trial, carryover_models, saturation_models,saturation_params

    def model_refit(self, data, target, features, media_channels, organic_channels, model_params, adstock_params, adstock_types, hill_slopes_params, hill_half_saturations_params, carryover_models, saturation_models,saturation_params, start_index, end_index):
        data_refit = data.copy()

        best_params = model_params
        adstock_alphas = adstock_params

        # Apply adstock transformation
        temporal_features = [feature if feature not in media_channels and feature not in organic_channels else f"{feature}_hill" for feature in features]

        for feature in media_channels + organic_channels:
            adstock_alpha = adstock_alphas[feature]
            carryover_model_name = carryover_models[feature]
            saturation_model_name = saturation_models[feature]
            print(f"Applying {carryover_model_name} adstock transformation on {feature} with alpha {adstock_alpha:0.3}")

            # Adstock transformation
            x_feature = data_refit[feature].values.reshape(-1, 1)
            if carryover_model_name == 'AdstockGeometric':
                temp_adstock = AdstockGeometric(theta=adstock_alpha).fit_transform(x_feature)
            elif carryover_model_name == 'AdstockWeibull':
                shape, scale = adstock_alpha
                adstock_type = adstock_types[feature]  
                temp_adstock = AdstockWeibull(shape=shape, scale=scale, adstock_type=adstock_type).fit_transform(x_feature)

            hill_slope = hill_slopes_params[feature]
            hill_half_saturation = hill_half_saturations_params[feature]
            print(f"Applying {saturation_model_name} saturation transformation on {feature} with slope {hill_slope:0.3} and half saturation {hill_half_saturation:0.3}")

            # saturation transformation
            if saturation_model_name == 'HillSaturation':
                hill_slope, hill_half_saturation = saturation_params[feature]
                temp_hill_saturation = HillSaturation(alpha=hill_slope, gamma=hill_half_saturation).fit_transform(temp_adstock)
            elif saturation_model_name == 'ExponentialSaturation':
                a = saturation_params[feature]
                temp_hill_saturation = ExponentialSaturation(a=a).fit_transform(temp_adstock)

            data_refit[f"{feature}_adstock"] = temp_adstock
            data_refit[f"{feature}_hill"] = temp_hill_saturation

        # Build the final model on the data until the end analysis index
        x_input = data_refit.iloc[0:end_index][temporal_features].copy()
        y_true_all = target[0:end_index]

        # Build the regression model using the best parameters
        if best_params['regression_model'] == 'Ridge':
            model = linear_model.Ridge(random_state=0, alpha=best_params['alpha_ridge'])
        elif best_params['regression_model'] == 'ElasticNet':
            model = linear_model.ElasticNet(random_state=0, alpha=best_params['alpha_enet'], l1_ratio=best_params['l1_ratio_enet'])

        model.fit(x_input, y_true_all)

        # Concentrate on the analysis interval
        y_true_interval = y_true_all[start_index:end_index]
        x_input_interval_transformed = x_input.iloc[start_index:end_index]

        # Revenue prediction for the analysis interval
        print(f"Predicting {len(x_input_interval_transformed)} instances")
        prediction = model.predict(x_input_interval_transformed)

        # Non-transformed data set for the analysis interval
        x_input_interval_nontransformed = data.iloc[start_index:end_index]

        return {
            'x_input_interval_nontransformed': x_input_interval_nontransformed,
            'x_input_interval_transformed': x_input_interval_transformed,
            'prediction_interval': prediction,
            'y_true_interval': y_true_interval,
            "model": model,
            "model_train_data": x_input,
            "model_data": data_refit,
            "model_features": temporal_features,
            "features": features
        }