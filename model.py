# commented out the use of the exponential caryover model as it was creating issues in the optuna tuning class file
#if saturation_model_name == 'HillSaturation':
#                saturation_model = HillSaturation(alpha=best_params[f'hill_slope_{col}'],
#                                                  gamma=best_params[f'hill_half_saturation_{col}'])
#in the above lines changes were made alpha->hil_slope and then gamma -> hill_half_saturation
import pdb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, ElasticNet
from carry_over import AdstockGeometric, AdstockWeibull, ExponentialCarryover
from saturation import HillSaturation, ExponentialSaturation
from sklearn.preprocessing import StandardScaler

# from target_scaling import AutoTargetScaler


class Model:
    def build_final_model(self, best_trial, delayed_channels, control_variables):
        # Extract best parameters
        best_params = best_trial.params
        regression_model_name = best_params['regression_model']

        # Create the adstock transformer with the best parameters
        transformers = []

        for col in delayed_channels:
            # pdb.set_trace()
            carryover_model_name = best_params[f'carryover_model_{col}']
            saturation_model_name = best_params[f'saturation_model_{col}']

            # Create carryover model with the best parameters for the channel
            if carryover_model_name == 'AdstockGeometric':
                carryover_model = AdstockGeometric(theta=best_params[f'theta_{col}'])
            elif carryover_model_name == 'AdstockWeibull':
                carryover_model = AdstockWeibull(shape=best_params[f'shape_{col}'],
                                                 scale=best_params[f'scale_{col}'],
                                                 adstock_type=best_params[f'adstock_type_{col}'])
            # elif carryover_model_name == 'ExponentialCarryover':
            #     carryover_model = ExponentialCarryover(strength=best_params[f'strength_{col}'],
            #                                            length=best_params[f'length_{col}'])

            # Create saturation model with the best parameters for the channel
            if saturation_model_name == 'HillSaturation':
                saturation_model = HillSaturation(alpha=best_params[f'hill_slope_{col}'],
                                                  gamma=best_params[f'hill_half_saturation_{col}'])
            elif saturation_model_name == 'ExponentialSaturation':
                saturation_model = ExponentialSaturation(a=best_params[f'a_{col}'])

            transformers.append((
                f'{col}_pipe',
                Pipeline([
                    ('carryover', carryover_model),
                    ('saturation', saturation_model),
                    # ('scaler', MinMaxScaler())  # Optional: Normalizing the data
                ]),
                [col]
            ))
        for col in control_variables:
            transformers.append((
                f'{col}_pipe',
                Pipeline([
                    ('scaler', MinMaxScaler())
                ]),
                [col]
            ))

        adstock = ColumnTransformer(
            transformers,
            remainder='passthrough'
        )

        # Create the regression model with the best parameters
        if regression_model_name == 'Ridge':
            regression_model = Ridge(alpha=best_params.get('alpha_ridge', 1.0),
                                     positive=best_params.get('positive_ridge', False))
        elif regression_model_name == 'ElasticNet':
            regression_model = ElasticNet(alpha=best_params.get('alpha_enet', 1.0),
                                          l1_ratio=best_params.get('l1_ratio_enet', 0.5),
                                          positive=best_params.get('positive_enet', False))

        # Construct the final pipeline
        final_model = Pipeline([
            ('adstock', adstock),
            # ('regression', TransformedTargetRegressor(regressor=regression_model, transformer=AutoTargetScaler()))
            ('regression', TransformedTargetRegressor(regressor=regression_model, transformer=StandardScaler()))

        ])

        return final_model
