from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from carry_over import AdstockGeometric, AdstockWeibull
from saturation import HillSaturation, ExponentialSaturation

def get_transformers(best_trial, delayed_channels, transformer_model="saturation"):
    best_params = best_trial.params
    transformers = []
    
    for col in delayed_channels:
        if transformer_model == "carryover":
            carryover_model_name = best_params.get(f'carryover_model_{col}', 'AdstockGeometric')
            if carryover_model_name == 'AdstockGeometric':
                model = AdstockGeometric(theta=best_params.get(f'theta_{col}', 0.5))
            elif carryover_model_name == 'AdstockWeibull':
                model = AdstockWeibull(shape=best_params.get(f'shape_{col}', 1.0),
                                       scale=best_params.get(f'scale_{col}', 1.0),
                                       adstock_type=best_params.get(f'adstock_type_{col}', 'pdf'))
            # elif carryover_model_name == 'ExponentialCarryover':
            #     model = ExponentialCarryover(strength=best_params.get(f'strength_{col}', 0.5),
            #                                  length=best_params.get(f'length_{col}', 30))
        else:  # default to saturation
            saturation_model_name = best_params.get(f'saturation_model_{col}', 'HillSaturation')
            if saturation_model_name == 'HillSaturation':
                model = HillSaturation(alpha=best_params.get(f'alpha_{col}', 1.0),
                                       gamma=best_params.get(f'gamma_{col}', 0.5))
            elif saturation_model_name == 'ExponentialSaturation':
                model = ExponentialSaturation(a=best_params.get(f'a_{col}', 1.0))

        transformers.append((
            f'{col}_pipe',
            Pipeline([
                (transformer_model, model)
            ]),
            [col]
        ))
    
    # Build a ColumnTransformer that applies the chosen effect to each delayed channel
    return ColumnTransformer(transformers, remainder='passthrough')