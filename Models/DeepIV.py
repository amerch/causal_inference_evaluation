from .Model import Model
import keras
from econml.deepiv import DeepIVEstimator

class DeepIV(Model):
    def __init__(self):
        self.reg = None

    def fit(self, x, t, y, nfolds=5, seed=282):
        # splits = super().get_splits(x, nfolds, seed)
        treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(2,)),
                                   keras.layers.Dropout(0.17),
                                   keras.layers.Dense(64, activation='relu'),
                                   keras.layers.Dropout(0.17),
                                   keras.layers.Dense(32, activation='relu'),
                                   keras.layers.Dropout(0.17)])
        response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(2,)),
                                  keras.layers.Dropout(0.17),
                                  keras.layers.Dense(64, activation='relu'),
                                  keras.layers.Dropout(0.17),
                                  keras.layers.Dense(32, activation='relu'),
                                  keras.layers.Dropout(0.17),
                                  keras.layers.Dense(1)])
        self.reg = DeepIVEstimator(n_components=10, # Number of gaussians in the mixture density networks)
                      m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])), # Treatment model
                      h=lambda t, x: response_model(keras.layers.concatenate([t, x])), # Response model
                      n_samples=1 # Number of samples used to estimate the response
                      )

        self.reg.fit(y, t, x)

    def predict(self, x, t):
        if self.reg is None:
            raise Exception('DeepIV not Initialized')

        print("x", x.shape, x)
        print("t", t.shape, t)
        return self.reg.effect(T0, T1, X_test)

    def get_predictors(self, x, t):
        return np.hstack([(t - 0.5).reshape(-1, 1) * x, x])