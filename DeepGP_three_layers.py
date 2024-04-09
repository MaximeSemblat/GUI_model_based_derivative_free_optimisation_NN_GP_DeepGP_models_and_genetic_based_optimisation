import gpflux
import gpflow
import tensorflow as tf
import numpy as np 

class dgp_three_layers:
    def __init__(self, x):
        
        X=x
        # Layer 1
        Z = np.linspace(X.min(), X.max(), X.shape[0] // 2).reshape(-1, 2)
        kernel1 = gpflow.kernels.SquaredExponential()
        inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z.copy())
        gp_layer1 = gpflux.layers.GPLayer(
           kernel1, inducing_variable1, num_data=X.shape[0],num_latent_gps=2)

        # Layer 2
        kernel2 = gpflow.kernels.RationalQuadratic()
        inducing_variable2 = gpflow.inducing_variables.InducingPoints(Z.copy())
        gp_layer2 = gpflux.layers.GPLayer(
           kernel2,
           inducing_variable2,
           num_data=X.shape[0],
           num_latent_gps=2,
        )
        # Layer 3
        kernel3 = gpflow.kernels.RationalQuadratic()
        inducing_variable3 = gpflow.inducing_variables.InducingPoints(Z.copy())
        gp_layer3= gpflux.layers.GPLayer(
           kernel3,
           inducing_variable3,
           num_data=X.shape[0],
           num_latent_gps=12,
           mean_function=gpflow.mean_functions.Zero(),
        )

        # Initialise likelihood and build model
        likelihood_layer_2 = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
        self.three_layer_dgp = gpflux.models.DeepGP([gp_layer1, gp_layer2,gp_layer3], likelihood_layer_2)
        
    def fit(self, x, y):
        X= x
        Y= y
        # Compile and fit
        model = self.three_layer_dgp.as_training_model()
        model.compile(tf.optimizers.Adam(0.01))
        model.fit({"inputs": X, "targets": Y}, epochs=int(1e3), verbose=0)
        
        return 'model fit and ready for prediction'
        
    def mean_predict(self,x):
        self.prediction_model = self.three_layer_dgp.as_prediction_model()
        self.mean = self.prediction_model(x).y_mean.numpy()
        
        return (self.mean) 
