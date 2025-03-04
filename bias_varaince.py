import numpy as np
import matplotlib.pyplot as plt


class biasVariance:
    def __init__(self, no_of_dataset, no_training_samples, no_test_samples, lambda_list, s=0.1, M=25, seed=42):
        self.no_of_dataset = no_of_dataset
        self.no_training_samples = no_training_samples
        self.no_test_samples = no_test_samples
        self.seed = seed
        self.s = s
        self.M = M
        self.lambda_list = lambda_list
        np.random.seed(self.seed)
        self.datasets = []

        self.centers = np.linspace(0, 1, self.M)
        self.x_test = np.linspace(0, 1, self.no_test_samples)
        self.h_test = np.sin(2 * np.pi * self.x_test)
        self.t_test = np.sin(2 * np.pi * self.x_test) + np.random.normal(0, 0.3, self.no_test_samples)

        self.generate_datasets()
        self.calculate_weights()
        self.calculate_predictions()
        self.calculate_bias_variance()
        self.plot_results()

    def generate_datasets(self):
        for _ in range(self.no_of_dataset):
            X = np.random.uniform(0, 1, self.no_training_samples)
            t = np.sin(2 * np.pi * X) + np.random.normal(0, 0.3, self.no_training_samples)  
            self.datasets.append((X, t))

    def non_linear_transformation(self, X):
        Phi = np.zeros((len(X), self.M))
        for i in range(len(X)):
            Phi[i] = np.exp(-((X[i] - self.centers) ** 2) / (2 * self.s ** 2))
        return Phi

    def calculate_weights(self):
        self.weights = {}
        for lam in self.lambda_list:
            weights = []
            for X, t in self.datasets:
                Phi = self.non_linear_transformation(X)
                regularizer = lam * np.identity(self.M)
                w = np.linalg.pinv(Phi.T @ Phi + regularizer) @ Phi.T @ t
                weights.append(w)
            self.weights[lam] = np.array(weights)

    def calculate_predictions(self):
        self.predictions = {}
        Phi_test = self.non_linear_transformation(self.x_test)
        for lam in self.weights:
            predictions = []
            for w in self.weights[lam]:
                pred = Phi_test @ w
                predictions.append(pred)
            self.predictions[lam] = np.array(predictions)
        #print(self.predictions)
    def calculate_bias_variance(self):
        self.bias = {}
        self.variance = {}
        self.test_error = {}
        for lam in self.predictions:
            preds = self.predictions[lam]
            mean_pred = np.mean(preds, axis=0)
            bias_squared = np.mean((mean_pred - self.h_test) ** 2)
            variance = np.mean(np.var(preds, axis=0))
            test_error = np.mean((preds - self.t_test[None, :]) ** 2)
            self.bias[lam] = bias_squared
            self.variance[lam] = variance
            self.test_error[lam] = test_error

    def plot_results(self):
        lamdas = list(self.bias.keys())
        log_lamdas = np.log(lamdas)
        bias_val = [self.bias[lam] for lam in lamdas]
        variance_val = [self.variance[lam] for lam in lamdas]
        test_error_val = [self.test_error[lam] for lam in lamdas]
        total = np.array(bias_val) + np.array(variance_val)
        # print(total)

        plt.figure(figsize=(10, 8))
        plt.plot(log_lamdas, bias_val, label=r'$(bias)^2$', color='blue', linewidth=2)
        plt.plot(log_lamdas, variance_val, label='variance', color='red', linewidth=2)
        plt.plot(log_lamdas, total, label=r'$(bias)^2 + variance$', color='magenta', linewidth=2)
        plt.plot(log_lamdas, test_error_val, label='test error', color='black', linewidth=2)

        # plt.xscale('log')
        plt.xlabel(r'$\ln(\lambda)$')
        plt.legend()
        plt.grid()
        plt.show()




if __name__ == "__main__":

    lambda_list =np.linspace(0.1, 15, 100)
    bias_variance_instance = biasVariance(100, 25, 1000, lambda_list)
           