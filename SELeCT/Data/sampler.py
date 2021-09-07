import numpy as np
import math
from skmultiflow.utils import check_random_state

from scipy.stats import norm
from scipy.special import expit

def sinh_archsinh_transformation(x,epsilon,delta):
    """ The Sinh archhsinh distribution lets us set more properties of the distribution
    compared to a normal distribution.
    """
    return norm.pdf(np.sinh(delta*np.arcsinh(x)-epsilon))*delta*np.cosh(delta*np.arcsinh(x)-epsilon)/np.sqrt(1+np.power(x,2))

def mean_stdev_skew_kurt_sample(rand_state, mean, stdev, skew, kurtosis):
    Z = rand_state.normal(loc = 0, scale = 1)
    X = np.sinh((1/kurtosis)*(np.arcsinh(Z) + skew))
    return ((mean / stdev) + X) * stdev


class Sampler():
    def __init__(self, num_features, random_state = None, features = ['distribution', 'autocorrelation', 'frequency'], strength = 1):
        """
        Features is the type of distribution change we want to introduce
        Strength is a multiplier for the adjusted distribution compared to a uniform random sample.
        """
        self.num_features = num_features
        self.random_state = check_random_state(random_state)
        self.features = features
        self.strength = strength
        self.feature_data = []
        for i in range(self.num_features):
            data = {}
            data['clock'] = 0
            data['mean'] = self.random_state.rand()
            data['stdev'] = self.random_state.rand()
            # data['skew'] = self.random_state.randint(-5, 5)
            data['skew'] = -1 + self.random_state.rand() * 2

            # data['kurtosis'] = self.random_state.randint(1, 10)
            data['kurtosis'] = 0.75 + self.random_state.rand() * 1
            data['pacf_0'] = self.random_state.rand() * 0.4
            data['pacf_1'] = self.random_state.rand() * 0.4
            data['pacf_2'] = self.random_state.rand() * 0.4
            data['f_0'] = self.random_state.rand()
            data['a_0'] = self.random_state.rand()*2
            data['f_1'] = self.random_state.rand()
            data['a_1'] = self.random_state.rand()*2
            data['f_2'] = self.random_state.rand()
            data['a_2'] = self.random_state.rand()*2
            self.feature_data.append(data)
            data['l_0'] = self.get_base_sample(i)
            data['l_1'] = self.get_base_sample(i)
            data['l_2'] = self.get_base_sample(i)


    
    def get_base_sample(self, i):
        if 'distribution' in self.features:
            mean = self.feature_data[i]['mean']
            stdev = self.feature_data[i]['stdev']
            skew = self.feature_data[i]['skew']
            kurtosis = self.feature_data[i]['kurtosis']
            return mean_stdev_skew_kurt_sample(self.random_state, mean, stdev, skew, kurtosis)
        else:
            return self.random_state.rand()
    
    def get_frequency(self, i):
        self.feature_data[i]['clock'] += 1
        wave_0 = self.feature_data[i]['a_0'] * np.sin((self.feature_data[i]['clock'] * self.feature_data[i]['f_0'])/(2*math.pi))
        wave_1 = self.feature_data[i]['a_1'] * np.sin((self.feature_data[i]['clock'] * self.feature_data[i]['f_1'])/(2*math.pi))
        wave_2 = self.feature_data[i]['a_2'] * np.sin((self.feature_data[i]['clock'] * self.feature_data[i]['f_2'])/(2*math.pi))
        return wave_0 + wave_1 + wave_2
    def get_sample(self, i):
        base = self.get_base_sample(i)
        auto_corr = self.feature_data[i]['l_0'] * self.feature_data[i]['pacf_0'] + \
            self.feature_data[i]['l_1'] * self.feature_data[i]['pacf_1'] + \
            self.feature_data[i]['l_2'] * self.feature_data[i]['pacf_2']
        self.feature_data[i]['l_2'] = self.feature_data[i]['l_1']
        self.feature_data[i]['l_1'] = self.feature_data[i]['l_0']
        self.feature_data[i]['l_0'] = base
        freq = self.get_frequency(i)
        sample = 0
        if 'distribution' in self.features:
            sample += base
        if 'autocorrelation' in self.features:
            sample += auto_corr
        if 'frequency' in self.features:
            sample += freq * 0.1
        # sample = base + auto_corr + freq * 0.1
        return (self.strength) * expit(sample) + (1-self.strength) * self.random_state.rand()