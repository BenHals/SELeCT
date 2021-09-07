from collections import deque
from math import sqrt

import numpy as np
from entropy import perm_entropy
from PyEMD import EMD
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf, pacf

# Based on code from the 'rolling' python library at https://github.com/ajcr/rolling
# Adapted to one class here.

class RollingTimeseries:
    def __init__(self, window_size: int, ddof=1):
        self.window_size = window_size
        self.ddof = ddof
        self.timeseries = deque()
        self.np_timeseries = None
        self.cache = {}
        self._nobs: int = 0
        self._sum = 0.0
        self._v_mean = 0.0  # mean of values
        self._v_sslm = 0.0  # sum of squared values less the mean

        self._x1 = 0.0
        self._x2 = 0.0
        self._x3 = 0.0
        self._x4 = 0.0

    def get_np_timeseries(self):
        if self.np_timeseries is not None:
            return self.np_timeseries
        self.np_timeseries = np.array(self.timeseries)
        return self.np_timeseries
    
    def get_mean(self):
        if self._nobs < 1:
            return 0
        if 'mean' in self.cache:
            return self.cache['mean']
        mean = self._sum / self._nobs
        self.cache['mean'] = mean
        return mean

    def get_variance(self):
        if self._nobs < 2:
            return 0
        if 'var' in self.cache:
            return self.cache['var']
        try:
            var = self._v_sslm / (self._nobs - self.ddof)
        except Exception as e:
            print(self._v_sslm)
            print(self._nobs)
            print(self.ddof)
            raise e
        self.cache['var'] = var
        return var
    
    def get_stdev(self):
        if self._nobs < 2:
            return 0
        if 'stdev' in self.cache:
            return self.cache['stdev']
        try:
            # Due to instability can sometimes be a very small negative
            var = max(self.get_variance(), 0)
            stdev = sqrt(var)
        except Exception as e:
            print(self._v_sslm)
            print(self._nobs)
            print(self.ddof)
            print(self._v_sslm / (self._nobs - self.ddof))
            raise e
        self.cache['stdev'] = stdev
        return stdev
    
    def get_skew(self):
        if self._nobs < 2:
            return 0
        if 'skew' in self.cache:
            return self.cache['skew']
        N = self._nobs

        # compute moments
        A = self._x1 / N
        B = self._x2 / N - A * A
        C = self._x3 / N - A * A * A - 3 * A * B

        if B <= 1e-14:
            return 0.0

        R = sqrt(B)

        # If correcting for bias
        # skew = (sqrt(N * (N - 0)) * C) / ((N - 1) * R * R * R)
        # Otherwise
        skew = C / (R * R * R)
        self.cache['skew'] = skew
        return skew
    
    def get_kurtosis(self):
        if self._nobs < 2:
            # -3 is the kurtosis for a normal distribution,
            # (with fisher sdjustment), so we use as default
            return -3   
        if 'kurtosis' in self.cache:
            return self.cache['kurtosis']
        N = self._nobs

        # compute moments
        A = self._x1 / N
        R = A * A

        B = self._x2 / N - R
        R *= A

        C = self._x3 / N - R - 3 * A * B
        R *= A

        D = self._x4 / N - R - 6 * B * A * A - 4 * C * A

        if B <= 1e-14:
            return -3

        # If correcting for bias
        # K = (N * N - 1) * D / (B * B) - 3 * ((N - 1) ** 2)
        # kurtosis = K / ((N - 2) * (N - 3))
        # Otherwise
        K = D / (B * B)
        kurtosis = K
        fisher_style = True
        kurtosis = kurtosis - 3 if fisher_style else kurtosis
        self.cache['kurtosis'] = kurtosis
        return kurtosis
    
    def get_turning_point_rate(self):
        if self._nobs < 3:
            return 0
        if 'turning_point_rate' in self.cache:
            return self.cache['turning_point_rate']
        np_timeseries = self.get_np_timeseries()
        if np_timeseries.dtype == np.bool_:
            np_timeseries = np_timeseries.astype(np.int_)
        dx = np.diff(np_timeseries)
        turning_point_rate = np.sum(dx[1:] * dx[:-1] < 0)
        # turning_point_rate = len([*argrelmin(np_timeseries), *argrelmax(np_timeseries)])
        self.cache['turning_point_rate'] = turning_point_rate
        return turning_point_rate

    def get_acf(self):
        if self._nobs < 3:
            return [0, 0]
        if 'acf' in self.cache:
            return self.cache['acf']
        np_timeseries = self.get_np_timeseries()
        acf_vals = acf(np_timeseries, nlags=3, fft=True)
        parsed_vals = []
        for i, v in enumerate(acf_vals):
            if i == 0:
                continue
            if i > 2:
                break
            parsed_vals.append(v if not np.isnan(v) else 0)
        self.cache['acf'] = parsed_vals
        return parsed_vals


    def get_pacf(self):
        if self._nobs < 3:
            return [0, 0]
        if 'pacf' in self.cache:
            return self.cache['pacf']
        np_timeseries = self.get_np_timeseries()
        try:
            # Error is timeseries is constant
            acf_vals = pacf(np_timeseries, nlags=3)
        except Exception as e:
            acf_vals = [0 for x in range(6)]
        parsed_vals = []
        for i, v in enumerate(acf_vals):
            if i == 0:
                continue
            if i > 2:
                break
            parsed_vals.append(v if not np.isnan(v) else 0)
        self.cache['pacf'] = parsed_vals
        return parsed_vals

    def get_MI(self):
        if self._nobs < 3:
            return 0
        if 'mi' in self.cache:
            return self.cache['mi']
        np_timeseries = self.get_np_timeseries()
        if len(np_timeseries) > 4:
            current = np_timeseries
            previous = np.roll(current, -1)
            current = current[:-1]
            previous = previous[:-1]
            X = np.array(current).reshape(-1, 1)
            # Setting the random state is mostly for testing.
            # It can induce randomness in MI which is weird for paired
            # testing, getting different results with the same feature vec.
            MI = mutual_info_regression(
                X=X, y=previous, random_state=42, copy=False)[0]
        else:
            MI = 0
        self.cache['mi'] = MI
        return MI

    def get_IMF(self):
        if self._nobs < 3:
            return [0, 0]
        if 'imf' in self.cache:
            return self.cache['imf']
        emd = EMD(max_imf=2, spline_kind='slinear')
        np_timeseries = self.get_np_timeseries()
        IMFs = emd(np_timeseries, max_imf=2)
        entropies = [perm_entropy(imf) for imf in IMFs]
        self.cache['imf'] = entropies
        return entropies

    def update(self, new_val):
        last_np_timeseries = self.np_timeseries
        self.np_timeseries = None
        self.cache = {}
        self.timeseries.append(new_val)
    
        self._nobs += 1
        self._sum += new_val

        # update parameters for variance
        delta = new_val - self._v_mean
        self._v_mean += delta / self._nobs
        self._v_sslm += delta * (new_val - self._v_mean)

        # update parameters for moments
        sq = new_val * new_val
        self._x1 += new_val
        self._x2 += sq
        self._x3 += sq * new_val
        self._x4 += sq ** 2

        if len(self.timeseries) > self.window_size:
            self._remove_old(step_back=False)  
            if last_np_timeseries is not None:
                self.np_timeseries = last_np_timeseries
                self.np_timeseries[:-1] = self.np_timeseries[1:]
                self.np_timeseries[-1] = new_val
        

    def _remove_old(self, step_back=True):
        self.np_timeseries = None
        self.cache = {}
        old = self.timeseries.popleft()
        self._nobs -= 1
        self._sum -= old

        #Update parameters for variance
        delta = old - self._v_mean
        self._v_mean -= delta / self._nobs
        self._v_sslm -= delta * (old - self._v_mean)

        # update parameters for moments
        sq = old * old
        self._x1 -= old
        self._x2 -= sq
        self._x3 -= sq * old
        self._x4 -= sq ** 2
    
    def get_stats(self, FI=None, ignore_features=None):
        """ Calculates a set of statistics for current data.
        """
        stats = {}
        timeseries = self.get_np_timeseries()
        with np.errstate(divide='ignore', invalid='ignore'):
            if 'IMF' not in ignore_features:
                IMFs = self.get_IMF()
                for i, imf in enumerate(IMFs):
                    if f"IMF_{i}" not in ignore_features:
                        stats[f"IMF_{i}"] = imf
                for i in range(3):
                    if f"IMF_{i}" not in stats and f"IMF_{i}" not in ignore_features:
                        stats[f"IMF_{i}"] = 0
            if 'mean' not in ignore_features:
                stats["mean"] = self.get_mean()
            if 'stdev' not in ignore_features:
                stats["stdev"] = self.get_stdev()
            if 'skew' not in ignore_features:
                stats["skew"] = self.get_skew()
            if 'kurtosis' not in ignore_features:
                stats['kurtosis'] = self.get_kurtosis()
            if 'turning_point_rate' not in ignore_features:
                tp = int(self.get_turning_point_rate())
                if len(timeseries) > 0:
                    tp_rate = tp / len(timeseries)
                else:
                    tp_rate = 0
                stats['turning_point_rate'] = tp_rate

            if 'acf' not in ignore_features:
                acf_vals = self.get_acf()
                for i, v in enumerate(acf_vals):
                    if f"acf_{i+1}" not in ignore_features:
                        stats[f"acf_{i+1}"] = v
            if 'pacf' not in ignore_features:
                pacf_vals =self.get_pacf()
                for i, v in enumerate(pacf_vals):
                    if f"pacf_{i+1}" not in ignore_features:
                        stats[f"pacf_{i+1}"] = v

            if 'MI' not in ignore_features:
                MI = self.get_MI()
                stats["MI"] = MI

            if 'FI' not in ignore_features:
                stats["FI"] = FI if FI is not None else 0

        return stats



class RollingAutocorrelation:
    def __init__(self, nlags, window_size):
        self.nlags = nlags
        self.window_size = window_size
        self.lag_indexes = [i-1 for i in range(nlags+1)]
        self.values = [None for i in range(window_size)]
        self.means = [None for i in range(window_size)]
        self.variances = [None for i in range(window_size)]
        self.stdevs = [None for i in range(window_size)]
        self.sxy = [0 for i in range(nlags+1)]
        self.count = 0
    
    def update(self, value, mean, variance):
        # self.lag_indexes = [(i+1)%self.window_size for i in self.lag_indexes]
        new_index = self.lag_indexes[-1]
        for i, li in enumerate(self.lag_indexes):
            l = self.nlags-i
            lag_count = self.count-l
            if self.values[li] is None or lag_count + 1 <= 0:
                continue
            lag_ratio = lag_count / (lag_count+1)
            self.sxy[i] += (
                (self.means[li] - self.values[li])
                * (self.means[new_index] - value)
                * lag_ratio
            )

        self.lag_indexes = [(i+1)%self.window_size for i in self.lag_indexes]
        new_index = self.lag_indexes[-1]
        self.values[new_index] = value
        self.means[new_index] = mean
        self.variances[new_index] = variance
        self.stdevs[new_index] = None

        self.count += 1

    def _remove_old(self, value, mean, variance):
        self.count -= 1
        if self.count >= 1:
            self.lag_indexes = [(i+1)%self.window_size for i in self.lag_indexes]
            new_index = self.lag_indexes[-1]
            self.means[new_index] = mean
            self.variances[new_index] = variance
            self.stdevs[new_index] = None
            for i, li in enumerate(self.lag_indexes):
                l = self.nlags-i
                if self.values[li] is None or self.count <= l+1:
                    continue
                self.sxy[i] -= (
                    (self.means[li] - value)
                    * (self.means[new_index] - value)
                    * (self.count-l)
                    / (self.count-l+1)
                )
        else:
            self.__init__(self.nlags, self.window_size)

    def correlation(self, l):
        """Correlation of values (with `ddof` degrees of freedom)."""
        # l = (lag+1) * -1
        i = self.nlags-l
        sxy = self.sxy[i]
        x_index = self.lag_indexes[-1]
        y_index = self.lag_indexes[i]
        if self.stdevs[x_index] is None:
            x_stdev = sqrt(self.variances[x_index]) if self.variances[x_index] > 0 else 0
            self.stdevs[x_index] = x_stdev
        else:
            x_stdev = self.stdevs[x_index]
        if self.stdevs[y_index] is None:
            y_stdev = sqrt(self.variances[y_index]) if self.variances[y_index] > 0 else 0
            self.stdevs[y_index] = y_stdev
        else:
            y_stdev = self.stdevs[y_index]
        term = x_stdev * y_stdev
        ddof = 1
        return sxy / (((self.count-l)- ddof) * term) if term > 0 else 0

            


class RollingBasic:
    def __init__(self, ddof=1):
        self.ddof = ddof
        self.np_timeseries = None
        self._nobs: int = 0
        self._sum = 0.0
        self._v_mean = 0.0  # mean of values
        self._v_sslm = 0.0  # sum of squared values less the mean
        self.cache = {}
    
    def get_mean(self):
        if self._nobs < 1:
            return 0
        if 'mean' in self.cache:
            return self.cache['mean']
        mean = self._sum / self._nobs
        self.cache['mean'] = mean
        return mean

    def get_variance(self):
        if self._nobs < 2:
            return 0
        if 'var' in self.cache:
            return self.cache['var']
        var = self._v_sslm / (self._nobs - self.ddof)
        self.cache['var'] = var
        return var
    
    def get_stdev(self):
        if self._nobs < 2:
            return 0
        if 'stdev' in self.cache:
            return self.cache['stdev']
        # Due to instability can sometimes be a very small negative
        var = max(self.get_variance(), 0)
        stdev = sqrt(var)
        self.cache['stdev'] = stdev
        return stdev

    def update(self, new_val):
        self.cache = {}
    
        self._nobs += 1
        self._sum += new_val

        # update parameters for variance
        delta = new_val - self._v_mean
        self._v_mean += delta / self._nobs
        self._v_sslm += delta * (new_val - self._v_mean)

    def _remove_old(self, old):
        self.cache = {}
        self._nobs -= 1
        self._sum -= old
        if self._nobs >= 1:
            #Update parameters for variance
            delta = old - self._v_mean
            self._v_mean -= delta / self._nobs
            self._v_sslm -= delta * (old - self._v_mean)
        else:
            self._sum = 0.0
            self._v_mean = 0.0  # mean of values
            self._v_sslm = 0.0  # sum of squared values less the mean



class RollingRegression:
    """
    Compute simple linear regression in a single pass.

    Computes the slope, intercept, and correlation.
    Regression objects may also be added together and copied.

    Based entirely on the C++ code by John D Cook at
    http://www.johndcook.com/running_regression.html
    """

    def __init__(self,ddof=1):
        """Initialize Regression object.
        """
        self.ddof = ddof
        self._xstats = RollingBasic(ddof=ddof)
        self._ystats = RollingBasic(ddof=ddof)
        self._count = self._sxy = 0.0

    def __len__(self):
        """Number of values that have been pushed."""
        return int(self._count)

    def push(self, xcoord, ycoord):
        """Add a pair `(x, y)` to the Regression summary."""
        self._sxy += (
            (self._xstats.get_mean() - xcoord)
            * (self._ystats.get_mean() - ycoord)
            * self._count
            / (self._count + 1)
        )
        self._xstats.update(xcoord)
        self._ystats.update(ycoord)
        self._count += 1
    
    def remove_old(self, xcoord, ycoord):
        self._count -= 1
        self._xstats._remove_old(xcoord)
        self._ystats._remove_old(ycoord)

        self._sxy -= (
            (self._xstats.get_mean() - xcoord)
            * (self._ystats.get_mean() - ycoord)
            * self._count
            / (self._count + 1)
        )



    def correlation(self):
        """Correlation of values (with `ddof` degrees of freedom)."""
        term = self._xstats.get_stdev() * self._ystats.get_stdev()
        return self._sxy / ((self._count - self.ddof) * term) if term > 0 else 0


