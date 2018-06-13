import numpy as np
from scipy.interpolate import bisplev
from scipy.stats import f
from scipy import interpolate
from numpy.fft import rfft, irfft
from numpy.random import uniform
from scipy.interpolate import LSQUnivariateSpline
from collections import defaultdict


class FeatureGenerator(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return None
class FourierFG(FeatureGenerator):
    def __init__(self, threshold=10, freq=True):
        super().__init__()
        self.threshold = threshold
        self.freq = freq

    def findBestComponents(self, data, class_labels, topCompNumber=None):
        if topCompNumber is None:
            topCompNumber = min(4,self.threshold)

        data_ts_classes = defaultdict(list)
        for i, label in enumerate(class_labels):
            data_ts_classes[label].append(data[i])

        count_top = dict([(j, dict([(i, 0) for i in range(0, self.threshold)]))
                          for j in data_ts_classes])
        for class_num in data_ts_classes:
            for ts in data_ts_classes[class_num]:
                result = [x for x in enumerate(self.__call__(ts))]
                result = sorted(result, key=lambda x: x[1], reverse=True)
                for i in range(0, topCompNumber):
                    count_top[class_num][result[i][0]] += 1

        tops = set([])
        for class_num in count_top:
            result = [(key, count_top[class_num][key])
                      for key in count_top[class_num]]
            result = sorted(result, key=lambda x: x[1], reverse=True)
            for i in range(0, topCompNumber):
                tops.add(result[i][0])

        return list(tops)

    def __call__(self, data, use_components=None):
        if use_components is None:
            use_components = list(range(0, self.threshold))
        y = data
        y_0 = np.fft.rfft(y)
        if self.freq:
            y_0[self.threshold:] = 0
        else:
            y_0 = [j * self.FH(abs(j)-self.threshold) for j in y_0]

        if self.freq and len(y_0) < self.threshold:
            y_0 = np.concatenate(y_0, np.zeros(self.threshold))

        result = y_0[:self.threshold]

        result = np.sqrt(np.power(result.imag, 2) + np.power(result.real, 2))

        return np.take(result / len(data), use_components)

    def FH(x):
        """
        Функция для чистки
        """
        if x>=0:
            q=1
        else:
            q=0
        return q



import pandas as pd
class FisherSplineFG(FeatureGenerator):
    def __init__(self, coeff_limit=20, spline_degree=3, p_value=0.4):
        super().__init__()
        self.coeff_limit = coeff_limit
        self.spline_degree = spline_degree
        self.p_value = p_value

    def __call__(self, data):
        spl = self.fisher_plot(data)
        self.spl = spl
        coeffs = np.concatenate((spl.get_coeffs(), np.zeros(self.coeff_limit)))
        return coeffs[:self.coeff_limit]

    def FF(self, df1, df2):
        """
        Функция принимает на вход две выборки и
        возвращает p-value критерия Фишера
        """
        v1 = np.var(df1)
        v2 = np.var(df2)

        if v1 < v2:
            F = np.var(df1) / np.var(df2)
        else:
            F = np.var(df2) / np.var(df1)

        return f.cdf(F, len(df1)-1, len(df1) - 1)

    def fisher_plot(self, data):
        """
        Функция, принимающая на вход сегмент временного ряда.
        Возвращает сплайн с внутренними узлами, определяемыми по критерию Фишера.
        Используются сплайны порядка spline_degree (по умолчанию 3)
        """

        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        p_val = self.p_value
        k_s = self.spline_degree
        val = []
        ind = []

        ind.append(data.index[0])
        left = data.index[0]

        while left in data.index:
            right = left + k_s + 1
            flag = True

            if right > data.index[-1]:
                flag = False
                left = right
            while flag == True:
                v = data.loc[left:right].values 
                i = data.loc[left:right].index
                df = interpolate.LSQUnivariateSpline(i, v, [],  k=k_s)

                if self.FF(v, df(i)) < p_val:
                    ind.append(right-1)
                    left= right - 1
                    flag= False
                else:
                    right = right + 1
                    if right + 1 > data.index[-1]:
                        left = right + 1
                        flag = False    

        spl = interpolate.LSQUnivariateSpline(
            data.index, data.values, ind[1:-1],  k=k_s) 
        return spl



class DataLengthException(Exception):
    def __init__(self, *args, **kvargs):
        super(DataLengthException, self).__init__(*args, **kvargs)

class AlgebraicSplineFG(FeatureGenerator):
    def __init__(self, spline_degree=3, seg_number=5):
        super().__init__()
        self.seg_number = seg_number
        self.spline_degree = spline_degree

    def __call__(self, data):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        if len(data.index) < (self.spline_degree + 1) * self.seg_number:
            raise DataLengthException("Too short data, try to decrease seg_number")
        #spls = []
        spls_coeffs = np.array([])
        for i in range(0, self.seg_number):
            start = (len(data.index) * i) // self.seg_number
            end = (len(data.index) * (i + 1)) // self.seg_number
            spl = interpolate.LSQUnivariateSpline(
                data.index[start:end], data.values[start:end], t=[], k=self.spline_degree)
            #spls.append(spl)
            spls_coeffs = np.concatenate((spls_coeffs, spl.get_coeffs()))
        return spls_coeffs


import scipy as sp
class SSAFG(FeatureGenerator):
    def __init__(self, window_length=5):
        super().__init__()
        self.window_length = window_length

    def __call__(self, data):
        if not isinstance(data, numpy.ndarray):
            data = np.array(data)

        if len(data) < self.window_length:
            raise DataLengthException("SSAFG: Too short data, try to decrease window_length")
        
        def ts_folder(ts):
            for i in range(len(data) - self.window_length + 1):
                yield ts[i: i + self.window_length]

        X = np.array([x for x in ts_folder(data)])

        (U, s, U_t) = sp.linalg.svd(X.dot(X.T) / float(self.window_length))

        return np.sqrt(s[:self.window_length])



import numpy
from pyearth import Earth
from pyearth._basis import ConstantBasisFunction, LinearBasisFunction
class MultivariateAdaptiveRegressionSplinesFG(FeatureGenerator):
    def __init__(self, max_terms=8):
        super().__init__()
        self.max_terms = max_terms

    def __call__(self, data):
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        model = Earth(max_terms=self.max_terms)
        model.forward_pass(data.index, data.values)

        spls_coeffs = np.zeros(self.max_terms//2)
        i = 0
        
        #print(len(model.basis_))
        for basis_func in model.basis_:
            #print(basis_func)
            if isinstance(basis_func, ConstantBasisFunction):
                continue
            if isinstance(basis_func, LinearBasisFunction):
                continue
            
            knot = basis_func.get_knot()
            if (spls_coeffs[i] == knot):
                i += 1
                continue
            else:
                spls_coeffs[i] = knot
        return spls_coeffs[:]
