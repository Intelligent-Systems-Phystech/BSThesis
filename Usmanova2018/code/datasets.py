from functools import lru_cache
import numpy as np
import scipy
import scipy.io
import pandas as pd
from datetime import datetime, timedelta

def language_info(global_russian_):
    global global_russian
    global_russian = global_russian_

def dataset_info(dataset_prefix_, dataset_title_, dataset_x_, dataset_y_, x_units_):
    global dataset_prefix
    dataset_prefix = dataset_prefix_
    global dataset_title
    dataset_title = dataset_title_
    global dataset_x
    dataset_x = dataset_x_
    global dataset_y
    dataset_y = dataset_y_
    global x_units
    x_units = x_units_

@lru_cache(maxsize=None)
def load_russian_railroads_data(prefix = 'data/russian_railroads'):
    global global_russian
    dataset_info('railroads',
                 'Суммарные перевозки нефти РЖД (тонн по дням)' if global_russian else 'Total oil transportation by day (tons)',
                 'День' if global_russian else 'Day',
                 'Перевозки нефти (тонн)' if global_russian else 'Oil transportation (tons)',
                 'дни' if global_russian else 'days')
    mat = scipy.io.loadmat(prefix + '/omsk_2007_(2012_12_20).mat')
    data = mat['num']
    
    data_names = [x[0][0] for x in scipy.io.loadmat(prefix + '/data_names.mat')['data_names']]
    
    # Leave only 'From', 'To', 'Good code', 'Total weight'
    
    data = data[:,[0,1,3,5]]
    nan_rows = np.isnan(data).any(axis=1)
    
    #print('NaN rows to remove:', np.count_nonzero(nan_rows))
    
    #for line in data[0:10]:
    #    print(line)
        
    # Station codes are 6-digit, with first 2 digits representing region
    # Replace station codes with regions
    region_from = np.array(data[:,0] // 1e4, dtype=int)
    region_to = np.array(data[:,1] // 1e4, dtype=int)
    good_codes = np.array(data[:,2], dtype=int)
        
    #print(region_from)
        
    #print(np.count_nonzero(np.isnan(data)))
    
    #print(data_names)
    #print(data)
    
    # good codes = 1, 86 -> 83
    filter_good_codes = [3]
    #filter_region_from = [86]
    #filter_region_to = [83]
    
    leave_rows = ~nan_rows
    #leave_rows = np.logical_and(leave_rows, np.in1d(region_from, filter_region_from))
    #leave_rows = np.logical_and(leave_rows, np.in1d(region_to, filter_region_to))
    #print(leave_rows)
    leave_rows = np.logical_and(leave_rows, np.in1d(good_codes, filter_good_codes))
    
    dates = np.array([np.datetime64(datetime.strptime(x[0][0], '%d.%m.%Y')) for x in mat['dat'][leave_rows]])
    
    df = pd.DataFrame()
    df['date'] = dates
    df['from'] = region_from[leave_rows]
    df['to'] = region_to[leave_rows]
    df['good_code'] = good_codes[leave_rows]
    df['weight'] = data[:,3][leave_rows]
    
    df = pd.DataFrame(df.groupby('date')['weight'].sum().sort_index())
    days = (df.index - df.index[0]).days
    
    min_day = days.min()
    n_days = days.max() - min_day + 1
    
    ts = np.zeros(n_days)
    ts[days] = df.values.flatten()
    return ts

@lru_cache(maxsize=None)
def load_airline_data(prefix='data/airline'):
    dataset_info('airline', 'Число пассажиров авиакомпании (логарифм)', 'Месяц', 'Число пассажиров (логарифм)', 'месяцы')
    mat = scipy.io.loadmat(prefix + '/Data_Airline.mat')
    #print(mat)
    mat = np.array(mat['Data'])
    mat = mat.reshape(mat.shape[0])
    mat = np.log(mat)
    return mat

def load_simple_csv(file):
    datatypes = [('strfield','S10'), ('float_field','f4')]
    data = np.loadtxt(file, delimiter=',', dtype=datatypes, skiprows=1)
    return np.array([row[1] for row in data])

@lru_cache(maxsize=None)
def load_precipation_data(prefix='data/precipation_usa'):
    dataset_info('precipation', 'Осадки в мм (месячные; Истпорт, США, 1887-1950)', 'Месяц', 'Осадки в мм', 'месяцы')
    return load_simple_csv(prefix + '/precipitation-in-mm-eastport-usa.csv')[-400:]

@lru_cache(maxsize=None)
def load_chemical_readings_data(prefix='data/chem'):
    dataset_info('chemical', 'Концентрация в химической реакции', 'Время', 'Концентрация')
    return load_simple_csv(prefix + '/chemical-concentration-readings.csv')

@lru_cache(maxsize=None)
def load_volcanic_data(prefix='data/volcanic'):
    dataset_info('volcanic', 'Индекс влияния вулканического пепла на экологию (северное полушарие, 1500-1969)', 'Год', 'Индекс влияния пепла', 'годы')
    return load_simple_csv(prefix + '/volcanic-dust-veil-index-norther.csv')[:-50]

@lru_cache(maxsize=None)
def load_poland_electricity_consumption_data(prefix='data/electricity_consumption_poland'):
    dataset_info('poland', 'Потребление электроэнергии в Польше (по дням)', 'День', 'Потребление электроэнергии (Польша)', 'дни')
    datatypes = [('s1','S10'), ('s2','S10'), ('s3','S10')]
    datatypes = datatypes + [('f' + str(i), 'f4') for i in range(24)]
    data = np.loadtxt(prefix + '/data.csv', delimiter=',', dtype=datatypes, skiprows=1)
    all_data = []
    for row in data:
        for i in range(24):
            all_data.append(row[3 + i])
    return np.array(all_data)[-500:]

@lru_cache(maxsize=None)
def load_german_electricity_price_data(prefix='data/electricity_price_german'):
    dataset_info('german', 'Цена на электроэнергию в Германии (по дням)', 'День', 'Цена электроэнергии (Германия)', 'дни')
    datatypes = [('s1','S10')]
    datatypes = datatypes + [('f' + str(i), 'f4') for i in range(24)]
    data = np.loadtxt(prefix + '/GermanSpotPrice.csv', delimiter=',', dtype=datatypes, skiprows=1)
    all_data = []
    for row in data:
        for i in range(24):
            all_data.append(row[1 + i])
    return np.array(all_data)[-500:]

def generate_synthetic_data(mu=1, scale=1):
    dataset_info('synthetic', 'Синтетические данные (синус, зашумлённый распределением Вальда, $\mu=' + str(mu) + ', \lambda=' + str(scale) + '$)', 'x', '$\sin(x)+\\varepsilon_{Wald}$', 'отсчёты')
    x = np.linspace(0, 20, 200)
    np.random.seed(12345)

    distribution = scipy.stats.invgauss(mu, scale=scale)
    noise = np.log(distribution.rvs(len(x))) * 0.25

    #plt.hist(noise, bins=20)
    return np.sin(x) + noise
