import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from numpy import linalg
from datetime import datetime


def plot_ts(ts_list, figsize=(10, 5), title=None, title_size=18, filename=None, 
            x_label=None, y_label=None):
    plt.figure(figsize=figsize)
    plt.grid()
    
    if x_label is not None:
        plt.xlabel(x_label[0], fontsize=x_label[1])
    if y_label is not None:
        plt.ylabel(y_label[0], fontsize=y_label[1])
    
    if title is not None:
        plt.title(title, fontsize=title_size)
    
    for ts in ts_list:
        if len(ts) == 3:
            plt.plot(ts[1], ts[0], label=ts[2])
            plt.legend(fontsize=12)
        else:
            plt.plot(ts[1], ts[0])
    
    if filename is not None:
        plt.savefig(filename)
    
    plt.show()
    

def hankel_restore_one_ts(hankel_matrix):
    L = hankel_matrix.shape[0]
    ts = np.zeros(hankel_matrix.shape[1] + L - 1)
    ts[0] = hankel_matrix[0][0]
    #print(hankel_matrix)
    #print('hankel shape', hankel_matrix.shape)
    for i in range(1, hankel_matrix.shape[1] + L - 1):
        #cnt = 0
        #s = 0
        #s_list = []
        #for j in range(min(i + 1, hankel_matrix.shape[0])):
        #    if i - j < hankel_matrix.shape[1]:
        #        s += hankel_matrix[j][i - j]
        #        s_list.append(hankel_matrix[j][i - j])
        #        cnt += 1
        #    
        #s /= cnt
        
        ts[i] = hankel_matrix.T[:,::-1].diagonal(L - i - 1).mean()
        #assert(abs(s - value) < 1e-6)
        #print('s ', s, s_list, diag, diag.mean())
        #if i > 10:
        #    raise RuntimeError('lol')
        
        #ts[i] = s
        
    return ts


def hankel_restore_two_ts(hankel_matrix):
    L = hankel_matrix.shape[0]
    k = int(hankel_matrix.shape[1] / 2)
    
    matrix_1 = (hankel_matrix.T[:k][:]).T
    matrix_2 = (hankel_matrix.T[k:][:]).T
    
    return hankel_restore_one_ts(matrix_1), hankel_restore_one_ts(matrix_2)


def hankel_restore_n_ts(hankel_matrix, ts_number):
    assert(hankel_matrix.shape[1] % ts_number == 0)
    k = int(hankel_matrix.shape[1] / ts_number)
    
    ts_list = []
    
    for i in range(ts_number):
        matrix = hankel_matrix[:, i * k : (i + 1) * k]
        ts_list.append(hankel_restore_one_ts(matrix))
        
    return ts_list


def ssa_n_ts(ts_list, restore_rank, L, comp_idx_list=None):
    ts_number = len(ts_list)
    
    #print('ts',len(ts_list[0]))
    
    hankel_matrix = np.array((scipy.linalg.hankel(ts_list[0], np.zeros(L)).T)[:,:-L+1])
    
    for ts_index in range(1, ts_number):
        X = np.array((scipy.linalg.hankel(ts_list[ts_index], np.zeros(L)).T)[:,:-L+1])
        
        
        if comp_idx_list is not None:
            # print('comps is not None')
            U, s, V = np.linalg.svd(X, full_matrices=False)
            s_pr = np.zeros(len(s))
            s_pr[comp_idx_list] = 1
            S = np.zeros((U.shape[1], V.shape[0]))
            S[:len(s), :len(s)] = s_pr
            
            # print(X.shape, U.shape, S.shape, V.shape)
            
            X = U.dot(S.dot(V))
            
        
        
        hankel_matrix = np.hstack((hankel_matrix, X))
        
    #print('res shape', hankel_matrix.shape, 'nts', ts_number)
    
    t0 = datetime.now()
    U, s, V = np.linalg.svd(hankel_matrix, full_matrices=False)
    # print('s:', s)
    t1 = datetime.now()
    X_approx = np.zeros(hankel_matrix.shape)
    regression_coefs = np.zeros(hankel_matrix.shape[0] - 1)
    
    verticality = 0
    for i in range(restore_rank):
        a = U[:,i].reshape(U.shape[0], 1) # column
        X_approx += a.dot(a.T.dot(hankel_matrix)) 

        pi = a[-1,0]
        verticality += pi * pi
        regression_coefs += pi * U[:-1,i]

    regression_coefs /= (1 - verticality)
    t2 = datetime.now()
    
    #print('svd', (t1 - t0).total_seconds(), 'regression', (t2 - t1).total_seconds())
    
    return X_approx, regression_coefs


def mssa_k(ts_list, restore_rank, L):
    ts_number = len(ts_list)
    
    hankel_matrix = np.array((scipy.linalg.hankel(ts_list[0], np.zeros(L)).T)[:,:-L+1])
    
    for ts_index in range(1, ts_number):
        X = np.array((scipy.linalg.hankel(ts_list[ts_index], np.zeros(L)).T)[:,:-L+1])
        hankel_matrix = np.hstack((hankel_matrix, X))

    U, s, V = np.linalg.svd(hankel_matrix, full_matrices=False)
    X_approx = np.zeros(hankel_matrix.shape)
    
    for i in range(restore_rank):
        a = U[:,i].reshape(U.shape[0], 1) # column
        X_approx += a.dot(a.T.dot(hankel_matrix)) 
        
    


def make_ssa(svd, restore_rank, hankel_matrix):
    X_hat = np.zeros(hankel_matrix.shape)
    # U, s, V = np.linalg.svd(hankel_matrix, full_matrices=False)
    U, s, V = svd
    
    regression_coefs = np.zeros(hankel_matrix.shape[0] - 1)

    verticality = 0
    for i in range(restore_rank):
        a = U[:,i].reshape(U.shape[0], 1) # column
        X_hat += a.dot(a.T).dot(hankel_matrix) 

        pi = a[-1,0]
        verticality += pi * pi
        regression_coefs += pi * U[:-1,i]

    regression_coefs /= (1 - verticality)
    
    return X_hat, regression_coefs


def make_forecast_one_point(ts_train, regression_coefs, real_value):
    forecast_value = (regression_coefs * ts_train[-regression_coefs.shape[0]:]).sum()
    return forecast_value, abs(forecast_value - real_value) ** 2


def mse(ts_approx, ts_real):
    error = (ts_approx - ts_real) ** 2 
    return error.mean()

def scale_ts(ts):
    avg = sum(abs(ts)) / len(ts)
    return np.array(ts) / avg

def get_mssa_result(mssa_results, 
                    restore_rank, L, 
                    all_data, used_names_list, train_bounds, make_key, comp_idx_list=None):
    """
    parametrs:  все временные ряды, 
                названия рядов, которые будут использоваться, 
                границы обучающей выборки 
                функция, которая делает ключ по названиям рядов
    """
    names_key = make_key(used_names_list, ', ')
    
    train_all = all_data.loc[train_bounds[0]:train_bounds[1]]
    used_train_part = []
    
    for name in used_names_list:
        used_train_part.append(train_all[name])
    
    if names_key not in mssa_results:
        mssa_results[names_key] = {}
    if (restore_rank, L) not in mssa_results[names_key]:
        mssa_results[names_key][(restore_rank, L)] = {}
      
    if train_bounds not in mssa_results[names_key][(restore_rank, L)]:
        mssa_results[names_key][(restore_rank, L)][train_bounds] = ssa_n_ts(used_train_part, 
                                                                            restore_rank, L, comp_idx_list)[1]
    
    return mssa_results[names_key][(restore_rank, L)][train_bounds]


def mean_forecast_error_for_n_points(mssa_results,
                                     restore_rank, L, 
                                     all_data, used_names_list, 
                                     first_train_bounds, 
                                     test_len, name_for_forecast, make_key, comp_idx_list=None):
    """
    parametrs:  все временные ряды,
                названия рядов, котрые будут использоваться,
                границы первой обучающей выборки,
                длина теста,
                название ряда, который хотим прогнозировать, 
                функция, которая делает ключ по названиям рядов
    """
    names_key = make_key(used_names_list, ', ')
    
    forecasted_ts = all_data[name_for_forecast]
    
    errors = np.zeros(test_len)
    forecasted_values = np.zeros(test_len)
    
    for i in range(test_len):
        # if i % 100 == 0:
        #     print(i)
            
        train_bounds = tuple(np.array(first_train_bounds) + i)
        
        reg_coefs = get_mssa_result(mssa_results, restore_rank, L, all_data, 
                                    used_names_list, train_bounds, make_key, comp_idx_list)
        
        forecasted_ts = all_data[name_for_forecast][:first_train_bounds[1] + i]
        real_value = all_data[name_for_forecast][first_train_bounds[1] + i]
        
        forecasted_values[i], errors[i] = make_forecast_one_point(forecasted_ts, reg_coefs, real_value)
        
    return forecasted_values, errors.mean()


def mean_forecast_const_reg_coefs(mssa_results, 
                                  restore_rank, L, 
                                  all_data, used_names_list, 
                                  first_train_bounds, 
                                  test_len, name_for_forecast, make_key):
    """
    Этой функцией лучше не пользоваться
    
    parametrs:  все временные ряды,
                названия рядов, котрые будут использоваться,
                границы первой обучающей выборки,
                длина теста,
                название ряда, который хотим прогнозировать, 
                функция, которая делает ключ по названиям рядов
    """
    names_key = ' '.join(used_names_list)
    
    forecasted_ts = all_data[name_for_forecast]
    
    errors = np.zeros(test_len)
    forecasted_values = np.zeros(test_len)
    
    reg_coefs = get_mssa_result(mssa_results, restore_rank, L, all_data, 
                                used_names_list, first_train_bounds, make_key)

    for i in range(test_len):
        forecasted_ts = all_data[name_for_forecast][:first_train_bounds[1] + i]
        real_value = all_data[name_for_forecast][first_train_bounds[1] + i]
        
        forecasted_values[i], errors[i] = make_forecast_one_point(forecasted_ts, reg_coefs, real_value)
        
    return forecasted_values, errors.mean()
