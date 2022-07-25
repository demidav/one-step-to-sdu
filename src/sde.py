import numpy as np
import scipy.stats as st
from scipy.linalg import fractional_matrix_power
"""SDE integrators"""
# -----------------------------------------------------------------------------
#   Функции для генерирования траектории винеровского процесса
# -----------------------------------------------------------------------------


def __time(N, interval=(0.0, 10.0)):
    """Функция разбивает временной интервал на части"""
    (t_0, T) = interval

    dt = (T - t_0)/float(N)
    t = np.arange(dt, T+dt, dt)
    return (dt, t)


def __scalar_wiener_process(N, dt):
    """Функция генерирует траекторию скалярного винеровского процесса"""
    ##
    dW = np.sqrt(dt)*np.random.normal(loc=0, scale=0.0001, size=N)
    W = np.cumsum(dW)
    ##
    return (dW, W)


def __multidimensional_wiener_process(N, dim, dt):
    """Функция генерирует многомерный (размерность dim) винеровский процесс"""
    dW = np.empty(shape=(N, dim))
    W = np.empty(shape=(N, dim))
    for i in range(0, dim):
        dW[:, i], W[:, i] = __scalar_wiener_process(N, dt)

    return (dW, W)


def __cov_multidimensional_wiener_process(N, dim, dt):
    """Функция генерирует многомерный (размерность dim) винеровский процесс
    с использованием функции multivariate_normal. Процессы W1,W2,...Wn могут
    коррелировать, если задать матрицу ковариации cov не диагональной"""
    cov = dt*np.identity(dim)
    mean = np.zeros(dim)
    dW = np.random.multivariate_normal(mean, cov, size=N)
    W = np.cumsum(dW, axis=0)
    return (dW, W)


def wiener_process(N, dim=1, interval=(0.0, 10.0)):
    """Функция обертка для генерирования винеровского процесса"""
    dt, t = __time(N, interval)
    size = len(t)
    if dim == 1:
        dW, W = __scalar_wiener_process(size, dt)
    elif dim > 1:
        dW, W = __cov_multidimensional_wiener_process(size, dim, dt)
    else:
        dW, W = (0.0, 0.0)
    return (dt, t, dW, W)


# -----------------------------------------------------------------------------
#   Функции для генерирования кратных интегралов Ито
# -----------------------------------------------------------------------------
def Ito1W1(dW):
    """Однократный интеграл Ито от 1 по dW
    для скалярного винеровского процесса"""
    return dW


def Ito2W1(dW, h):
    """Генерирование значений двухкратного интеграла
    Ито для скалярного процесса Винера"""
    N = len(dW)
    Ito = np.empty(shape=(N, 2, 2))
    dzeta = np.sqrt(h)*np.random.normal(loc=1.0, scale=0.0001, size=N)
    Ito[:, 1, 1] = 0.5*(dW**2 - h)
    Ito[:, 1, 0] = 0.5*h*(dW + dzeta/np.sqrt(3))
    Ito[:, 0, 1] = 0.5*h*(dW - dzeta/np.sqrt(3))

    return Ito


def __Levi_integral(dw, h, m, num):
    """Интеграл Леви для вычисления аппроксимации интеграла Ито
    num --- число членов ряда"""
    X = np.random.normal(loc=0.0, scale=0.0001, size=(m, num))
    Y = np.random.normal(loc=0.0, scale=0.0001, size=(m, num))

    K = range(0, num, 1)

    A1 = [np.outer(X[:, k], (Y[:, k] + np.sqrt(2.0/h)*dw)) for k in K]
    A2 = [np.outer((Y[:, k] + np.sqrt(2.0/h)*dw), X[:, k]) for k in K]
    A = ((1.0/(k+1))*(A1[k] - A2[k]) for k in K)
    A = (h/np.pi)*np.sum(A)

    return A


def Ito1Wm(dW, h):
    """Генерирование значений однократного интеграла Ито для многомерного
    винеровского процесса"""
    (N, m) = dW.shape
    Ito = np.empty(shape=(N, m+1))
    Ito[:, 0] = h
    Ito[:, 1:] = dW[:, :]
    return Ito


def Ito2Wm(dW, h, n):
    """Вычисление интеграла Ито (по точной формуле или по аппроксимации)
    Возвращает список матриц I(i,j)"""
    (N, m) = dW.shape
    E = np.identity(m)
    Ito = np.empty(shape=(N, m+1, m+1))
    dzeta = np.random.normal(loc=0, scale=h, size=(N, m))
    Ito[:, 0, 0] = h
    for dw, I, dz in zip(dW, Ito, dzeta):
        I[0, 1:] = 0.5*h*(dw - dz / np.sqrt(3))
        I[1:, 0] = 0.5*h*(dw + dz / np.sqrt(3))
        I[1:, 1:] = 0.5*(np.outer(dw, dw) - h*E) + __Levi_integral(dw, h, m, n)
    return Ito


def Ito3W1(dW, h):
    """Генерирование трехкратного интеграла Ито
    для скалярного процесса Винера"""
    I = 0.5*(dW**2 - h)
    I = (1.0 / 6.0)*h*(I**3 - 3.0*h*dW)
    return I

# -----------------------------------------------------------------------------
# Непосредственно численные методы
# -----------------------------------------------------------------------------


def EulerMaruyama(f, g, h, x_0, dW):
    '''Метод Эйлера-Маруйамы для скалярного винеровского процесса'''
    x = []
    x_tmp = x_0
    for dw in dW:
        x_tmp = x_tmp + f(x_tmp)*h + g(x_tmp)*dw
        x.append(x_tmp)
    return np.asarray(x)


def EulerMaruyamaWm(f, g, h, x_0, dW):
    '''Метод Эйлера-Маруйамы для многомерного винеровского процесса'''
    x = []
    x_tmp = x_0
    for dw in dW:
        #x_tmp = x_tmp + f(x_tmp)*h + np.tensordot(g(x_tmp), dw, axes=(1, 0))
        x_tmp = x_tmp + f(x_tmp)*h + np.tensordot(fractional_matrix_power(g(x_tmp), 0.5), dw, axes=(1, 0))
        x.append(x_tmp)
    return np.asarray(x)

def EulerMaruyamaWm_NEW(f, g, h, x_0, dW):
    '''Метод Эйлера-Маруйамы для многомерного винеровского процесса'''
    x = []
    x_tmp = x_0
    for dw in dW:
        #x_tmp = x_tmp + f(x_tmp)*h + np.tensordot(g(x_tmp), dw, axes=(1, 0))
        #x_tmp = x_tmp + f(x_tmp)*h + np.tensordot(fractional_matrix_power(g(x_tmp), 0.5), dw, axes=(1, 0))
        x_tmp = x_tmp + f(x_tmp)*h + np.tensordot(fractional_matrix_power(g(x_tmp), 0.5), dw, axes=(1, 0))
        x_tmp[x_tmp <= 0]=0.0
        x.append(x_tmp)
    return np.asarray(x)

def __strong_method_selector(name):
    """Заполнение массивов таблицы Бутчера для конкретного метода"""
    if name == 'SRK1W1':
        s = 4
        a = np.array([1.0/3.0, 2.0/3.0, 0.0, 0.0])
        b1 = np.array([-1.0, 4.0/3.0, 2.0/3.0, 0.0])
        b2 = np.array([-1.0, 4.0/3.0, -1.0/3.0, 0.0])
        b3 = np.array([2.0, -4.0/3.0, -2.0/3.0, 0.0])
        b4 = np.array([-2.0, 5.0/3.0, -2.0/3.0, 1.0])

        c1 = np.array([0.0, 3.0/4.0, 0.0, 0.0])
        c2 = np.array([0.0, 1.0/4.0, 1.0, 1.0/4.0])

        A0 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [3.0/4.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]])
        A1 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [1.0/4.0, 0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0/4.0, 0.0]])
        B0 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [3.0/2.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]])
        B1 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0],
                       [-5.0, 3.0, 0.5, 0.0]])
    elif name == 'SRK2W1':
        s = 4
        a = np.array([1.0/6.0, 1.0/6.0, 2.0/3.0, 0.0])
        b1 = np.array([-1.0, +4.0/3.0, +2.0/3.0, 0.0])
        b2 = np.array([+1.0, -4.0/3.0, +1.0/3.0, 0.0])
        b3 = np.array([+2.0, -4.0/3.0, -2.0/3.0, 0.0])
        b4 = np.array([-2.0, +5.0/3.0, -2.0/3.0, 1.0])

        c1 = np.array([0.0, 1.0, 0.5, 0.0])
        c2 = np.array([0.0, 0.25, 1.0, 0.25])

        A0 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0],
                       [1.0/4.0, 1.0/4.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]])
        A1 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [1.0/4.0, 0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0/4.0, 0.0]])
        B0 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [1.0, 0.5, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]])
        B1 = np.array([[0.0, 0.0, 0.0, 0.0],
                       [-0.5, 0.0, 0.0, 0.0],
                       [+1.0, 0.0, 0.0, 0.0],
                       [+2.0, -1.0, 0.5, 0.0]])
    elif name == 'KlPl':
        # Kloeden an Platen
        s = 2
        a = np.array([1.0, 0.0])
        b1 = np.array([1.0, 0.0])
        b2 = np.array([-1.0, 1.0])
        b3 = np.array([0.0, 0.0])
        b4 = np.array([0.0, 0.0])
        c1 = np.array([0.0, 0.0])
        c2 = np.array([0.0, 0.0])

        A0 = np.array([[0.0, 0.0],
                       [0.0, 0.0]])
        A1 = np.array([[0.0, 0.0],
                       [1.0, 0.0]])
        B0 = np.array([[0.0, 0.0],
                       [0.0, 0.0]])
        B1 = np.array([[0.0, 0.0],
                       [1.0, 0.0]])
    elif name == 'SRK1Wm':
        s = 3
        a = np.array([1.0, 0.0, 0.0])
        b1 = np.array([1.0, 0.0, 0.0])
        b2 = np.array([0.0, 0.5, -0.5])

        c1 = np.array([0.0, 0.0, 0.0])
        c2 = np.array([0.0, 0.0, 0.0])

        A0 = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        A1 = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        B0 = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        B1 = np.array([[0.0, 0.0, 0.0],
                       [+1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0]])
    elif name == 'SRK2Wm':
        s = 3
        a = np.array([0.5, 0.5, 0.0])
        b1 = np.array([1.0, 0.0, 0.0])
        b2 = np.array([0.0, 0.5, -0.5])

        c1 = np.array([0.0, 1.0, 0.0])
        c2 = np.array([0.0, 1.0, 1.0])

        A0 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        A1 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]])
        B0 = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        B1 = np.array([[0.0, 0.0, 0.0],
                       [+1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0]])
    # end if
    if name in {'SRK1W1', 'SRK2W1', 'KlPl'}:
        return (s, a, b1, b2, b3, b4, c1, c2, A0, A1, B0, B1)
    elif name in {'SRK1Wm', 'SRK2Wm'}:
        return (s, a, b1, b2, c1, c2, A0, A1, B0, B1)


def __weak_method_selector(name):
    """Заполнение массивов таблицы Бутчера для конкретного метода"""
    if name == 'SRK1Wm':
        s = 3
        a = np.array([0.1, 3.0/14.0, 24.0/35.0])
        b1 = np.array([1.0, -1.0, -1.0])
        b2 = np.array([0.0, 1.0, -1.0])
        b3 = np.array([0.5, -0.25, -0.25])
        b4 = np.array([0.0, 0.5, -0.5])
        c0 = np.array([0.0, 1.0, 5.0/12.0])
        c1 = np.array([0.0, 0.25, 0.25])
        c2 = np.array([0.0, 0.0, 0.0])
        A0 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [25.0/144.0, 35.0/144.0, 0.0]])
        A1 = np.array([[0.0, 0.0, 0.0],
                       [0.25, 0.0, 0.0],
                       [0.25, 0.0, 0.0]])
        A2 = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        B0 = np.array([[0.0, 0.0, 0.0],
                       [1.0/3.0, 0.0, 0.0],
                       [-5.0/6.0, 0.0, 0.0]])
        B1 = np.array([[0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [-0.5, 0.0, 0.0]])
        B2 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0]])
    elif name == 'SRK2Wm':
        s = 3
        a = np.array([0.5, 0.5, 0.0])
        b1 = np.array([0.5, 0.25, 0.25])
        b2 = np.array([0.0, 0.5, -0.5])
        b3 = np.array([-0.5, 0.25, 0.25])
        b4 = np.array([0.0, 0.5, -0.5])
        c0 = np.array([0.0, 1.0, 0.0])
        c1 = np.array([0.0, 1.0, 1.0])
        c2 = np.array([0.0, 0.0, 0.0])
        A0 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        A1 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]])
        A2 = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        B0 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
        B1 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0]])
        B2 = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0]])
    # end if
    return (s, a, b1, b2, b3, b4, c0, c1, c2, A0, A1, A2, B0, B1, B2)


def strongSRKW1(f, g, h, x_0, dW, name='SRK2W1'):
    """Стохастический метод Рунге-Кутты сильного порядка p = 1.5
    для скалярного процесса Винера"""

    if(not isinstance(name, str)):
        name = 'SRK2W1'
        print("Argument `name` is not string! Using `name = 'SRK2W1'`")

    (s, a, b1, b2, b3, b4,
     c1, c2, A0, A1, B0, B1) = __strong_method_selector(name)

    sqh = np.sqrt(h)
    x = []
    x_tmp = x_0

    X0 = np.zeros(s)
    X1 = np.zeros(s)

    I1 = Ito1W1(dW)
    I2 = Ito2W1(dW, h)
    I3 = Ito3W1(dW, h)
    for i1, i2, i3 in zip(I1, I2, I3):
        for i in range(0, s, 1):
            X0[i] = x_tmp + np.dot(A0[i, :], f(X0))*h
            X0[i] += np.dot(B0[i, :], g(X1))*i2[1, 0]/h
            ##
            X1[i] = x_tmp + np.dot(A1[i, :], f(X0))*h
            X1[i] += np.dot(B1[i, :], g(X1))*sqh
        # end for
        x_tmp = x_tmp + np.dot(a, f(X0))*h + np.dot(
            (b1*i1 + b2*i2[1, 1]/sqh + (b3*i2[1, 0] + b4*i3)/h), g(X1))
        x.append(x_tmp)
    # end for
    return x


def oldstrongSRKp1Wm(f, G, h, x_0, dW, name='SRK1Wm'):
    """Стохастический метод Рунге-Кутты сильного порядка p = 1.0
    для многомерного винеровского процесса. Функция медленная и лучше вместо
    нее использовать `strongSRKp1Wm` или специаальные функции для конкретного
    метода"""

    s, a, b1, b2, c1, c2, A0, A1, B0, B1 = __strong_method_selector(name)

    (N, m) = dW.shape

    sqh = np.sqrt(h)
    x = []
    x_tmp = x_0
    I1 = Ito1Wm(dW, h)
    I2 = Ito2Wm(dW, h, n=10)

    X = np.zeros(shape=(m+1, s, m))
    # Чудо-юдо циклы
    for i1, i2 in zip(I1, I2):
        for i in range(s):
            S01 = np.array([
                np.sum([
                    A0[i, j]*f(X[0, j, :])[n]*h
                    for j in range(s)])
                for n in range(m)])
            # -------------------
            S02 = np.array([np.sum([
                    [B0[i, j]*G(X[l, j, :])[n, l]*i1[l+1]
                        for j in range(s)]
                    for l in range(m)])
                for n in range(m)])
            # -------------------
            X[0, i, :] = x_tmp + S01 + S02
            for k in range(1, m+1):
                S11 = np.array([
                    np.sum([
                        A1[i, j]*f(X[0, j, :])[n]*h
                        for j in range(s)])
                    for n in range(m)])
                # -------------------
                S12 = np.array([np.sum([
                        [B1[i, j]*G(X[l, j, :])[n, l]*i2[l+1, k]/sqh
                            for j in range(s)]
                        for l in range(m)])
                    for n in range(m)])
                # -------------------
                X[k, i, :] = x_tmp + S11 + S12
        S1 = np.array([
                np.sum([
                    a[i]*f(X[0, i, :])[n]*h
                    for i in range(s)])
                for n in range(m)])
        # -------------------
        S2 = np.array([
                np.sum([
                    [(b1[i]*i1[k+1] + b2[i]*sqh)*G(X[k+1, i, :])[n, k]
                        for i in range(s)]
                    for k in range(m)])
                for n in range(m)])
        # -------------------
        x_tmp = x_tmp + S1 + S2
        x.append(x_tmp)
    return np.asarray(x)


def strongSRKp1Wm(f, G, h, x_0, dW, name='SRK1Wm'):
    """Стохастический метод Рунге-Кутты сильного порядка p = 1.0
    для многомерного винеровского процесса"""

    (N, m) = dW.shape
    (s, a, b1, b2, c1, c2, A0, A1, B0, B1) = __strong_method_selector(name)

    sqh = np.sqrt(h)

    x_num = []
    x_tmp = x_0
    I1 = Ito1Wm(dW, h)
    I2 = Ito2Wm(dW, h, n=10)

    
    
    X = np.zeros(shape=(m+1, s, m))
    # Здесь индекс a будет обозначать \alpha
    for i1, i2 in zip(I1, I2):
        for i in range(s):
            _f = np.array([f(var) for var in X[0]])
            _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
            #_G = np.array([[fractional_matrix_power(G(var2), 0.5) for var2 in var1] for var1 in X[1:]])
            
            X[0, i, :] = x_tmp + np.einsum("j, ja", A0[i], _f*h)
            X[0, i, :] += np.einsum("j, ljal, l", B0[i], _G, i1[1:])

            for k in range(1, m+1, 1):
                _f = np.array([f(var) for var in X[0]])
                _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
                #_G = np.array([[fractional_matrix_power(G(var2), 0.5) for var2 in var1] for var1 in X[1:]])
                
                X[k, i, :] = x_tmp + np.einsum("j, ja", A1[i], _f*h)
                X[k, i, :] += np.einsum("j, ljal, l", B1[i], _G, i2[1:, k])/sqh

        _f = np.array([f(var) for var in X[0]])
        _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
        #_G = np.array([[fractional_matrix_power(G(var2), 0.5) for var2 in var1] for var1 in X[1:]])
        
        x_tmp = x_tmp + np.einsum("i, ia", a, _f*h)
        x_tmp += np.einsum("i, k, kiak", b1, i1[1:], _G)
        x_tmp += np.einsum("i, kiak", b2*sqh, _G)

        x_num.append(x_tmp)

    return np.asarray(x_num)


def strongSRKp1Wm1(f, G, h, x_0, dW):
    """Частный случай стохастического метода Рунге-Кутты сильного порядка
    для p = 1.0 и многомерного процесса Винера. Эта функция нужна для
    быстродействия"""

    (N, m) = dW.shape
    s = 3

    sqh = np.sqrt(h)

    x_num = []
    x_tmp = np.asarray(x_0)
    I1 = Ito1Wm(dW, h)
    I2 = Ito2Wm(dW, h, n=10)

    X = np.zeros(shape=(m+1, s, m))
    for i1, i2 in zip(I1, I2):
        for i in range(s):
            X[0, i, :] = x_tmp
        for k in range(1, m+1):
            X[k, 0, :] = x_tmp

        _G = G(x_tmp)
        for k in range(1, m+1):
            X[k, 1, :] = x_tmp + np.tensordot(_G, i2[1:, k]/sqh, axes=(1, 0))
            X[k, 2, :] = x_tmp - np.tensordot(_G, i2[1:, k]/sqh, axes=(1, 0))

        _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])
        x_tmp = x_tmp + f(x_tmp)*h + np.einsum("k,kak", i1[1:], _G[:, 0, :, :])
        x_tmp += 0.5*sqh*np.einsum("kak", _G[:, 1, :, :])
        x_tmp -= 0.5*sqh*np.einsum("kak", _G[:, 2, :, :])
        
        #if x_tmp.min() < 0.0:
        #    break
        # Обнуляем отрицательные  ______________!!!!!!_________-
        x_tmp[x_tmp <= 0]=0.0
        
        x_num.append(x_tmp)

    return np.asarray(x_num)


# ----------------------------------------------------------------------------
# Численные методы со слабой сходимостью
# ----------------------------------------------------------------------------
def n_point_distribution(values, probabilities, shape):
    """n-point distribution"""
    index = np.arange(len(values))
    n_point = st.rv_discrete(name='n_point', values=(index, probabilities))
    num = np.prod(shape)
    res = np.asarray(values)[n_point.rvs(size=num)].reshape(shape)
    return res


def weakIto(h, dim, N):
    """Функция вычисляет аппроксимации для интегралов Ито в случае слабой
    сходимости.
    h --- step
    dim --- SDE dimension
    N --- the number of points on the time interval
    """
    # Two point distribution
    x = (-np.sqrt(h), np.sqrt(h))
    p = (0.5, 0.5)
    I_tilde = n_point_distribution(x, p, shape=(N, dim))

    # Three point distribution
    x = (-np.sqrt(3.0*h), 0.0, np.sqrt(3.0*h))
    p = (1.0/6.0, 2.0/3.0, 1.0/6.0)
    I_hat = n_point_distribution(x, p, shape=(N, dim))

    I = np.empty(shape=(N, dim, dim))

    it = np.nditer(I, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        i = it.multi_index[0]
        k = it.multi_index[1]
        l = it.multi_index[2]
        if k < l:
            it[0] = 0.5*(I_hat[i, k]*I_hat[i, l] - np.sqrt(h)*I_tilde[i, k])
        elif l < k:
            it[0] = 0.5*(I_hat[i, k]*I_hat[i, l] + np.sqrt(h)*I_tilde[i, l])
        elif l == k:
            it[0] = 0.5*(I_hat[i, k]**2 - h)
        it.iternext()

    return (I_hat, I)


def weakSRKp2Wm(f, G, h, x_0, dW, name='SRK1Wm'):
    """Стохастический метод Рунге-Кутты слабого порядка p = 2.0
    для многомерного винеровского процесса"""

    (N, dim) = dW.shape
    (s, a, b1, b2, b3, b4, c0, c1,
     c2, A0, A1, A2, B0, B1, B2) = __weak_method_selector(name)

    sqh = np.sqrt(h)

    x_num = []
    x_tmp = x_0
    (I1, I2) = weakIto(h, dim, N)

    X = np.zeros(shape=(dim+1, s, dim))
    X_hat = np.zeros(shape=(dim, s, dim))
    # Здесь индекс a будет обозначать \alpha
    for i1, i2 in zip(I1, I2):
        for i in range(s):
            _f = np.array([f(var) for var in X[0]])
            _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])

            X[0, i, :] = x_tmp + np.einsum("j, ja", A0[i], _f*h)
            X[0, i, :] += np.einsum("j, ljal, l", B0[i], _G, i1)

            for k in range(1, dim+1, 1):
                _f = np.array([f(var) for var in X[0]])
                _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])

                X[k, i, :] = x_tmp + np.einsum("j, ja", A1[i], _f*h)
                X[k, i, :] += np.einsum("j, ja", B1[i], _G[k, :, :, k])*sqh

                X_hat[k, i, :] = x_tmp + np.einsum("j, ja", A2[i], _f*h)
                # из суммы надо исключить случай l = k, поэтому придется
                # суммировать циклами, а не einsum
                X_hat[k, i, :] += np.array([np.sum([
                                    [B2[i, j]*_G[l, j, alpha, l]*i2[k-1, l]/sqh for l in range(dim) if l != k-1] for j in range(s)]) for alpha in range(dim)])

        _f = np.array([f(var) for var in X[0]])
        _G = np.array([[G(var2) for var2 in var1] for var1 in X[1:]])

        x_tmp = x_tmp + np.einsum("i, ia", a, _f)*h
        x_tmp += np.einsum("i, k, kiak", b1, i1, _G)
        x_tmp += np.einsum("i, kiak, kk", b2, _G, i2)/sqh
        x_tmp += np.einsum("i, kiak, k", b3, _G, i1)
        x_tmp += np.einsum("i, kiak", b4, _G)*sqh

        x_num.append(x_tmp)

    return np.asarray(x_num)
