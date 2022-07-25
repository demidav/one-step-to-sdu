import numpy as np
from sympy import *
import sympy as sp

# -----------------------------------------------------------------------------
#   Функции для получения коэффициентов уравнения Фоккера-Планка
# -----------------------------------------------------------------------------

def Π(x, n):
    """Элемент x^{i}!/(x^{i} - N^{j}_{i})! в произведении s^{+}_{j}(x) 
    x -- символ, n -- целое число"""
    return sp.prod([x for i in range(n)])


def S_plus(X, K_plus, N):
    """Вычисляет [s^{+}_{1}(x),...,s^{+}_{s}(x)]"""
    res = []
    for i in range(len(K_plus)):
        # Вычисляем элементы произведения
        Πs = [Π(x, int(n)) for (x, n) in zip(X, N[i, :])]
        # Находим само произведение
        res.append(K_plus[i]* sp.prod(Πs))
    # в результате получается список [s^{+}_1, s^{+}_2, s^{+}_3, ..., s^{+}_s]
    return res


def S_minus(X, K_minus, M):
    """Вычисляет [s^{-}_{1}(x),...,s^{-}_{s}(x)]"""
    res = []
    for i in range(len(K_minus)):
        # Вычисляем элементы произведения
        Πs = [Π(x, int(n)) for (x, n) in zip(X, M[i, :])]
        # Находим само произведение
        res.append(K_minus[i]* sp.prod(Πs))
    # в результате получается список [s^{-}_1, s^{-}_2, s^{-}_3, ..., s^{-}_s]
    return res


def S(X, K_plus, K_minus, N, M):
    """Вычисляет разность S_plus - S_minus"""
    return [s_p - s_m for s_p, s_m in zip(S_plus(X, K_plus, N), S_minus(X, K_minus, M))]

# -----------------------------------------------------------------------------
#   Функция построения матрицы A (вектор сноса) уравнения Фоккера-Планка
# -----------------------------------------------------------------------------

def drift_vector(X, K_plus, N, M):
    """Вектор сноса"""
    res = sp.zeros(rows=len(X), cols=1)
    R = np.int8(N.T - M.T)
    R = sp.Matrix(R)
    for i in range(len(K_plus)):
        res += R[:, i] * S_plus(X, K_plus, M)[i]
    return res

# -----------------------------------------------------------------------------
#   Функция построения матрицы В (матрица диффузии) уравнения Фоккера-Планка
# -----------------------------------------------------------------------------

def diffusion_matrix(X, K_plus, N, M):
    """Матрица диффузии"""
    res = sp.zeros(rows=len(X), cols=len(X))
    R = np.int8(N.T - M.T)
    R = sp.Matrix(R)
    for i in range(len(K_plus)):
        res += R[:, i] * R[:, i].T * S_plus(X, K_plus, M)[i]
    return res

# -----------------------------------------------------------------------------
#   Функции преобразования А и В для численных вычислений
# -----------------------------------------------------------------------------

def func_for_rk(x, p, f):
    F = sp.lambdify(args=(*x, *p), expr=f, modules='numpy')
    def func_f(t, x, p):
        return F(*x, *p).flatten()
    return func_f

def func_for_sdu(x, p, g):
    G = sp.lambdify(args=(*x, *p), expr=g, modules='numpy')
    def func_g(x, p):
        return G(*x, *p)#.flatten()
    return func_g

# -----------------------------------------------------------------------------
#   Функции преобразования А и В для численных вычислений в СДУ
# -----------------------------------------------------------------------------

def matr_B(X, k, param, func):
    def G(x):
        GG = func_for_sdu(X, k, func)
        return GG(x, param)
    return G

def matr_A(X, k, param, func):
    def F(x):
        FF = func_for_rk(X, k, func)
        return FF(0, x, param)
    return F