import numpy as np
from sympy import *
import sympy as sp
from IPython.display import display, Latex, Math
""" construction interaction scheme """

class PopModel:
 
    def __init__(self, value):
        """ value — количество популяций """
        self.value = value     
        self.M = np.zeros(self.value)
        self.N = np.zeros(self.value)
        self.eq_nub = 0
        self.coef= []

# -----------------------------------------------------------------------------
#   Функция для определиния типа взаимодействия
# -----------------------------------------------------------------------------

    def interaction(self,type_int):

        # Естественное размножение популяции
        if type_int == 1:
            return [1,0,2,0]
        # Естественное гибель популяции
        if type_int == 2:
            return [1,0,0,0]
        # симбиоз 
        if type_int == 3:
            return [1,1,2,1]
        # хищник-жертва
        if type_int == 4:
            return [1,1,0,2]
        # внутривидовая конкуренция
        if type_int == 5:
            return [2,0,1,0]
        # межвидовая конкуренция
        if type_int == 6:
            return [1,1,1,0]
        # миграция
        if type_int == 7:
            return [1,0,0,1]
        
# -----------------------------------------------------------------------------
#   Вспомогательная функция для определиния определения коэффициентов если они не заданы
# -----------------------------------------------------------------------------

    def coef(self):
        k=[]
        
# -----------------------------------------------------------------------------
#   Функция для добавления уравнений в схему взаимодействия
# -----------------------------------------------------------------------------
        
    def adder(self,type_int,id_pop_1,id_pop_2,coef=0):
        """
        type_int --- тип взаимодействия (1 — естественное размножение популяции; 2 — естественное гибель популяции;
        3 — симбиоз; 4 — хищник-жертва; 5 — внутривидовая конкуренция; 6 — межвидовая конкуренция; 7 — миграция),
        id_pop_1 --- идентификатор 1-ой взаимодейтствующей популяции,
        id_pop_2 --- идентификатор 2-ой взаимодейтствующей популяции,
        coef --- коэффициенты взаимодействия (по умолчанию k_i).
        """
        self.eq_nub=self.eq_nub+1
        if coef==0:
            self.coef.append('k_{0}'.format(self.eq_nub))
        else:
            self.coef.append(coef)
        if np.sum(self.M)==0:
            if id_pop_2!=0:
                self.N[id_pop_2-1]=self.interaction(type_int)[3]
                self.M[id_pop_2-1]=self.interaction(type_int)[1]
            self.M[id_pop_1-1]=self.interaction(type_int)[0]
            self.N[id_pop_1-1]=self.interaction(type_int)[2]
            
        else:
            M_i = np.zeros(self.value)
            N_i = np.zeros(self.value)
            if id_pop_2!=0:
                M_i[id_pop_2-1]=self.interaction(type_int)[1]
                N_i[id_pop_2-1]=self.interaction(type_int)[3]
            M_i[id_pop_1-1]=self.interaction(type_int)[0]
            N_i[id_pop_1-1]=self.interaction(type_int)[2]
            
               
            self.M = np.vstack([self.M,M_i])
            self.N = np.vstack([self.N,N_i])

# -----------------------------------------------------------------------------
#   Функция для построения матрицы M (состояние до взаимодействия)
# -----------------------------------------------------------------------------

    def matr_M(self):
        return(self.M)

# -----------------------------------------------------------------------------
#   Функция для построения матрицы N (состояние после взаимодействия)
# -----------------------------------------------------------------------------

    def matr_N(self):
        return(self.N)    

# -----------------------------------------------------------------------------
#   Функция отображения схемы взаимодейтсвия
# -----------------------------------------------------------------------------

    def display_infos(self,model,X):
        coef=sp.Matrix(model.coef)
        M=X.T*sp.Matrix(np.int8(model.matr_M()).T)
        N=X.T*sp.Matrix(np.int8(model.matr_N()).T)
        for i in range(len(M)):
            result = "$${} ".format(latex(M[i]))+ "=" + "[{}]".format(latex(coef[i])) + "\Rightarrow"    +" {}$$".format(latex(N[i]))
            display(Latex(result))