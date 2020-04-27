import math
import numpy as np
from scipy.stats import t,f
import random as r
import prettytable as p

class LabSix():
    def __init__(self, m, p):
        self.m = m
        self.P = p
        self.k = 3
        self.X1_range = [15, 45]
        self.X2_range = [15, 50]
        self.X3_range = [15, 30]
        self.X_ranges = [self.X1_range, self.X2_range, self.X3_range]
        self.X0i = [self.get_x0(i) for i in self.X_ranges]
        self.detXi = [self.get_detX(self.X0i[0], self.X1_range[1]), self.get_detX(self.X0i[1], self.X2_range[1]), self.get_detX(self.X0i[2], self.X3_range[1])]
        self.Xcp_min = self.get_average([self.X1_range[0]+ self.X2_range[0]+self.X3_range[0]])
        self.Xcp_max = self.get_average([self.X1_range[1]+ self.X2_range[1]+self.X3_range[1]])
        self.l = 1.73
        self.X1_norm = [-1, -1, -1, -1, 1, 1, 1, 1, -self.l, self.l, 0, 0, 0, 0, 0]
        self.X2_norm = [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -self.l, self.l, 0, 0, 0]
        self.X3_norm = [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -self.l, self.l, 0]
        self.X_norm = [self.X1_norm, self.X2_norm, self.X3_norm]
        self.N = len(self.X1_norm)
        self.X1_abs = self.get_abs_values(self.X1_norm, 1)
        self.X2_abs = self.get_abs_values(self.X2_norm, 2)
        self.X3_abs = self.get_abs_values(self.X3_norm, 3)
        self.X_abs = [self.X1_abs, self.X2_abs, self.X3_abs]
        self.add_efects(self.X_norm)
        self.add_efects(self.X_abs)
        self.X_types = X_types = ["X1", "X2", "X3", "X12", "X13", "X23", "X123", "X1^2", "X2^2", "X3^2"]
    

    def get_abs_values(self, X_norm, n):
        n = n - 1
        X_abs = []
        for i in X_norm:
            if(i == 1):
                X_abs.append(self.X_ranges[n][1])
            elif(i == -1):
                X_abs.append(self.X_ranges[n][0])
            elif(i == 0):
                X_abs.append(self.X0i[n])
            elif(i == self.l):
                X_abs.append(round(self.l*self.detXi[n] + self.X0i[n], 3))
            else:
                X_abs.append(round(-self.l*self.detXi[n] + self.X0i[n], 3))
        return X_abs

    def get_x0(self, X_range):
        return (X_range[0]+X_range[1])/2
    
    def get_detX(self, x_max, x0):
        return x_max - x0

    def get_average(self, y):
        return sum(y)/len(y)

    def make_experiment(self):
        self.Y_exp = [[self.varint_func(self.X1_abs[i], self.X2_abs[i], self.X3_abs[i]) for _ in range(self.m)] for i in range(self.N)]

        

    def create_and_print_table(self, X_list, print_that):
        table = p.PrettyTable()
        for i in range(len(X_list)):
            table.add_column(self.X_types[i], X_list[i])
        for i in range(self.m):
            table.add_column("Y{0}".format(i+1), [j[i] for j in self.Y_exp])
        print(print_that)
        print(table)
        return table

    def add_efects(self, X_list):
        x1 = X_list[0]
        x2 = X_list[1]
        x3 = X_list[2]
        x12 = [round(x1[i]*x2[i], 3) for i in range(self.N)]
        x13 = [round(x1[i]*x3[i], 3) for i in range(self.N)]
        x23 = [round(x2[i]*x3[i], 3) for i in range(self.N)]
        x123 = [round(x1[i]*x2[i]*x3[i], 3) for i in range(self.N)]
        x1pow = [round(i**2, 3) for i in x1]
        x2pow = [round(i**2, 3) for i in x2]
        x3pow = [round(i**2, 3) for i in x3]
        X_list.append(x12)
        X_list.append(x13) 
        X_list.append(x23)
        X_list.append(x123)
        X_list.append(x1pow)
        X_list.append(x2pow)
        X_list.append(x3pow)
        

    def set_y_aver(self):
        self.y_aver = [self.get_average(i) for i in self.Y_exp]
        

    def find_coefs(self):
        x0 = [1]*15
        x1 = self.X_abs[0]
        x2 = self.X_abs[1]
        x3 = self.X_abs[2]
        x12 = self.X_abs[3]
        x13 = self.X_abs[4]
        x23 = self.X_abs[5]
        x123 = self.X_abs[6]
        x1pow = self.X_abs[7]
        x2pow = self.X_abs[8]
        x3pow = self.X_abs[9]
        self.lines_of_matr = list(zip(x0, x1, x2, x3, x12, x13, x23, x123, x1pow, x2pow, x3pow))
        self.b = np.linalg.lstsq(self.lines_of_matr, self.y_aver, rcond=None)[0]
        self.b = [round(i, 3) for i in self.b]
        print("Апроксимуюча функція:")
        print("y = {0} + {1}*x1 + {2}*x2 + {3}*x3 + {4}*x1*x2 + {5}*x1*x3 + {6}*x2*x3 + {7}*x1*x2*x3 + {8}*x1*x1 + {9}*x2*x2 + {10}*x3*x3".format(*self.b))
        print("Задана функція:")
        print("y = 3.5 + 6.6*x1 + 3.9*x2+ 1.8*x3 + 6.0*x1*x2+ 0.8*x1*x3+ 9.4*x2*x3 + 3.0*x1*x2*x3 + 5.3*x1*x1 + 0.5*x2*x2 + 4.3*x3*x3")
    
    
    def varint_func(self, x1, x2, x3):
        return 3.5+6.6*x1+3.9*x2+1.8*x3+5.3*x1*x1+0.5*x2*x2+4.3*x3*x3+6.0*x1*x2+0.8*x1*x3+9.4*x2*x3+3.0*x1*x2*x3 + r.randrange(0, 10) - 5

    def get_func_value(self, position):
        return sum([self.b[i]*self.lines_of_matr[position][i] for i in range(len(self.b))])

    def check_func_values(self):
        func_values = [round(self.get_func_value(i), 3) for i in range(15)]
        for i in range(self.N):
            print("y{0} = {1} ≈ {2}".format(i+1, round(func_values[i], 3), round(self.y_aver[i], 3)))
        return func_values

    def get_cohren_critical(self, prob, f1, f2):
        f_crit = f.isf((1 - prob) / f2, f1, (f2 - 1) * f1)
        return f_crit / (f_crit + f2 - 1)

    def cohren_check(self):
        self.f1 = self.m - 1
        self.f2 = self.N
        self.y_disp = [self.get_dispersion(self.y_aver[i], self.Y_exp[i]) for i in range(self.N)]
        Gp = max(self.y_disp)/sum(self.y_disp)
        Gt = self.get_cohren_critical(self.P, self.f1, self.f2)
        self.separator()
        if(Gp < Gt):
            print("Дисперсії однорідні")
            return True
        else:
            print("Дисперсії не однорідні, m+=1")
            self.m += 1
            self.Y_exp[0].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[1].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[2].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[3].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[4].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[5].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[6].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[7].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[8].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[9].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[10].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[11].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[12].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[13].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            self.Y_exp[14].append(r.randint(math.floor(self.Y_min), math.floor(self.Y_max)))
            return False

    def get_dispersion(self, y_aver, y):
        return sum([(i-y_aver)**2 for i in y])/len(y)        
    
    def get_fisher_teor(self, q, f3, f4):
       return f.isf(q, f4, f3)

    def get_student_teor(self, f3, q):
        return t.ppf(q / 2, f3)

    def before_chochren_steps(self):
        self.norm_table = self.create_and_print_table(self.X_norm, "Нормалізована матриця")
        self.abs_table = self.create_and_print_table(self.X_abs, "Абсолютна матриця")
        self.set_y_aver()
        return self.cohren_check()

    def student_crit(self):
        self.d = 0
        S2B = sum(self.y_disp)/self.N
        S2b = S2B/(self.N*self.m)
        Sb = math.sqrt(S2b)
        betai = np.zeros(len(self.lines_of_matr[0]))
        betai[0] = sum(self.y_aver)
        for i in range(len(self.X_norm)):
            betai[i+1] = sum([self.X_norm[i][j]*self.y_aver[j] for j in range(self.N)])
        ts = []
        for b in betai:
            ts.append(b/Sb)
        self.f3 = self.f1*self.f2
        Stud_teor = self.get_student_teor(self.f3, 1 - self.P)
        for i in range(len(ts)):
            if ts[i] < Stud_teor:
                self.b[i] = 0
            else:
                self.d += 1
        print("Кількість значимих коефіцієнтів:", self.d)
        print("Апроксимуюча функція після перевірки значимості коефіцієнтів:")
        print("y = {0} + {1}*x1 + {2}*x2 + {3}*x3 + {4}*x1*x2 + {5}*x1*x3 + {6}*x2*x3 + {7}*x1*x2*x3 + {8}*x1*x1 + {9}*x2*x2 + {10}*x3*x3".format(*self.b))
        print("Після перевірки значимості коефіцієнтів")
        self.y_new = self.check_func_values()
        

    def fisher_critical(self):
        self.f4 = self.N - self.d
        S2ad = abs((self.m/self.f4)*sum([(self.y_new[i] - self.y_aver[i])**2 for i in range(self.N)]))
        Sa = sum(self.y_disp)/self.N
        Fp = S2ad/Sa
        Fkr = self.get_fisher_teor(1 - self.P, self.f3, self.f4)
        if(Fkr > Fp):
            print("Рівняння адекватне оригіналу")
            print("Fkr =", Fkr, "> Fp =", Fp)
            self.is_adekvat = True
        else:
            print("Рівняння неадекватне оригіналу")
            print("Fkr =", Fkr, "< Fp =", Fp, "\nПочинаємо спочатку")
            self.separator()
            
    def separator(self):
        print("\n"+"-"*188)

    def run(self):
        self.is_adekvat = False
        while(not self.is_adekvat):
            self.make_experiment()
            while not self.before_chochren_steps():
                pass
            self.find_coefs()
            self.check_func_values()
            self.separator()
            self.student_crit()
            self.fisher_critical()
            
        
labSix = LabSix(3, 0.95)
labSix.run()
