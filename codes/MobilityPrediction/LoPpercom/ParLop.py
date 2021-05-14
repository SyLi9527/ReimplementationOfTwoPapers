import numpy as np
import scipy
from sympy import Symbol, solve
from scipy import optimize
# import sympy
# from sympy.parsing.sympy_parser import parse_expr
# t = sympy.symbols('t')
# parsed = parse_expr("log2(t) + t - 5",
#                     local_dict={"log2": lambda x: sympy.log(x, 2)})
# print(sympy.solve(parsed, t)[0] + 1)


# def myfunc(x, s=0, n=0):
#     # return x*np.log2(x)+(1-x)*np.log2(1-x) + s + n + 0.9
#     return (-(x*np.log2(x)+(1-x)*np.log2(1-x)) +
#             (1-x)*np.log2(n-1)) - s - 0.9


def getLimit(s, n):
    result = []
    for x0 in np.arange(0.05, 1, 0.005):
        result.append(scipy.optimize.fsolve(
            lambda x:  (-(x*np.log2(x)+(1-x)*np.log2(1-x)) + (1-x)*np.log2(n-1)) - s - 0.9, x0=x0)[0])

    return np.max(result)


print(getLimit(0, 0))


def rtn(S, N):
    res = []
    for i in range(len(S)):
        try:
            if S[i] > np.log2(N[i]):
                res.append(-99)
            else:
                res.append(getLimit(S[i], N[i]))
        except ValueError as e:
            res.append(-88)
            print("log2(0) causes ValueError")
            raise
    return res


# def getLimit(S, N):
#     x = Symbol('x')
#     res = solve((-(x*np.log2(x)+(1-x)*np.log2(1-x)) +
#                 (1-x)*np.log2(N-1)) - S - 0.9, x)[0]
#     # res = solve(S, N, "(-(x*log2(x)+(1-x)*log2(1-x))+(1-x)*log2(N-1))-S)=0.9",
#     #             var='x', para1='S', para2='N')

#     return res


def solve(S, N, equation, var='x', para1='S', para2='N'):
    equation = equation.replace("=", "-(")+")"
    result = eval(equation, {var: 1j, para1: S, para2: N})
    return -result.real/result.imag
