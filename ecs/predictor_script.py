# coding=utf-8
import math
import copy
import random
import datetime

class Vm:
    def __init__(self):
        self.num = 0
        self.type = 'flavor'
        self.core = 2
        self.mem = 5


class Ser:
    def __init__(self):
        self.core = 25
        self.mem = 50
        self.packing_type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Vm_Type = {0:16 , 1:65536 , 2:16 , 3: 32768, 4: 16, 5: 16384, 6: 8, 7: 32768, 8: 8, 9: 16384, 10: 8, 11: 8192, 12: 4, 13: 16384, 14: 4, 15: 8192, 16: 4, 17:4096 , 18:2  , 19:8192 , 20:2  , 21:4096 , 22:2  , 23:2048 , 24:1  , 25:4096 , 26:1  , 27:2048 , 28:1  , 29:1024  }
Vm_list = []


vm_flavor15 = Vm()
vm_flavor15.core = 16
vm_flavor15.mem = 65536
vm_flavor15.type = 15

vm_flavor14 = Vm()
vm_flavor14.core = 16
vm_flavor14.mem = 32768
vm_flavor14.type = 14

vm_flavor13 = Vm()
vm_flavor13.core = 16
vm_flavor13.mem = 16384
vm_flavor13.type = 13

vm_flavor12 = Vm()
vm_flavor12.core = 8
vm_flavor12.mem = 32768
vm_flavor12.type = 12

vm_flavor11 = Vm()
vm_flavor11.core = 8
vm_flavor11.mem = 16384
vm_flavor11.type = 11

vm_flavor10 = Vm()
vm_flavor10.core = 8
vm_flavor10.mem = 8192
vm_flavor10.type = 10

vm_flavor9 = Vm()
vm_flavor9.core = 4
vm_flavor9.mem = 16384
vm_flavor9.type = 9

vm_flavor8 = Vm()
vm_flavor8.core = 4
vm_flavor8.mem = 8192
vm_flavor8.type = 8

vm_flavor7 = Vm()
vm_flavor7.core = 4
vm_flavor7.mem = 4096
vm_flavor7.type = 7

vm_flavor6 = Vm()
vm_flavor6.core = 2
vm_flavor6.mem = 8192
vm_flavor6.type = 6

vm_flavor5 = Vm()
vm_flavor5.core = 2
vm_flavor5.mem = 4096
vm_flavor5.type = 5


vm_flavor4 = Vm()
vm_flavor4.core = 2
vm_flavor4.mem = 2048
vm_flavor4.type = 4

vm_flavor3 = Vm()
vm_flavor3.core = 1
vm_flavor3.mem = 4096
vm_flavor3.type = 3

vm_flavor2 = Vm()
vm_flavor2.core = 1
vm_flavor2.mem = 2048
vm_flavor2.type = 2

vm_flavor1 = Vm()
vm_flavor1.core = 1
vm_flavor1.mem = 1024
vm_flavor1.type = 1

vm_list_temp_flavor = ['flavor15', 'flavor14', 'flavor13', 'flavor12', 'flavor11', 'flavor10', 'flavor9', 'flavor8',
                       'flavor7', 'flavor6', 'flavor5', 'flavor4', 'flavor3', 'flavor2', 'flavor1']
flavor_list_temp = {'flavor15': vm_flavor15, 'flavor14': vm_flavor14, 'flavor13': vm_flavor13, 'flavor12': vm_flavor12,
                    'flavor11': vm_flavor11, 'flavor10': vm_flavor10, 'flavor9': vm_flavor9, 'flavor8': vm_flavor8,
                    'flavor7': vm_flavor7, 'flavor6': vm_flavor6, 'flavor5': vm_flavor5, 'flavor4': vm_flavor4,
                    'flavor3': vm_flavor3, 'flavor2': vm_flavor2, 'flavor1': vm_flavor1}
def ExSmooth(x):
    sx3 = []
    sx2 = []
    sx1 = []
    N = len(x)
    EX_alpha = 0.322
    sx1.append(copy.deepcopy(x[0][1]))
    sx2.append(copy.deepcopy(sx1[0]))
    sx3.append(copy.deepcopy(sx2[0]))
    para = [0, 0, 0]
    for i in range(1, N):
        sx1.append(0)
        sx1[i] = EX_alpha * x[i-1][1] + (1 - EX_alpha) * sx1[i-1]

    for i in range(1, N):
        sx2.append(0)
        sx2[i] = EX_alpha * sx1[i-1] + (1 - EX_alpha) * sx2[i-1]
    a = 2 * sx1[N-1] - sx2[N-1]
    b = EX_alpha/(1-EX_alpha)*(sx1[N-1] - sx2[N - 1])
    para[0] = a
    para[1] = b
    return para
    # for i in range(1, N):
    # sx3.append(0)
    #     sx3[i] = EX_alpha * sx2[i-1] + (1-EX_alpha)*sx3[i-1]
    # beta = EX_alpha/(2*(1-EX_alpha) * (1 - EX_alpha))
    # a = 3 * sx1[N-1] - 3 * sx2[N-1] + sx3[N-1]
    # b = beta * ((6-5 * EX_alpha) * sx1[N-1] - 2 * (5 - 4 * EX_alpha) * sx2[N - 1] + (4 - 3 * EX_alpha) * sx3[N-1])
    # c = beta * EX_alpha * (sx1[N-1] - 2*sx2[N-1] + sx3[N-1])
    # para[0] = a
    # para[1] = b
    # para[2] = c
    # #
    # return para

def ExSmooth_1(x,h):
    alpha = 0.321
    beta = 0.1
    gama =0.1
    k = 14 # zhou qi
    s = []
    t = []
    p = []
    result=[]
    N = len(x)
    s.append(float(x[0][1]))
    t.append(float(x[1][1] - x[0][1]))
    p.append(0.0) # multipul p =1
    for i in range(N):
        # s.append(alpha * x[i][1] / p[i-k] + (1-alpha)*(s[i-1] + t[i-1]))
        # t.append((beta * (s[i] - s[i-1])) + (1-beta) * t[i-1])
        # p.append((gama * x[i] / s[i]) + ((1-gama) * p[i-k]))
        if i > k:
            s.append(alpha * (x[i][1] - p[i - k]) + (1 - alpha) * (s[i - 1] + t[i - 1]))
        else:
            s.append(alpha * (x[i][1] - p[i]) + (1 - alpha) * (s[i - 1] + t[i - 1]))
        t.append((beta * (s[i] - s[i - 1])) + (1 - beta) * t[i - 1])
        if i > k:
            p.append(gama * (x[i][1] - s[i]) + ((1 - gama) * p[i - k]))
        else:
            p.append(gama * (x[i][1] - s[i]) + ((1 - gama) * p[i]))
        # s.append(alpha * (x[i][1]) + (1 - alpha) * (s[i - 1] + t[i - 1]))
        # t.append((beta * (s[i] - s[i - 1])) + (1 - beta) * t[i - 1])
    result.append(s[N - 1])
    result.append(t[N - 1])
    result.append(p[N - 1 + h - k])
    return result


def matix_mul(matrix_1, matrix1_hang, matrix1_lie, matrix_2, matrix2_hang, matrix2_lie):
    if matrix1_lie != matrix2_hang:
        print 'matrix input error!'
        return 0
    else:
        result = []
        result_temp = []
        for i in range(0, matrix1_hang):
            for k in range(0, matrix2_lie):
                temp = 0
                for j in range(0, matrix1_lie):
                    temp += matrix_1[i][j] * matrix_2[j][k]
                result_temp.append(temp)
            result.append(result_temp)
            result_temp = []
    return result


def ws_first(x, w):
    result = 0
    for i in range(0, len(x)):
        w[i] = x[i] * w[i]
        result += w[i] * x[i]
    return 1.0 / result


def ws_second(x, w, y):
    result = 0
    for i in range(len(x)):
        w[i] = x[i] * w[i]
        result += y[i] * w[i]
    return result

def output(x,target,k):
    W = []
    for i in range(len(x)):
        temp = math.exp((-((x[i] - target) ** 2))/(2*(k*k)))
        W.append(temp)
    return W


# def weight(data, k):
#     i = 0
#     M = len(data)
#     N = len(data[0])
#     k = 0.01
#     while i < M:
#         for j in range(0, M):
#             W = math.exp(-((data[j][1] - data[i][1]) ** 2) / (2 * (k ** 2)))
#             j += 1
#         i
#     return W

def lwlr(testPoint,xArr,yArr,k=0.5):
    matrix_lenght = len(xArr)
    diffMat = []
    temp = []
    xTx = 0
    weight =[]
    for i in range(matrix_lenght):
        diffMat.append(testPoint - xArr[i])
        weight.append(math.exp((diffMat[i] * diffMat[i])/(-2.0 * k * k)))
    for i in range(matrix_lenght):
        temp.append(xArr[i] * weight[i])
    for i in range(matrix_lenght):
        xTx += xArr[i] * temp[i]
    temp = []
    temp_1 = 0
    for i in range(matrix_lenght):
        temp.append(weight[i] * yArr[i])
    for j in range(matrix_lenght):
        temp_1 += xArr[j] * temp[j]
    ws = (1.0/xTx) * temp_1
    return testPoint * ws



def f(w, x):
    N = len(w)
    i = 0
    y = 0
    while i < N - 1:
        y += w[i] * x[i]
        i += 1
    y += w[N - 1]
    return y


def gradient(data, w, j):
    M = len(data)
    N = len(data[0])
    i = 0
    g = 0
    while i < M:
        # Gassi core  w[i] = math.exp(-(data[i] - ))
        y = f(w, data[i])
        if (j != N - 1):
            g += (y - data[i][N - 1]) * data[i][j]
            # g += (data[i][N - 1] - y) * data[i][j]  #sum of the direction of gradient
        else:
            g += y - data[i][N - 1]
            # g += data[i][N - 1] - y
        i += 1
        g_temp = float(g)
        # if i == 39:
        #     i = i
    r = g_temp / M
    # r = g_temp
    return r


def isSame(a, b):
    n = len(a)
    i = 0
    while i < n:
        if abs(a[i] - b[i]) > 0.00001:
            return False
        i += 1
    return True


def fw(w, data, a):# lossfunction
    a = 0.1
    M = len(data)
    N = len(data[0])
    i = 0
    s = 0
    # W = output(x, target, tao)
    while i < M:
        e = data[i][N - 1] - f(w, data[i])
        s += (e ** 2)
        s = s + ((a/2) * (w[0] ** 2))
        i += 1
    return s / 2


def numberProduct(n, vec, w):  # downgradient algorithm
    N = len(vec)
    i = 0
    while i < N:
        w[i] -= vec[i] * n
        i += 1


def assign(a):
    L = []
    for x in a:
        L.append(x)
    return L


# a = b
def assign2(a, b):
    i = 0
    while i < len(a):
        a[i] = b[i]
        i += 1


def dotProduct(a, b):
    N = len(a)
    i = 0
    dp = 0
    while i < N:
        dp += a[i] * b[i]
        i += 1
    return dp


def calcAlpha(w, g, a, data):
    c1 = 0.3
    now = fw(w, data,a)
    wNext = assign(w)
    numberProduct(a, g, wNext)
    next = fw(wNext, data,a)
    count = 30
    while next < now:
        a *= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fw(wNext, data,a)
        count -= 1
        if count == 0:
            break

    count = 50
    while next > now - c1 * a * dotProduct(g, g):
        a /= 2
        wNext = assign(w)
        numberProduct(a, g, wNext)
        next = fw(wNext, data,a)

        count -= 1
        if count == 0:
            break
    return a


def normalize(g):
    s = 0
    for x in g:
        s += x * x
    s = math.sqrt(s)
    i = 0
    N = len(g)
    while i < N:
        g[i] /= s
        i += 1


def calcCoefficient(data, listA, listW, listLostFunction,a):
    N = len(data[0])
    w = [0 for i in range(N)]
    wNew = [0 for i in range(N)]
    g = [0 for i in range(N)]

    times = 0
    alpha = 10.0
    while times < 100:
        j = 0
        while j < N:
            g[j] = gradient(data, w, j)
            j += 1
        #normalize(g)
        alpha = calcAlpha(w, g, alpha, data)
        numberProduct(alpha, g, wNew)

        # print ("times,alpha,fw,w,g:\t"), times, alpha, fw(w, data), w, g
        if isSame(w, wNew):
            break
        assign2(w, wNew)
        times += 1

        listA.append(alpha)
        listW.append(assign(w))
        listLostFunction.append(fw(w, data, a))

    return w

def ff_calculation_cpu(random_list, serv_core, serv_mem, Vm_Type_num):
    serv_all_num = 1
    serv_num_temp = 1
    # packing_flag = 0
    j = 0
    serv_dict = {}
    serv_dict[101] = serv_core
    serv_dict[102] = serv_mem
    for zz in range(len(Vm_Type_num)):
        serv_dict[102 + Vm_Type_num[zz].type] = 0
    serv_dict["last_space"] = serv_core
    for i in random_list:
        serv_num_temp = 1
        for k in range(1, serv_all_num + 1):
            if (Vm_list[i].core > serv_dict[k * 100 + 1] or Vm_list[i].mem > serv_dict[k * 100 + 2]):
                serv_num_temp += 1
                if (serv_num_temp > serv_all_num):
                    serv_all_num += 1
                    serv_dict[serv_all_num * 100 + 1] = serv_core
                    serv_dict[serv_all_num * 100 + 2] = serv_mem
                    for zz in range(len(Vm_Type_num)):
                        serv_dict[serv_all_num * 100 + Vm_Type_num[zz].type + 2] = 0
                    serv_dict[serv_all_num * 100 + 1] = serv_dict[serv_all_num * 100 + 1] - Vm_list[i].core
                    serv_dict[serv_all_num * 100 + 2] = serv_dict[serv_all_num * 100 + 2] - Vm_list[i].mem
                    serv_dict[serv_all_num * 100 + 2 + Vm_list[i].type] = serv_dict[serv_all_num * 100 + 2 + Vm_list[
                        i].type] + 1
                    break
            else:
                serv_dict[k * 100 + 1] = serv_dict[k * 100 + 1] - Vm_list[i].core
                serv_dict[k * 100 + 2] = serv_dict[k * 100 + 2] - Vm_list[i].mem
                serv_dict[k * 100 + 2 + Vm_list[i].type] = serv_dict[k * 100 + 2 + Vm_list[i].type] + 1
                break
    serv_dict["all_num_serv"] = serv_all_num
    # for i in Vm_Type_num:
    # serv_dict["last_space"] = serv_dict["last_space"] - serv_dict[serv_dict["all_num_serv"] * 100 + i.type + 2] * Vm_Type[2 * (i.type - 1)]
    serv_dict["last_space"] = serv_dict[serv_dict["all_num_serv"] * 100 + 1]
    return serv_dict

def ff_calculation_mem(random_list, serv_core, serv_mem, Vm_Type_num):
    serv_all_num = 1
    serv_num_temp = 1
    # packing_flag = 0
    j = 0
    serv_dict = {}
    serv_dict[101] = serv_core
    serv_dict[102] = serv_mem
    for zz in range(len(Vm_Type_num)):
        serv_dict[102 + Vm_Type_num[zz].type] = 0
    serv_dict["last_space"] = serv_mem
    for i in random_list:
        serv_num_temp = 1
        for k in range(1, serv_all_num + 1):
            if (Vm_list[i].core > serv_dict[k * 100 + 1] or Vm_list[i].mem > serv_dict[k * 100 + 2]):
                serv_num_temp += 1
                if (serv_num_temp > serv_all_num):
                    serv_all_num += 1
                    serv_dict[serv_all_num * 100 + 1] = serv_core
                    serv_dict[serv_all_num * 100 + 2] = serv_mem
                    for zz in range(len(Vm_Type_num)):
                        serv_dict[serv_all_num * 100 + Vm_Type_num[zz].type + 2] = 0
                    serv_dict[serv_all_num * 100 + 1] = serv_dict[serv_all_num * 100 + 1] - Vm_list[i].core
                    serv_dict[serv_all_num * 100 + 2] = serv_dict[serv_all_num * 100 + 2] - Vm_list[i].mem
                    serv_dict[serv_all_num * 100 + 2 + Vm_list[i].type] = serv_dict[serv_all_num * 100 + 2 + Vm_list[
                        i].type] + 1
                    break
            else:
                serv_dict[k * 100 + 1] = serv_dict[k * 100 + 1] - Vm_list[i].core
                serv_dict[k * 100 + 2] = serv_dict[k * 100 + 2] - Vm_list[i].mem
                serv_dict[k * 100 + 2 + Vm_list[i].type] = serv_dict[k * 100 + 2 + Vm_list[i].type] + 1
                break
    serv_dict["all_num_serv"] = serv_all_num
    # for i in Vm_Type_num:
    #     serv_dict["last_space"] = serv_dict["last_space"] - serv_dict[serv_dict["all_num_serv"] * 100 + i.type + 2] * Vm_Type[2 * (i.type-1) + 1]
    serv_dict["last_space"] = serv_dict[serv_dict["all_num_serv"] * 100 + 2]
    return serv_dict


def packing_list(ff_temp,Vm_Type_num):
    list_temp = []
    for k in Vm_Type_num:
        for i in range(1, ff_temp['all_num_serv'] + 1):
            for j in range(0, ff_temp[i * 100 + k.type + 2]):
                list_temp.append(i)
    return list_temp


def list_to_matrix(Vm_num, list_matrix_temp, cal_list):
    next_list = []
    list_temp = [0]
    for j in range(2, len(list_matrix_temp) + 1):
        list_matrix_temp[j] += list_matrix_temp[j - 1]
        list_temp.append(list_matrix_temp[j - 1])
    for i in range(0, Vm_num):
        list_temp[cal_list[i] - 1] += 1
        next_list.append(list_temp[cal_list[i] - 1] - 1)
    return next_list


def pso_cpu(M, serv_core, serv_mem, Vm_Type_num):
    Vm_num = M
    Vm_temp = [[], [], [], [], [], [], [], [], []]
    Vm_Matrix = [[], [], [], [], [], [], [], [], [], []]
    Vm_Matrix_list = [[], [], [], [], [], [], [], [], [], []]
    P_best = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    P_best_list = [[], [], [], [], [], [], [], [], [], []]
    calculation_list = [[], [], [], [], [], [], [], [], [], []]
    G_best_list = []
    P_best_temp = []
    #    ff_calculate_extra_space = 9999
    #    p_best_extra_space = 9999
    G_best = []
    w = 0.8
    c1 = 7
    c2 = 9
    ff_temp = {}
    list_temp = []
    calculation_temp = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    Type_sum_list = {}
    min_serv_num = 0
    first_list = []
    count = 0
    for i in range(0, M):
        first_list.append(count)
        count += 1
    ff_temp = copy.deepcopy(ff_calculation_cpu(first_list, serv_core, serv_mem, Vm_Type_num))
    G_best = copy.deepcopy(ff_temp)
    list_temp.append(copy.deepcopy(packing_list(ff_temp,Vm_Type_num)))
    P_best[0] = copy.deepcopy(ff_temp)
    Vm_Matrix[0] = copy.deepcopy(first_list)
    min_serv_num = P_best[0]['all_num_serv']
    for i in range(1, 10):
        #    P_best_temp.append(ff_temp[]) ff_calculation(first_list, serv_core, serv_mem)[Vm_num + 1]
        #    for j in range(1, Vm_num + 1):
        Vm_temp[i - 1] = [k for k in range(0, M)]
        random.shuffle(Vm_temp[i - 1])
        Vm_Matrix[i] = Vm_temp[i - 1]
        P_best[i] = copy.deepcopy(ff_calculation_cpu(Vm_Matrix[i], serv_core, serv_mem,Vm_Type_num))
        list_temp.append(copy.deepcopy(packing_list(P_best[i], Vm_Type_num)))

    for iter in range(300):
        for i in range(0, 10):
            calculation_temp[i] = (copy.deepcopy(ff_calculation_cpu(Vm_Matrix[i], serv_core, serv_mem, Vm_Type_num)))
            if (P_best[i]['all_num_serv'] > calculation_temp[i]['all_num_serv']) or (
                        (P_best[i]['all_num_serv'] == calculation_temp[i]['all_num_serv']) and (
                                P_best[i]['last_space'] < calculation_temp[i]['last_space'])):
                P_best[i] = copy.deepcopy(calculation_temp[i])

            if i > 0:
                if P_best[i]['all_num_serv'] < P_best[i - 1]['all_num_serv']:
                    min_serv_num = P_best[i]['all_num_serv']

        for i in range(0, 10):
            if (G_best['all_num_serv'] > P_best[i]['all_num_serv']) or (
                        (G_best['all_num_serv'] == P_best[i]['all_num_serv']) and (
                        G_best['last_space'] < P_best[i]['last_space'])):
                G_best = copy.deepcopy(P_best[i])

        G_best_list = copy.deepcopy(packing_list(G_best,Vm_Type_num))

        for i in range(10):
            P_best_list[i] = copy.deepcopy(packing_list(P_best[i],Vm_Type_num))
            calculation_list[i] = copy.deepcopy(packing_list(calculation_temp[i],Vm_Type_num))
            for k in range(1, min_serv_num + 1):
                Type_sum_list[k] = 0
            for j in range(0, Vm_num):
                r1 = 1 + copy.deepcopy((random.random() % G_best['all_num_serv']))
                r2 = 1 + copy.deepcopy((random.random() % G_best['all_num_serv']))
                pre_temp = copy.deepcopy(w * calculation_list[i][j] + c1 * r1 * (
                    P_best_list[i][j] - calculation_list[i][j]) + c2 * r2 * (
                                             G_best_list[j] - calculation_list[i][j]))
                calculation_list[i][j] = copy.deepcopy(int(pre_temp) % G_best['all_num_serv'] + 1)
                if calculation_list[i][j]:
                    Type_sum_list[calculation_list[i][j]] += 1
            Vm_Matrix[i] = list_to_matrix(Vm_num, Type_sum_list, calculation_list[i])


    return G_best

def pso_mem(M, serv_core, serv_mem, Vm_Type_num):
    Vm_num = M
    Vm_temp = [[], [], [], [], [], [], [], [], []]
    Vm_Matrix = [[], [], [], [], [], [], [], [], [], []]
    Vm_Matrix_list = [[], [], [], [], [], [], [], [], [], []]
    P_best = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    P_best_list = [[], [], [], [], [], [], [], [], [], []]
    calculation_list = [[], [], [], [], [], [], [], [], [], []]
    G_best_list = []
    P_best_temp = []
    #    ff_calculate_extra_space = 9999
    #    p_best_extra_space = 9999
    G_best = []
    w = 0.8
    c1 = 7
    c2 = 9
    ff_temp = {}
    list_temp = []
    calculation_temp = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    Type_sum_list = {}
    min_serv_num = 0
    first_list = []
    count = 0
    for i in range(0,M):
        first_list.append(count)
        count += 1
    ff_temp = copy.deepcopy(ff_calculation_mem(first_list, serv_core, serv_mem, Vm_Type_num))
    G_best = copy.deepcopy(ff_temp)
    list_temp.append(copy.deepcopy(packing_list(ff_temp,Vm_Type_num)))
    P_best[0] = copy.deepcopy(ff_temp)
    Vm_Matrix[0] = copy.deepcopy(first_list)
    min_serv_num = P_best[0]['all_num_serv']
    for i in range(1, 10):
        #    P_best_temp.append(ff_temp[]) ff_calculation(first_list, serv_core, serv_mem)[Vm_num + 1]
        #    for j in range(1, Vm_num + 1):
        Vm_temp[i - 1] = [k for k in range(0, M)]
        random.shuffle(Vm_temp[i - 1])
        Vm_Matrix[i] = Vm_temp[i - 1]
        P_best[i] = copy.deepcopy(ff_calculation_mem(Vm_Matrix[i], serv_core, serv_mem,Vm_Type_num))
        list_temp.append(copy.deepcopy(packing_list(P_best[i], Vm_Type_num)))

    for iter in range(300):
        for i in range(0, 10):
            calculation_temp[i] = (copy.deepcopy(ff_calculation_mem(Vm_Matrix[i], serv_core, serv_mem, Vm_Type_num)))
            if (P_best[i]['all_num_serv'] > calculation_temp[i]['all_num_serv']) or (
                        (P_best[i]['all_num_serv'] == calculation_temp[i]['all_num_serv']) and (
                                P_best[i]['last_space'] < calculation_temp[i]['last_space'])):
                P_best[i] = copy.deepcopy(calculation_temp[i])

            if i > 0:
                if P_best[i]['all_num_serv'] < P_best[i - 1]['all_num_serv']:
                    min_serv_num = P_best[i]['all_num_serv']

        for i in range(0, 10):
            if (G_best['all_num_serv'] > P_best[i]['all_num_serv']) or (
                        (G_best['all_num_serv'] == P_best[i]['all_num_serv']) and (
                        G_best['last_space'] < P_best[i]['last_space'])):
                G_best = copy.deepcopy(P_best[i])

        G_best_list = copy.deepcopy(packing_list(G_best,Vm_Type_num))

        for i in range(10):
            P_best_list[i] = copy.deepcopy(packing_list(P_best[i],Vm_Type_num))
            calculation_list[i] = copy.deepcopy(packing_list(calculation_temp[i],Vm_Type_num))
            for k in range(1, min_serv_num + 1):
                Type_sum_list[k] = 0
            for j in range(0, Vm_num):
                r1 = 1 + copy.deepcopy((random.random() % G_best['all_num_serv']))
                r2 = 1 + copy.deepcopy((random.random() % G_best['all_num_serv']))
                pre_temp = copy.deepcopy(w * calculation_list[i][j] + c1 * r1 * (
                    P_best_list[i][j] - calculation_list[i][j]) + c2 * r2 * (
                                             G_best_list[j] - calculation_list[i][j]))
                calculation_list[i][j] = copy.deepcopy(int(pre_temp) % G_best['all_num_serv'] + 1)
                if calculation_list[i][j]:
                    Type_sum_list[calculation_list[i][j]] += 1
            Vm_Matrix[i] = list_to_matrix(Vm_num, Type_sum_list, calculation_list[i])


    return G_best

# analog backfire
def SA_cpu(M, serv_core, serv_mem, Vm_Type_num):
    Vm_num = M
    random_swap = [0, 0]
    T = 100.0
    Tmin = 1
    r = 0.9999
    Vm_Matrix = []
    temp_list = {}
    best = {}
    for i in range(0, M):
        Vm_Matrix.append(i)
    best = ff_calculation_cpu(Vm_Matrix, serv_core, serv_mem, Vm_Type_num)
   #jiang vm fang ru dice zhong
    while (T > Tmin):
    #jiao huan vm wei zhi
        random_swap[0] = random.randint(0, M-1)
        random_swap[1] = random.randint(0, M-1)
        temp = Vm_Matrix[random_swap[0]]
        Vm_Matrix[random_swap[0]] = Vm_Matrix[random_swap[1]]
        Vm_Matrix[random_swap[1]] = temp
        temp_list = ff_calculation_cpu(Vm_Matrix,serv_core, serv_mem,Vm_Type_num)
        if (best['all_num_serv'] > temp_list['all_num_serv']) or (
                (best['all_num_serv'] == temp_list['all_num_serv']) and (
                            best['last_space'] < temp_list['last_space'])):
            best = copy.deepcopy(temp_list)
        else:
            if math.exp((best['all_num_serv'] - temp_list['all_num_serv']) / T) > (random.randint(0, 100) / 100):
                best = temp_list

        T = r * T

    return best


def SA_mem(M, serv_core, serv_mem, Vm_Type_num):
    Vm_num = M
    random_swap = [0, 0]
    T = 100.0
    Tmin = 1
    r = 0.9999
    Vm_Matrix = []
    temp_list = {}
    best = {}
    for i in range(0, M):
        Vm_Matrix.append(i)
    best = ff_calculation_mem(Vm_Matrix, serv_core, serv_mem, Vm_Type_num)
    #sui ji xuan qu liang ge vm wei zhi jiao huan
#     for (int i = 0; i < vec_flavors.size(); i++) {
#     dice.push_back(i);
#
# }   #jiang vm fang ru dice zhong
    while (T > Tmin):
    #jiao huan vm wei zhi
        random_swap[0] = random.randint(0, M-1)
        random_swap[1] = random.randint(0, M-1)
        # new_vec_flavors = vec_flavors
        temp = Vm_Matrix[random_swap[0]]
        Vm_Matrix[random_swap[0]] = Vm_Matrix[random_swap[1]]
        Vm_Matrix[random_swap[1]] = temp
        temp_list = ff_calculation_mem(Vm_Matrix,serv_core, serv_mem,Vm_Type_num)
        if (best['all_num_serv'] > temp_list['all_num_serv']) or (
                (best['all_num_serv'] == temp_list['all_num_serv']) and (
                            best['last_space'] < temp_list['last_space'])):
            best = copy.deepcopy(temp_list)
        else:
            if math.exp((best['all_num_serv'] - temp_list['all_num_serv']) / T) > ((random.random() % 100) / 100):
                best = temp_list

        T = r * T

    return best


# www =   []

def ave(list):

    s = 0

    for x in list:
        s += x
        s = float(s)
    average = s / len(list)
    return average


def average(list):

    avg = 0
    avg = sum(list) / float(len(list) * 1.0)  # 调用sum函数求和
    return avg


def var(list, average):
    """利用list 和均值计算方差"""
    var1 = 0
    for i in list:
        var1 += float((i - average) ** 2 * 1.0)
    var2 = (math.sqrt(var1 / (len(list)) * 1.0))
    return var2



def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    pack_print = []
    train_first_date = 0
    serv_core = 0
    serv_mem = 0
    result = []
    i = 0
    Vm_Type ={}
    Vm_num = 0
    temp_1 = []
    data =[]
    consider = 'MEM'
    Vm_Type_list =[]
    Vm_date_num = {}
    predict_Vm = {}
    date = 1
    Vm_date_num_count = 0
    a = []
    if ecs_lines is None:
        print ('ecs information is none')
        return result
    if input_lines is None:
        print ('input file information is none')
        return result

    for line in input_lines:
        print ("index of input data")
        i += 1
        if i == 1:
            input_values = line.split(" ")
            temp = input_values[2].split('\n')
            input_values[2] = temp[0]
            serv_core = int(input_values[0])
            serv_mem  = int(input_values[1])
            serv_mem  = serv_mem * 1024
            serv_disk = int(input_values[2])
            serv_disk = serv_disk * 1024
        #
        else:
            if i == 3:
                input_values = line.split("\n")
                Vm_num = int(input_values[0])
        #
            else:
                if (3 < i) and (i < (4 + Vm_num)):
                    input_values = line.split(" ")
                    temp = input_values[2].split('\n')
                    input_values[2] = temp[0]
                    Vm_Type[input_values[0]] = (int(input_values[1]), int(input_values[2]))
                    Vm_Type_list.append(input_values[0])
                    flavor_list_temp[input_values[0]].core = int(input_values[1])
                    flavor_list_temp[input_values[0]].mem = int(input_values[2])
                else:
                    if i == 3 + Vm_num + 2:
                        input_values = line.split("\n")
                        consider = input_values[0]
                    else:
                        if 6 + Vm_num < i < 9 + Vm_num:
                            input_values = line.split(' ')
                            temp_1 =(input_values[0].split('-'))
                            start_year  = int(temp_1[0])
                            start_month = int(temp_1[1])
                            start_day   = int(temp_1[2])
                            a.append(datetime.datetime(start_year, start_month, start_day))

    predict_day = (a[1]-a[0]).days
    # predict_day_start_to_begin = (a[0]-).days
    for i in Vm_Type_list:
        Vm_date_num[i] = [[1, 0]]
        predict_Vm[i] = 0

    month_flag = 0
    day_temp  = 1
    month_day = 0
    All_Time = 0
    month_list = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    for line in ecs_lines:
        values = line.split("\t")
        temp = values[2].split('\n')
        values[2] = temp[0]
        uuid = values[0]
        temp = values[2].split(' ')
        temp = temp[0]
        temp = temp.split('-')
        train_first_date = datetime.datetime(int(temp[0]),int(temp[1]),int(temp[2]))
        break
    predict_day_start_to_begin = []
    predict_day_start_to_end = []
    predict_day_start_to_begin.append((a[0] - train_first_date).days)
    predict_day_start_to_end.append((a[1] - train_first_date).days)


    Vm_nozero = {}
    Vm_nozero2day = {}
    for line in ecs_lines:
        values = line.split("\t")
        temp = values[2].split('\n')
        values[2] = temp[0]
        uuid = values[0]
        temp = values[2].split(' ')
        temp = temp[0]
        temp = temp.split('-')
        createTime_month = int(temp[1])
        train_date_pt = datetime.datetime(int(temp[0]),int(temp[1]),int(temp[2]))
        # month_day = month_list[createTime_month]
        createTime = (train_date_pt - train_first_date).days + 1
        # if createTime < day_temp:
        #      month_flag = 1
        #      month_day = month_list[createTime_month - 1]
        # if month_flag:
        #     createTime += month_day
        # day_temp = copy.deepcopy(int(temp[2]))

        flavorName = values[1]

        if flavorName in Vm_Type:
            if createTime == date:
                Vm_date_num[flavorName][Vm_date_num_count][1] += 1  #create vm on the same day
            else:
                for i in Vm_date_num:
                    date = createTime
                    Vm_date_num[i].append([date, 0])
                Vm_date_num_count += 1
                Vm_date_num[flavorName][Vm_date_num_count][1] += 1  #create vm on the differ day
            # if Vm_date_num[flavorName][Vm_date_num_count][1] > 0:
            #
            #     Vm_nozero[flavorName].append(Vm_date_num[flavorName][Vm_date_num_count][1])
        All_Time = createTime
    # Vm_nozero = []
    # Vm_nozero2day = {}
    nozero2day_temp = []


    Grubbs_99 = [1.155, 1.492, 1.749, 1.944, 2.097, 2.22, 2.323, 2.410, 2.485, 2.550, 2.607, 2.659, 2.705, 2.747, 2.785, 2.821, 2.954, 2.884, 2.912, 2.939, 2.963, 2.987, 3.009, 3.029, 3.049, 3.068, 3.085, 3.103, 3.119, 3.135, 3.150, 3.164, 3.178, 3.191, 3.204, 3.216, 3.228, 3.240, 3.251, 3.261, 3.271, 3.282, 3.292, 3.302, 3.310, 3.319, 3.329, 3.336, 3.345, 3.353, 3.361, 3.388, 3.376, 3.383, 3.391, 3.397, 3.405, 3.411, 3.418, 3.424, 3.430, 3.437, 3.442, 3.449, 3.454, 3.460, 3.466, 3.471, 3.476, 3.482, 3.487, 3.492, 3.496, 3.502, 3.507, 3.511, 3.516, 3.521, 3.525, 3.529, 3.534, 3.539, 3.543, 3.547, 3.551, 3.555, 3.559, 3.563, 3.567, 3.570, 3.575, 3.579, 3.582, 3.586, 3.589, 3.593, 3.597, 3.600]
    Grubbs_90 = [1.148, 1.425, 1.602, 1.729, 1.828, 1.909, 1.977, 2.036, 2.088, 2.134, 2.175, 2.213, 2.247, 2.279, 2.309, 2.335, 2.361, 2.385, 2.408, 2.429, 2.448, 2.467, 2.486, 2.502, 2.519, 2.534, 2.549, 2.583, 2.577, 2.591, 2.604, 2.616, 2.628, 2.639, 2.650, 2.661, 2.671, 2.682, 2.692, 2.700, 2.710, 2.719, 2.727, 2.736, 2.744, 2.753, 2.760, 2.768, 2.775, 2.783, 2.790, 2.798, 2.804, 2.811, 2.818, 2.824, 2.831, 2.837, 2.842, 2.849, 2.854, 2.860, 2.866, 2.871, 2.877, 2.883, 2.888, 2.893, 2.897, 2.903, 2.908, 2.912, 2.917, 2.922, 2.927, 2.931, 2.935, 2.940, 2.945, 2.949, 2.953, 2.957, 2.961, 2.966, 2.970, 2.973, 2.977, 2.981, 2.984, 2.989, 2.993, 2.996, 3.000, 3.003, 3.006, 3.011, 3.014, 3.017]
    for Vm_type_count in Vm_Type_list:
        Vm_nozero = []
        Vm_nozero2day = {}
        for d in range(len(Vm_date_num[Vm_type_count])):
            if Vm_date_num[Vm_type_count][d][1] > 0 :
                Vm_nozero.append(Vm_date_num[Vm_type_count][d][1])
                if Vm_date_num[Vm_type_count][d][1] in Vm_nozero2day:
                    Vm_nozero2day[Vm_date_num[Vm_type_count][d][1]].append(d)
                else:
                    Vm_nozero2day[Vm_date_num[Vm_type_count][d][1]] = []
                    Vm_nozero2day[Vm_date_num[Vm_type_count][d][1]].append(d)

        pingjun = ave(Vm_nozero) #平均值
        biaozhuncha = var(Vm_nozero,pingjun) #标准差
        if biaozhuncha > 0:
            Vm_nozero = sorted(Vm_nozero)
            for i in range(len(Vm_nozero)):
                ave_min = pingjun - Vm_nozero[0]
                max_ave = Vm_nozero[-1] - pingjun
                if max_ave > ave_min:
                    G = max_ave/biaozhuncha
                    if G > Grubbs_99[len(Vm_nozero)-3]:
                        # for iiii in range(len(Vm_nozero2day[Vm_nozero[-1]])):
                            Vm_date_num[Vm_type_count][Vm_nozero2day[Vm_nozero[-1]][0]][1] = math.ceil(pingjun)
                            Vm_nozero.pop()
                    else:
                            break
                else:
                    G = ave_min / biaozhuncha
                    if G > Grubbs_99[len(Vm_nozero)-3]:
                        Vm_date_num[Vm_type_count][Vm_nozero2day[Vm_nozero[0]][0]][1] = math.ceil(pingjun)
                        Vm_nozero.pop(0)
                    else:
                        break


        k = 0
        # data.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # if Vm_date_num[Vm_type_count][0][0] == 1:
        #     if Vm_date_num[Vm_type_count][0][1]:
        #         data[0][0] = Vm_date_num[Vm_type_count][0][1]
        #
        # for i in range(1,10):
        #     if Vm_date_num[Vm_type_count][i][0] < 10:
        #         if Vm_date_num[Vm_type_count][i][1]:
        #             data[0][i] = Vm_date_num[Vm_type_count][i][1] + data[0][i-1]
        #     else:
        #         data[0][i] += data[0][i-1]
        #     k += 1

        k = 0
        predict_y = 0
        Vm_total  = 0
        # for i in range(1, All_Time - 8):
        #     data.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        #     # if i == 35:
        #     #     i=i
        #     for j in range(9):
        #         data[i][j] = data[i-1][j+1]
        #     while Vm_date_num[Vm_type_count][k][0] < i + 6 and k <= All_Time - 1:
        #         k += 1
        #     if Vm_date_num[Vm_type_count][k][0] == i + 6:
        #         data[i][9] = Vm_date_num[Vm_type_count][k][1] + data[i][8]
        #     else:
        #         data[i][9] += data[i][8]
        #     # data[i][9]

        # 2 dimension
        data.append([0, 0])
        data[0][0] = 1
        #x.append(data[0][0])
        if (Vm_date_num[Vm_type_count][0][0] == 1) and (Vm_date_num[Vm_type_count][0][1]):
            data[0][1] = Vm_date_num[Vm_type_count][0][1]
        #y.append(data[0][1])

        for i in range(1, All_Time):
        #for i in range(1,len(predict_day_start_to_end)):
            #x.append(i+1)
            data.append([0, 0])
            data[i][0] += i+1
            while Vm_date_num[Vm_type_count][k][0] < data[i][0] and k <= All_Time - 1:
                k += 1
            if Vm_date_num[Vm_type_count][k][0] == data[i][0] and (Vm_date_num[Vm_type_count][k][1]):

                data[i][1] = Vm_date_num[Vm_type_count][k][1] + data[i-1][1]
            else:
                data[i][1] += data[i-1][1]
            #y.append(data[i][1])
        k = 0

        predict_y = 0
        #W = []
        # for i in range(predict_day):
        # listA = []
        # listW = []
        # listLostFunction = []
        # w = calcCoefficient(data, listA, listW, listLostFunction,a)


        # data.append([0,0,0,0,0,0,0,0,0,0])
        # # data.append(copy.deepcopy(data[All_Time-9]))
        #
        # for i in range(0,predict_day):
        #     predict_y = f(w, data[All_Time - 8])
        #     predict_y = abs(math.ceil(predict_y))
        #     predict_y = int(predict_y)
        #     data[All_Time - 8].pop(0)
        #     data[All_Time - 8].append(predict_y)
        #
        # predict_Vm[Vm_type_count] = data[All_Time - 8][9] - data[All_Time - 9][9]
        # predict_Vm[Vm_type_count] += 1
        # predict_y = f(w, predict_day_start_to_end)
        # predict_x = f(w, predict_day_start_to_begin)
        # predict_Vm[Vm_type_count] = int(math.ceil(predict_y - predict_x))
        #data = []

        # # ExSmooth
        parameter = ExSmooth(data)
        predict_y = parameter[0] + parameter[1] * predict_day #+ parameter[2] * predict_day * predict_day
        predict_y = int(math.ceil(predict_y) - data[len(data) - 1][1])

        # # ExSmooth_1
        # parameter = ExSmooth_1(data,predict_day)
        # predict_y = (parameter[0] + parameter[1] * predict_day) + parameter[2]
        # predict_y = int(math.ceil(predict_y) - data[len(data) - 1][1])

        # for i in range(predict_day):
        #     W = output(x, All_Time + i + 1, tao)
        #     result1 = ws_first(x, W)
        #     result2 = ws_second(x, W, y)
        #
        #     predict_y = result1 * result2 * (All_Time + i + 1)
        # predict_y = f(w, x)
        # predict_y = int(math.ceil(predict_y))
        #     x.append(x[-1] + 1)
        #      y.append(predict_y)

        # xarr_temp = []
        # yarr_temp = []
        #
        # xarr = []
        # yarr = []
        # for iiiii in range(len(data)):
        #     xarr.append(data[iiiii][0])
        #     xarr_temp.append(data[iiiii][0])
        #     yarr_temp.append(data[iiiii][1])
        #     yarr.append(data[iiiii][1])
        # for kkkkk in range(predict_day):
        #     predict_y = math.ceil(lwlr(All_Time + kkkkk, xarr_temp, yarr_temp, 1.0))
        #     yarr_temp.append(predict_y)
        #     xarr_temp.append(xarr_temp[-1] + 1)
        # predict_y = int(math.ceil(predict_y) - data[len(data) - 1][1])

        if predict_y <= 0:
            predict_y = 1
        predict_Vm[Vm_type_count] = int(predict_y)
        data = []
    Vm_total = 0
    for j in Vm_Type_list:
        if j in predict_Vm:
            for kk in range(predict_Vm[j]):
                Vm_total += 1
    #     Vm_total += predict_Vm[Vm_type_count]
    pack_print.append(str(Vm_total))

    for j in range(0, len(Vm_Type_list)):
        output_predict_line = str(Vm_Type_list[j]) + ' ' + str(predict_Vm[Vm_Type_list[j]])
        pack_print.append(output_predict_line)

        #pack_print.append('\n')

    all_num_vm = 0
    Vm_Type_num1 = []
    #
    for ii in vm_list_temp_flavor:
        if ii in predict_Vm:
            all_num_vm += predict_Vm[ii]
            for kk in range(predict_Vm[ii]):
                Vm_list.append(copy.deepcopy(flavor_list_temp[ii]))
            if predict_Vm[ii] != 0:
                Vm_Type_num1.append(copy.deepcopy(flavor_list_temp[ii]))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor15))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor14))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor13))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor12))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor11))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor10))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor9))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor8))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor7))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor6))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor5))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor4))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor3))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor2))
    # for iii in range(55):
    #     Vm_list.append(copy.deepcopy(vm_flavor1))
    #
    #
    # all_num_vm = 825
    # Vm_Type_num1.append(vm_flavor15)
    # Vm_Type_num1.append(vm_flavor14)
    # Vm_Type_num1.append(vm_flavor13)
    # Vm_Type_num1.append(vm_flavor12)
    # Vm_Type_num1.append(vm_flavor11)
    # Vm_Type_num1.append(vm_flavor10)
    # Vm_Type_num1.append(vm_flavor9)
    # Vm_Type_num1.append(vm_flavor8)
    # Vm_Type_num1.append(vm_flavor7)
    # Vm_Type_num1.append(vm_flavor6)
    # Vm_Type_num1.append(vm_flavor5)
    # Vm_Type_num1.append(vm_flavor4)
    # Vm_Type_num1.append(vm_flavor3)
    # Vm_Type_num1.append(vm_flavor2)
    # Vm_Type_num1.append(vm_flavor1)
    # consider = 'MEM'

    # if consider == 'CPU':
    #     o = SA_cpu(all_num_vm, serv_core, serv_mem, Vm_Type_num1)
    # else:
    #     o = SA_mem(all_num_vm, serv_core, serv_mem, Vm_Type_num1)


    if consider == 'CPU':
        o = pso_cpu(all_num_vm, serv_core, serv_mem, Vm_Type_num1)
    else:
        o = pso_mem(all_num_vm, serv_core, serv_mem, Vm_Type_num1)

    pack_print.append('')
    pack_print.append(str(o['all_num_serv']))
    for i in range(1, o['all_num_serv']+1):
        output_packing_line = str(i)
        for k in range(0, len(Vm_Type_num1)):
            # if k < (len(Vm_Type_num1)-1):
            if ((i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type) in o) and (o[i * 100 + 2 + Vm_Type_num1[
                        len(Vm_Type_num1) - k - 1].type] > 0):
                output_packing_line += ' ' + str(vm_list_temp_flavor[15 - Vm_Type_num1[len(Vm_Type_num1)  - 1 - k].type]) + ' ' + str(o[ i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type])
            # else:
            #     output_packing_line += str(
            #         vm_list_temp_flavor[15 - Vm_Type_num1[len(Vm_Type_num1) - 1 - k].type]) + ' ' + str(
            #         o[i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type])
        pack_print.append(output_packing_line)
    #


    # pack_print.append(str(all_num_vm))
    # for i in range(0, all_num_vm):
    #     output_packing_line = str(i + 1)
    #     output_packing_line += ' ' + str(vm_list_temp_flavor[15 - Vm_list[i].type]) + ' 1'
            # if k < (len(Vm_Type_num1)-1):
            # if ((i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type) in o) and (o[i * 100 + 2 + Vm_Type_num1[
            #             len(Vm_Type_num1) - k - 1].type] > 0):
            #     output_packing_line += ' ' + str(vm_list_temp_flavor[15 - Vm_Type_num1[len(Vm_Type_num1)  - 1 - k].type]) + ' ' + str(o[ i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type])
            #  else:
            #     output_packing_line += str(
            #         vm_list_temp_flavor[15 - Vm_Type_num1[len(Vm_Type_num1) - 1 - k].type]) + ' ' + str(
            #         o[i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type])
        # pack_print.append(output_packing_line)


    return pack_print
    # return pack_print1







