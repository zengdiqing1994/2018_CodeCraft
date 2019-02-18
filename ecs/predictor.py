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

        else:
            if i == 3:
                input_values = line.split("\n")
                Vm_num = int(input_values[0])

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

        All_Time = createTime

    nozero2day_temp = []

    Grubbs_97_5 = [1.155, 1.481, 1.715, 1.887, 2.020, 2.126, 2.215, 2.290, 2.355, 2.412, 2.462, 2.507, 2.549, 2.585, 2.620, 2.651, 2.681, 2.709, 2.733, 2.758, 2.781, 2.802, 2.822, 2.841, 2.859, 2.876, 2.893, 2.908, 2.924, 2.938, 2.952, 2.965, 2.979, 2.991, 3.003, 3.014, 3.025, 3.036, 3.046, 3.057, 3.067, 3.075, 3.085, 3.094, 3.103, 3.111, 3.120, 3.128, 3.136, 3.143, 3.151, 3.158, 3.166, 3.172, 3.180, 3.186, 3.193, 3.199, 3.205, 3.212, 3.218, 3.224, 3.230, 3.235, 3.241, 3.246, 3.252, 3.257, 3.262, 3.267, 3.272, 3.278, 3.282, 3.287, 3.291, 3.297, 3.301, 3.305, 3.309, 3.315, 3.319, 3.323, 3.327, 3.331, 3.335, 3.339, 3.343, 3.347, 3.350, 3.355, 3.358, 3.362, 3.365, 3.369, 3.372, 3.377, 3.380, 3.383]
    Grubbs_99_5 = [1.155, 1.496, 1.764, 1.973, 2.139, 2.274, 2.387, 2.482, 2.564, 2.636, 2.699, 2.755, 2.806, 2.852, 2.894, 2.932, 2.968, 3.001, 3.031, 3.060, 3.087, 3.112, 3.135, 3.157, 3.178, 3.199, 3.218, 3.236, 3.253, 3.270, 3.286, 3.301, 3.316, 3.330, 3.343, 3.356, 3.369, 3.381, 3.393, 3.404, 3.415, 3.425, 3.435, 3.445, 3.455, 3.464, 3.474, 3.483, 3.491, 3.500, 3.507, 3.516, 3.524, 3.531, 3.539, 3.546, 3.553, 3.560, 3.566, 3.573, 3.579, 3.586, 3.592, 3.598, 3.605, 3.610, 3.617, 3.622, 3.627, 3.633, 3.638, 3.643, 3.648, 3.654, 3.658, 3.663, 3.669, 3.673, 3.677, 3.682, 3.687, 3.691, 3.695, 3.699, 3.704, 3.708, 3.712, 3.716, 3.720, 3.725, 3.728, 3.732, 3.736, 3.739, 3.744, 3.747, 3.750, 3.754]
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
        predict_y = 0
        Vm_total  = 0

        # 2 dimension
        data.append([0, 0])
        data[0][0] = 1
        if (Vm_date_num[Vm_type_count][0][0] == 1) and (Vm_date_num[Vm_type_count][0][1]):
            data[0][1] = Vm_date_num[Vm_type_count][0][1]

        for i in range(1, All_Time):
            data.append([0, 0])
            data[i][0] += i+1
            while Vm_date_num[Vm_type_count][k][0] < data[i][0] and k <= All_Time - 1:
                k += 1
            if Vm_date_num[Vm_type_count][k][0] == data[i][0] and (Vm_date_num[Vm_type_count][k][1]):

                data[i][1] = Vm_date_num[Vm_type_count][k][1] + data[i-1][1]
            else:
                data[i][1] += data[i-1][1]

        k = 0
        predict_y = 0

        # ExSmooth
        parameter = ExSmooth(data)
        predict_y = parameter[0] + parameter[1] * predict_day #+ parameter[2] * predict_day * predict_day
        predict_y = int(math.ceil(predict_y) - data[len(data) - 1][1])


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

    if consider == 'CPU':
        o = pso_cpu(all_num_vm, serv_core, serv_mem, Vm_Type_num1)
    else:
        o = pso_mem(all_num_vm, serv_core, serv_mem, Vm_Type_num1)

    pack_print.append('')
    pack_print.append(str(o['all_num_serv']))
    for i in range(1, o['all_num_serv']+1):
        output_packing_line = str(i)
        for k in range(0, len(Vm_Type_num1)):
            if ((i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type) in o) and (o[i * 100 + 2 + Vm_Type_num1[
                        len(Vm_Type_num1) - k - 1].type] > 0):
                output_packing_line += ' ' + str(vm_list_temp_flavor[15 - Vm_Type_num1[len(Vm_Type_num1)  - 1 - k].type]) + ' ' + str(o[ i * 100 + 2 + Vm_Type_num1[len(Vm_Type_num1) - k - 1].type])

        pack_print.append(output_packing_line)

    return pack_print








