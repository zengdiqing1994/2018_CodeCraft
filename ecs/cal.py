import sys
data=[]
data_temp=[]
class Vm:
    def __init__(self):
        self.num = 0
        self.type = 'flavor'
        self.core = 2
        self.mem = 5
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


core = 0
mem = 0
def read():
    i = 0
    f = open('E:/1/text.txt', 'r')
    for line in f:
        core = 0
        mem = 0
        temp = line.split('\n')
        data.append(temp[0])
        data_temp.append(data[i].split(' '))
        for k in range(len(data_temp[i])):
            if k % 2 == 0:
                core += flavor_list_temp[data_temp[i][k]].core * int(data_temp[i][k + 1])
                mem += (flavor_list_temp[data_temp[i][k]].mem * int(data_temp[i][k + 1])/1024)
        if core > 56 or mem >128:
            print str(core)+' '+str(mem)
        else:
            print str(0)
        i += 1

read()

