import sys
exp_name=sys.argv[1]
exp_type=sys.argv[2]
print(exp_name)


acc0 = []
with open('eval/'+exp_name+'_id_0-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if 'valid_acc' in line:
            line = line.split()
            acc0.append(line[-1])

acc1 = []
with open('eval/'+exp_name+'_id_1-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if 'valid_acc' in line:
            line = line.split()
            acc1.append(line[-1])

acc2 = []
with open('eval/'+exp_name+'_id_2-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if 'valid_acc' in line:
            line = line.split()
            acc2.append(line[-1])

acc3 = []
with open('eval/'+exp_name+'_id_3-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if 'valid_acc' in line:
            line = line.split()
            acc3.append(line[-1])


acc4 = []
with open('eval/'+exp_name+'_id_4-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if 'valid_acc' in line:
            line = line.split()
            acc4.append(line[-1])

acc5 = []
with open('eval/'+exp_name+'_id_5-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if 'valid_acc' in line:
            line = line.split()
            acc5.append(line[-1])

acc6 = []
with open('eval/'+exp_name+'_id_6-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if 'valid_acc' in line:
            line = line.split()
            acc6.append(line[-1])

acc7 = []
if exp_type == 'ws':
    with open('eval/'+exp_name+'_id_7-2-cutout-16-1.0-auxiliary-0.4/log.txt') as f:
        for line in f.readlines():
            line = line.strip()
            if 'valid_acc' in line:
                line = line.split()
                acc7.append(line[-1])
else:
    acc7 = 600*[-1.0]


global acc
import numpy as np

acc = [acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7]
best_acc = np.array([acc0[-1], acc1[-1], acc2[-1], acc3[-1], acc4[-1], acc5[-1], acc6[-1], acc7[-1]])

def avg_of_list(l):
  total = 0
  for val in l:
      total = total + float(val)
  return total/len(l)

def moving_avg_of_list(l):
    avg = 0
    for id, val in enumerate(l):
        avg = avg * id * 0.1 + float(val) * 0.9
        avg = avg / (id+1)
    return avg

def max_of_list(l):
  total = 0
  for val in l:
      if float(val) >= total:
          total = float(val)
  return total

def avg_of_acc(start, end, type='avg'):
    acc_avg = np.zeros(8)
    if type == 'avg':
        for i in range(8):
            acc_avg[i] = avg_of_list(acc[i][start:end])
    elif type == 'moving_avg':
        for i in range(8):
            acc_avg[i] = moving_avg_of_list(acc[i][start:end])
    elif type == 'max':
        for i in range(8):
            acc_avg[i] = max_of_list(acc[i][start:end])
    return acc_avg

# first phase
acc_enc = np.zeros(8)
acc_avg = avg_of_acc(0, 30, type='moving_avg')
top_index = np.argpartition(acc_avg, -4)[-4:]
acc_enc[top_index] = 1
best_top_index = np.argpartition(best_acc, -4)[-4:]
print('phase 1 real top: ', best_acc[best_top_index])
print('phase 1: ', acc_avg[top_index], top_index, best_acc[top_index])

# second phase
acc_enc = np.zeros(8)
acc_enc[top_index] = 1
acc_avg = avg_of_acc(31, 100, type='moving_avg')
acc_avg[acc_enc==0] = 0
# print(acc_avg)
top_index = np.argpartition(acc_avg, -2)[-2:]
best_top_index = np.argpartition(best_acc, -2)[-2:]
print('phase 2 real top: ', best_acc[best_top_index])
print('phase 2: ', acc_avg[top_index], best_acc[top_index])

# # third phase
# acc_enc = np.zeros(8)
# acc_enc[top_index] = 1
# acc_avg = avg_of_acc(100, 150, type='moving_avg')
# acc_avg[acc_enc==0] = 0
# # print(acc_avg)
# top_index = np.argpartition(acc_avg, -2)[-2:]
# best_top_index = np.argpartition(best_acc, -2)[-2:]
# print('phase 3 real top: ', best_acc[best_top_index])
# print('phase 3: ', acc_avg[top_index], best_acc[top_index])




