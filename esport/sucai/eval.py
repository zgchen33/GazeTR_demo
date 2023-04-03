from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

tarzan_xy = []
LP_xy = []
scout_xy = []
hang_xy = []
leave_xy = []
jiejie_xy = []
ale_xy = []
meiko_xy = []
fofo_xy = []

folder_path = os.getcwd()
cnt =0
for file_name in tqdm(os.listdir(folder_path)):

    if file_name.endswith('.json'):

        with open(os.path.join(folder_path,file_name),'r') as f:
            cnt = cnt+1
            json_data = json.load(f)
            labels = json_data['shapes']
            flag = np.zeros(9)
            for item in labels:
                xy = list(itertools.chain(*item['points']))
                if item['label']=='tarzan':
                    tarzan_xy.append(xy)
                    flag[0] += 1
                elif item['label']=='scout':
                    scout_xy.append(xy)
                    flag[1] += 1
                elif item['label']=='hang':
                    hang_xy.append(xy)
                    flag[2] += 1
                elif item['label']=='LP':
                    LP_xy.append(xy)
                    flag[3] += 1
                elif item['label']=='leave':
                    leave_xy.append(xy)
                    flag[4] += 1
                elif item['label']=='jiejie':
                    jiejie_xy.append(xy)
                    flag[5] += 1
                elif item['label']=='ale':
                    ale_xy.append(xy)
                    flag[6] += 1
                elif item['label']=='meiko':
                    meiko_xy.append(xy)
                    flag[7] += 1
                elif item['label']=='fofo':
                    fofo_xy.append(xy)
                    flag[8] += 1

            if flag[0]==0:
                tarzan_xy.append([-1,-1])
            if flag[1]==0:
                scout_xy.append([-1,-1])
            if flag[2]==0:
                hang_xy.append([-1,-1])
            if flag[3]==0:
                LP_xy.append([-1,-1])
            if flag[4]==0:
                    leave_xy.append([-1,-1])
            if flag[5]==0:
                    jiejie_xy.append([-1,-1])
            if flag[6]==0:
                    ale_xy.append([-1,-1])
            if flag[7]==0:
                    meiko_xy.append([-1,-1])
            if flag[8]==0:
                    fofo_xy.append([-1,-1])

            if len(meiko_xy)!=cnt:
                print(cnt)

print(cnt)
tarzan_xy=np.array(tarzan_xy)
print(tarzan_xy.shape)
np.save('tarzan.npy',tarzan_xy)

scout_xy=np.array(scout_xy)
np.save('scout.npy',scout_xy)
print(scout_xy.shape)
hang_xy=np.array(hang_xy)
np.save('hang.npy',hang_xy)
print(hang_xy.shape)
LP_xy=np.array(LP_xy)
np.save('LP.npy',LP_xy)
print(LP_xy.shape)
leave_xy=np.array(leave_xy)
np.save('leave.npy',leave_xy)
print(leave_xy.shape)
jiejie_xy=np.array(jiejie_xy)
np.save('jiejie.npy',jiejie_xy)
print(jiejie_xy.shape)
ale_xy=np.array(ale_xy)
np.save('ale.npy',ale_xy)
print(ale_xy.shape)
meiko_xy=np.array(meiko_xy)
np.save('meiko.npy',meiko_xy)
print(meiko_xy.shape)
fofo_xy=np.array(fofo_xy)
np.save('fofo.npy',fofo_xy)
print(fofo_xy.shape)