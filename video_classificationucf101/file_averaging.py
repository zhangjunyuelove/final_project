# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




import os
from collections import defaultdict, Counter
import pickle
import pandas as pd

SUBMIT_PATH = ''
SIGFIGS = 6

def read_models(model_weights, blend=None):
    
    
    if not blend:
        blend = defaultdict(Counter)
    for m, w in model_weights.items():
        print(m, w)
        with open(os.path.join(SUBMIT_PATH, m + '.csv'), 'r') as f:
            f.readline()
            for l in f:
                id, r ,label= l.split(',')
                id, r ,label= id, r.split(' '), int(label)
                
                n = len(r) // 2
                for i in range(0, n, 2):
                    k = int(r[i])
                    v = int(10**(SIGFIGS - 1) * float(r[i+1]))
                    blend[id][k] += w * v
    return blend


def write_models(blend, file_name, total_weight):
    with open(os.path.join(SUBMIT_PATH, file_name + '.csv'), 'w') as f:
        f.write('VideoID,LabelConfidencePairs\n')
        for id, v in blend.items():
            l = ' '.join(['{} {:{}f}'.format(t[0]
                                            , float(t[1]) / 10 ** (SIGFIGS - 1) / total_weight
                                            , SIGFIGS) for t in v.most_common(1)])
            f.write(','.join([str(id), l + '\n']))
    return None

def find_label(file_name):
    label_list=[]
    with open(os.path.join(SUBMIT_PATH, file_name + '.csv'), 'r') as csv_reader:
        csv_reader.readline()
        
        for l in csv_reader:
            id,r,label= l.split(',')
            id, r,label = id, r.split(' '),int(label)
            label_list.append(label)
    return label_list



def calculate_accuracy(list1,file_name):
    with open(os.path.join(SUBMIT_PATH, file_name + '.csv'), 'r') as csv_reader:
        csv_reader.readline()

        count_true=0
        count_all=0
        for l in csv_reader:
            id,r= l.split(',')
            id, r = id, r.split(' ')
            if int(r[0])==list1[count_all]:
                count_true=count_true+1

            count_all=count_all+1
        accuracy= count_true/count_all
        print(accuracy)



model_pred = { 
              

 
              'E:/Moe_Junyue/predictions_vlad':1,
              'E:/Moe_Junyue/predictions_moe':1,
              'E:/Moe_Junyue/predictions_3moe':1,
              'E:/Moe_Junyue/predictions_test_overall':1,
              'E:/Moe_Junyue/predictions_fv':1,
              'E:/Moe_Junyue/predictions_newmoe2':1,

                   
                 }

avg = read_models(model_pred)
list1=find_label('E:/Moe_Junyue/predictions_vlad')

write_models(avg, 'Junyue_submission', sum(model_pred.values()))
calculate_accuracy(list1,'Junyue_submission')
