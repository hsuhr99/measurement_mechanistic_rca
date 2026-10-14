import re

from scipy.io import arff
import pandas as pd
import numpy as np
import os
import pickle


def convert_arff_csv(filepath):
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    df['class'] = df['class'].astype(np.float32)
    return df


def extract_infos(filepath):
    subspaces = []
    infos = []
    indexes = []

    with open(filepath) as f:
        lines = f.readlines()

    for l in lines:
        w = l.split()
        if 'subspace' in w:
            ndims = int(w[4])
            sub = np.empty(ndims)
            for i in range(len(sub)):
                sub[i] = int(re.sub('[\[,\]]', "", w[5+i]))
            subspaces.append(sub)
        if 'pattern' in w:
            indexes.append(int(w[1]))
            subs = []
            for e in w[3]:
                subs.append(int(e))
            subs.reverse()
            infos.append(subs)

    infos_df = pd.DataFrame(np.where(np.array(infos) == 1, True, False), columns=np.arange(len(infos[0])))
    infos_df['index'] = indexes

    return subspaces, infos_df

if __name__=='__main__':
    path = 'D:/Dottorato/MaskingModelExplainer/datasets/synthetic/'
    files = os.listdir(path)

    for f in files:
        if f.endswith(".arff"):
            data = convert_arff_csv(path+f)
            data.to_csv(path+'processed/'+f.replace('.arff', '.csv'), index=False)
            print(f'File {f} processed!')
        elif f.endswith(".info"):
            subspaces, info_df = extract_infos(path+f)
            pickle.dump(subspaces, open(path+'processed/info_'+f.replace('.info', '.joblib'), 'wb'))
            info_df.to_csv(path+'processed/info_' + f.replace('.info', '.csv'), index=False)
            print(f'File {f} processed!')
        else:
            print(f'File {f} skipped!')
