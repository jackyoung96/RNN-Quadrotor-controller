import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvirtualdisplay import Display

from sklearn.manifold import TSNE

files = [file for file in os.listdir('evaluation') if 'hiddens.pkl' in file]

with Display(visible=False, size=(100, 60)) as disp:
    for file in files:
        env_name, seed, num_eval, rand_ratio, *_ = file.strip('.pkl').split("_")
        df = pd.read_pickle(os.path.join('evaluation', file))
        # print("\n%s evaluation result (seed: %s, num_eval: %s, random ratio: %s)"%(env_name, seed, num_eval, rand_ratio))
        columns = ['algo','iter','step','hidden','d_hidden','params']
        algo_list = ['td3lstm_single','td3lstm_random','td3lstm_multitask']

        # print(df.apply(lambda row: np.linalg.norm(row['d_hidden']), axis=1))
        df['d_hidden'] = df.apply(lambda row: np.abs(row['d_hidden']).mean(), axis=1)

        for algo in algo_list:
            df_algo = df.loc[df['algo']==algo]
            sns.lineplot(data=df_algo, x='step', y='d_hidden')
            folder = 'figs/%s'%file.strip('_hiddens.pkl')
            if not os.path.isdir(folder):
                os.makedirs(folder)
            plt.savefig('%s/%s_%s_dhidden.png'%(folder, env_name, algo))
            plt.close()

            sns.lineplot(data=df_algo.loc[df_algo['iter']==0], x='step', y='d_hidden')
            plt.savefig('%s/%s_%s_dhidden_only1.png'%(folder, env_name, algo))
            plt.close()


            # TSNE part
            # Only 10 iterations
            n = 10
            df_iter = df_algo.loc[df_algo['iter']<n]
            for i in range(n):
                params = df_algo.loc[df_algo['iter']==i]['params'].head(1).item()
                num_comp = len(params)
                print("class%d :"%i, params)
            df_iter['hidden'] = df_iter.apply(lambda row: row['hidden'].squeeze().tolist(), axis=1)
            x = df_iter['hidden'].tolist()
            y = df_iter['iter'].tolist()
            tsne = TSNE(n_components=num_comp, random_state=123)
            z = tsne.fit_transform(x)

            df_tsne = pd.DataFrame()
            df_tsne["y"] = y
            df_tsne["comp-1"] = z[:,0]
            df_tsne["comp-2"] = z[:,1]
            df_tsne["comp-3"] = z[:,2]

            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            sns.scatterplot(ax=axes[0],x="comp-1", y="comp-2", hue=df_tsne.y.tolist(),
                            palette=sns.color_palette("hls", n),
                            data=df_tsne)
            sns.scatterplot(ax=axes[1],x="comp-1", y="comp-3", hue=df_tsne.y.tolist(),
                            palette=sns.color_palette("hls", n),
                            data=df_tsne)
            sns.scatterplot(ax=axes[2],x="comp-2", y="comp-3", hue=df_tsne.y.tolist(),
                            palette=sns.color_palette("hls", n),
                            data=df_tsne)
            plt.savefig('%s/%s_%s_tsne.png'%(folder, env_name, algo))
            plt.close()
