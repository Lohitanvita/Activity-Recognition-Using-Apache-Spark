#=========================MATPLOT=====================================
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('seaborn-deep')

# Plots--------------------------------------------------------------")
numeric_data = data.select(numeric_features).sample(False, 0.10).toPandas()
n = len(numeric_data.columns)
for i in range(n):
    for j in range(n):
        ax = plt.subplot(numeric_data.plot.hexbin(x=i,y=j,sharex=False))
        xlbl = ax.xaxis.get_label_text()
        ylbl = ax.yaxis.get_label_text()
        plt.savefig(r"wisdm_main_ver_0.0/plot/Fig: "+str(xlbl)+"_"+str(ylbl)+".png")

axs = pd.scatter_matrix(numeric_data,diagonal='hist', alpha=0.2, figsize=(16,16));
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
plt.savefig(r"wisdm_main_ver_0.0/plot/Scatter_Matrix.png")