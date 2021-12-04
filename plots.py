import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import params
import lib_train


sigmas = [0, 0.05, 0.1, 0.25, 0.5]


accuracy_training = [[1, 0.84995, 0.7500166666666667, 0.5632166666666667, 0.3676333333333333], [0.9921666666666666, 0.9901333333333333, 0.9783166666666666, 0.7733666666666666, 0.5146666666666667], [0.9889333333333333, 0.9886, 0.9855333333333334, 0.9209333333333334, 0.6081], [0.97755, 0.9777333333333333, 0.9771333333333333, 0.9691833333333333, 0.8510666666666666], [0.9462666666666667, 0.9458, 0.9447, 0.93795, 0.90455]]
accuracy_testing = [[0.9802, 0.8312, 0.7363, 0.5708, 0.3744], [0.9762, 0.9752, 0.9658, 0.7682, 0.5199], [0.9752, 0.976, 0.9732, 0.9117, 0.6083], [0.9703, 0.9714, 0.9702, 0.9637, 0.8534], [0.9499, 0.9501, 0.948, 0.9404, 0.9096]]
smoothed_accuracy_training = [[0.6672, 0.57665, 0.48435, 0.28591666666666665, 0.14401666666666665], [0.8878166666666667, 0.8435, 0.8092333333333334, 0.6120833333333333, 0.31133333333333335], [0.9800166666666666, 0.9749333333333333, 0.9610833333333333, 0.7995, 0.47185], [0.97735, 0.9769333333333333, 0.97595, 0.9667333333333333, 0.8165], [0.9438666666666666, 0.9433333333333334, 0.9425333333333333, 0.9363333333333334, 0.9054333333333333]]
smoothed_accuracy_testing = [[0.6673, 0.5805, 0.49, 0.2894, .1484], [0.8796, 0.8369, 0.8006, 0.6177, 0.3149], [0.9702, 0.9657, 0.9503, 0.792, 0.4728], [0.9712, 0.9712, 0.9708, 0.9624, 0.8175], [0.9495, 0.9483, 0.9471, 0.9399, 0.9089]]
accuracy_attacked = [0.068, 0.332, 0.612, 0.876, 0.716]
smoothed_accuracy_attacked = [[0.438, 0.462, 0.454, 0.438, 0.446], [0.6, 0.586, 0.582, 0.58, 0.574], [0.766, 0.768, 0.764, 0.76, 0.772], [0.946, 0.948, 0.95, 0.946, 0.946], [0.868, 0.868, 0.87, 0.864, 0.866]]


def plot_accuracy_matrix (array, Type): # Type can be "accuracy training", "accuracy testing", "smoothed accuracy training" or "smoothed accuracy testing". 
    plt.clf()
    array = np.transpose(np.array(array))
    df = pd.DataFrame(array, columns=sigmas, index=sigmas)
    plt.figure(figsize=(len(sigmas),len(sigmas)))
    heatmap = sns.heatmap(df, vmin=0, vmax=1, annot=True, cmap='BrBG', annot_kws={"size": 18})
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 18)
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 18)
    heatmap.invert_yaxis()
    if Type == "accuracy training":
        heatmap.set_title(r'Accuracy on the training dataset', fontdict={'fontsize':18}, pad=12)
        figname='accuracy_training_matrix'
    elif Type == "accuracy testing":
        heatmap.set_title(r'Accuracy on the testing dataset', fontdict={'fontsize':18}, pad=12)
        figname='accuracy_testing_matrix'
    elif Type == "smoothed accuracy training":
        heatmap.set_title(r'Randomized smoothing accuracy on the training dataset ($\sigma='+str(params.sigma)+'$)', fontdict={'fontsize':18}, pad=12)
        figname='smoothed_accuracy_training_matrix'
    elif Type == "smoothed accuracy testing":
        heatmap.set_title(r'Randomized smoothing accuracy on the testing dataset ($\sigma='+str(params.sigma)+'$)', fontdict={'fontsize':18}, pad=12)
        figname='smoothed_accuracy_testing_matrix'
    else:
        return 1
    plt.xlabel(r'$\sigma$ of injected noise', size=24)
    plt.ylabel(r'$\sigma$ of noise', size=24)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/"+figname+".png", dpi=100)
    plt.gcf().savefig("plots/"+figname+".eps", dpi=100)
    plt.close()
    
    
def plot_attacked_accuracy (accuracy_attacked, smoothed_accuracy_attacked, sigmas):
    plt.clf()
    plt.title(r'ZOO-attacked testing set', size=24)
    plt.xlabel(r'$\sigma$ of injected noise', size=24)
    plt.ylabel('Accuracy', size=24)
    plt.tick_params(labelsize=18)
    plt.plot(sigmas, accuracy_attacked, color='blue', linewidth=3)
    plt.plot(sigmas, [np.mean(a) for a in smoothed_accuracy_attacked], color='red', linewidth=3)
    plt.fill_between(sigmas, np.array([np.mean(a) for a in smoothed_accuracy_attacked])-np.array([np.std(a) for a in smoothed_accuracy_attacked]), np.array([np.mean(a) for a in smoothed_accuracy_attacked])+np.array([np.std(a) for a in smoothed_accuracy_attacked]), color='red', alpha=0.33)
    
    recs = [mpatches.Rectangle((0,0),1,1,fc='blue'), mpatches.Rectangle((0,0),1,1,fc='red')]
    legend_labels = ["Original classifier", r'Randomized smoothing classifier ($\sigma='+str(params.sigma)+'$)']
    plt.legend(recs, legend_labels, loc=4, fontsize=18)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/attacked_accuracy.png", dpi=100)
    plt.gcf().savefig("plots/attacked_accuracy.eps", dpi=100)
    plt.close()
    
    
def plot_four_way_comparison (accuracy_testing, smoothed_accuracy_testing, accuracy_attacked, smoothed_accuracy_attacked, sigmas):
    plt.clf()
    plt.title(r'Original classifier (OC) versus randomized smoothing classifier (RSC), $\sigma=0.2$', size=24)
    plt.xlabel(r'$\sigma$ of injected noise', size=24)
    plt.ylabel('Accuracy', size=24)
    plt.tick_params(labelsize=18)
    plt.plot(sigmas, [a[0] for a in accuracy_testing], color='blue', linewidth=3)
    plt.plot(sigmas, [a[3] for a in accuracy_testing], color='blue', linewidth=3, linestyle='--')
    plt.plot(sigmas, [a[4] for a in accuracy_testing], color='blue', linewidth=3, linestyle='-.')
    plt.plot(sigmas, accuracy_attacked, color='blue', linewidth=3, linestyle=':')
    plt.plot(sigmas, [a[1] for a in smoothed_accuracy_testing], color='red', linewidth=3)
    plt.plot(sigmas, [a[3] for a in smoothed_accuracy_testing], color='red', linewidth=3, linestyle='--')
    plt.plot(sigmas, [a[4] for a in smoothed_accuracy_testing], color='red', linewidth=3, linestyle='-.')
    plt.plot(sigmas, [np.mean(a) for a in smoothed_accuracy_attacked], color='red', linewidth=3, linestyle=':')
    
    recs = [mlines.Line2D([],[],color='blue'), mlines.Line2D([],[],color='blue',linestyle='--'), mlines.Line2D([],[],color='blue',linestyle='-.'), mlines.Line2D([],[],color='blue',linestyle=':'), mlines.Line2D([],[],color='red'), mlines.Line2D([],[],color='red',linestyle='--'), mlines.Line2D([],[],color='red',linestyle='-.'), mlines.Line2D([],[],color='red',linestyle=':')]
    legend_labels = [r'OC: no noise', r'OC: $\sigma=0.25$', r'OC: $\sigma=0.5$', r'OC: ZOO', r'RSC: $\sigma=0.05$', r'RSC: $\sigma=0.25$', r'RSC: $\sigma=0.5$', 'RSC: ZOO']
    plt.legend(recs, legend_labels, loc=4, fontsize=18)
    plt.gcf().set_size_inches(25.6, 14.4)
    plt.gcf().savefig("plots/four_way_comparison.png", dpi=100)
    plt.gcf().savefig("plots/four_way_comparison.eps", dpi=100)
    plt.close()


if __name__ == '__main__':

    time_begin = time.time() # Updated Print function that, in addition to string s, prints out a time stamp.
    def Print(s):
        print (str(int(time.time()-time_begin))+' sec:        ' + s)
        
    #plot_accuracy_matrix (accuracy_training, "accuracy training")
    #plot_accuracy_matrix (accuracy_testing, "accuracy testing")
    #plot_accuracy_matrix (smoothed_accuracy_training, "smoothed accuracy training")
    #plot_accuracy_matrix (smoothed_accuracy_testing, "smoothed accuracy testing")
    #plot_attacked_accuracy (accuracy_attacked, smoothed_accuracy_attacked, sigmas)
    plot_four_way_comparison (accuracy_testing, smoothed_accuracy_testing, accuracy_attacked, smoothed_accuracy_attacked, sigmas)