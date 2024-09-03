import numpy as np
from sklearn.metrics import roc_curve,auc
import scipy.io as scio
import sys
sys.path.append("..")
from config import get_args
from matplotlib import pyplot as plt


def get_epoch_AUC(result_map, seed, is_show):
    args = get_args()
    path = fr'dataset/{args.data_name}/data.mat'
    gt = scio.loadmat(fr'{path}')['gt']
    try:
        nr, nc, _ = gt.shape
    except:
        print(gt.shape)
    # normalize
    try:
        min_val = np.min(result_map)
        max_val = np.max(result_map)
        range_val = max_val - min_val
        if range_val == 0:
            print("Warning: min and max values are equal. Division by zero.")
            print(min_val, max_val)   
            return 0
        else:
            result_norm = (result_map - min_val) / range_val
    except Exception as e:
        print("Unexpected error:", e)
        print(result_map.min(), result_map.max())
    gt = gt.flatten()
    result_norm = result_norm.flatten()
    FPR, TPR, threshold = roc_curve(gt, result_norm)
    epoch_auc = auc(FPR, TPR)
    if is_show:
        plt.title('ROC curve AUC = %0.4f' % epoch_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.plot(FPR, TPR, color='orange', linewidth=3)
        plt.plot([0, 1], [0, 1], color='m', linestyle='--') 
        plt.savefig('{}/{}/ROC_curve_{}.png'.format(args.save_path, args.data_name, seed))
        plt.clf()
        plt.close()

    return epoch_auc
