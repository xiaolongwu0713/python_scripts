from others_works.EEGNet_in_pytorch.src.data import DataBuildClassifier
import os
import numpy as np
from others_works.EEGNet_in_pytorch.src.utils import single_auc_loging
from others_works.EEGNet_in_pytorch.src.utils import prepare_dirs,write_results_table, separte_last_block
from others_works.EEGNet_in_pytorch.src.model_torch import train_model_eegnet
from sklearn.model_selection import StratifiedKFold

experiment_res_dir = './res/' #Path to save results and training|testing statistics
all_subjects = [25,26,27,28,29,30,32,33,34,35,36,37,38]
data = DataBuildClassifier('/home/likan_blk/BCI/NewData')


params = {'resample_to': 369,
                 'D': 3,
                 'F1': 12,
                 'dropoutRate1': 0.52,
                 'dropoutRate2': 0.36,
                 'lr': 0.00066,
                 'norm_rate': 0.275
                 }

subjects = data.get_data(all_subjects,shuffle=False, windows=[(0.2,0.5)],baseline_window=(0.2,0.3),resample_to=params['resample_to'])

def cv_per_subj_test(x,y,params,path_to_subj, test_on_last_block=False, plot_fold_history=False):
    model_path = os.path.join(path_to_subj,'checkpoints')
    best_val_epochs = []
    best_val_aucs = []
    folds = 4  # To preserve split as 0.6 0.2 0.2
    if test_on_last_block:
        x_tr,y_tr,x_tst,y_tst = separte_last_block(x,y,test_size=0.2)

    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    cv_splits = list(cv.split(x_tr, y_tr))
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        fold_model_path = os.path.join(model_path, '%d' % fold)
        os.makedirs(fold_model_path)
        x_tr_fold, y_tr_fold = x_tr[train_idx], y_tr[train_idx]
        x_val_fold, y_val_fold = x_tr[val_idx], y_tr[val_idx]
        val_history, fold_model = train_model_eegnet(x_tr_fold,y_tr_fold,params,(x_val_fold,y_val_fold),epochs=200,
                                                     batch_size=32, shuffle=True,
                                                     model_path=os.path.join(fold_model_path,'model{}'.format(fold)))
        best_val_epochs.append(np.argmax(val_history['val_auc']) + 1)  # epochs count from 1 (not from 0)
        best_val_aucs.append(np.max(val_history['val_auc']))
        if plot_fold_history:
            single_auc_loging(val_history, 'fold %d' % fold, fold_model_path)

    if test_on_last_block:
        test_history, final_model = train_model_eegnet(x_tr, y_tr, params, epochs=int(np.mean(best_val_epochs)),
                                                       validation_data=(x_tst, y_tst), batch_size=32, shuffle=True,
                                                       model_path=os.path.join(path_to_subj,'naive_model'))

    single_auc_loging(test_history, 'test_history', path_to_save=path_to_subj)
    with codecs.open('%s/res.txt' % path_to_subj, 'w', encoding='utf8') as f:
        f.write(u'Val auc %.02f±%.02f\n' % (np.mean(best_val_aucs),np.std(best_val_aucs)))
        f.write('Test auc naive %.02f\n' % (test_history['val_auc'][-1]))

    return {'val_auc':test_history['val_auc'][-1]}, final_model

experiment_res_dir = './res/'
subjs_test_stats = {}
for train_subject in all_subjects:
    path_to_subj = prepare_dirs(experiment_res_dir, train_subject)
    x = subjects[train_subject][0]
    x = x.transpose(0, 2, 1)[:, np.newaxis, :, :]
    y=subjects[train_subject][1]
    test_stats, model = cv_per_subj_test(x, y, params, path_to_subj,test_on_last_block=True, plot_fold_history=True)
    subjs_test_stats[train_subject] = test_stats

write_results_table(subjs_test_stats, path_to_exp=experiment_res_dir)
