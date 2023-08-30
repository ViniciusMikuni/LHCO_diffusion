import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import os
import utils
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from classify import combine_part_jet,class_loader,get_classifier
import matplotlib.pyplot as plt
import horovod.tensorflow.keras as hvd

if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/LHCO/', help='Folder containing data and MC files')    
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--config', default='config_jet.json', help='Training parameters')    
    parser.add_argument('--nbkg', type=int,default=100000,help='Number of injected signal events')
    parser.add_argument('--nid', type=int,default=1,help='Number of independent trainings performed')
    parser.add_argument('--load', action='store_true', default=False,help='Load classifier results form previous evaluation')

    
    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)


    data_jet,data_part,data_mjj,labels = class_loader(flags.data_folder,
                                                      flags.file_name,
                                                      npart=flags.npart,
                                                      use_SR=True,
                                                      nsig = 100000,
                                                      nbkg = 100000,
                                                      mjjmax=config['MJJMAX'],
                                                      mjjmin=config['MJJMIN']
                                                      )
    data_j,data_p,data_mjj = combine_part_jet(data_jet,data_part,data_mjj,npart=flags.npart)
    mask_data = data_p[:,:,:,0]!=0
        
    nsigs = [500,1000,2000,3000,4000,5000,6000,7000,10000]
    models = {
        #'Hamburg': 'Flow Matching',
        config['MODEL_NAME']: 'Diffusion',
        'supervised': 'Idealized'
    }
    colors = ['#7570b3','#31a354','#FF6961']    
    sics = {model: [] for model in models.keys()}
    

    model = get_classifier(SR=True)
    for nsig in nsigs:
        print("NSIG: {}".format(nsig))
        for sample_name in models.keys():
            means = []
            for nid in range(flags.nid): #Load independent trainings

                if flags.load:
                    pred = np.load('../pred/pred_{}_SR_nsig_{}_nbkg_{}_nid{}.npy'.format(sample_name,nsig,flags.nbkg,nid))
                
                else:
                    checkpoint_folder = '../checkpoints/{}_SR_nsig_{}_nbkg_{}_nid{}/checkpoint'.format(sample_name,
                                                                                                       nsig,flags.nbkg,nid)
                    model.load_weights(checkpoint_folder).expect_partial()
                    pred = model.predict([data_j,data_p,mask_data],batch_size=2000)
                    np.save('../pred/pred_{}_SR_nsig_{}_nbkg_{}_nid{}'.format(sample_name,nsig,flags.nbkg,nid),pred)
                    
                fpr, tpr, _ = roc_curve(labels,pred, pos_label=1)
                auc_res =auc(fpr, tpr)
                print("AUC: {}".format(auc_res))
                sic = np.ma.divide(tpr,np.sqrt(fpr)).filled(0)
                sic[fpr<1e-4]=1.0                            
                max_sic = np.max(sic)
                means.append(max_sic)
                
                print("Max SIC: {}".format(max_sic))
                tf.keras.backend.clear_session()

            mean = np.mean(means)
            std = np.std(means)
            sics[sample_name].append([mean,std])
            

    for ic, model in enumerate(models.keys()):            
        plt.errorbar(nsigs,np.array(sics[model])[:,0],yerr = np.array(sics[model])[:,1],
                     color=colors[ic],linewidth=2,linestyle='-',label=models[model])
        
    plt.legend(loc='best',fontsize=16,ncol=1)
    plt.ylabel('Max. SIC')
    plt.xlabel('Injected Signal Events (nbkg = {})'.format(flags.nbkg))
    plt.savefig("{}/Max_SIC.pdf".format(flags.plot_folder))
