import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import pandas as pd
import os
import utils
import tensorflow as tf
from GSGM import GSGM
from GSGM_distill import GSGM_distill
from deepsets_cond import DeepSetsClass
import time
import gc
import sys

from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras import  Input

def class_loader(data_path,
                 file_name,
                 use_SR=False,
                 nsignal=15000,
                 nbkg=60671,
                 
):

    if not use_SR:
        nsignal = 0

    parts_bkg,jets_bkg,mjj_bkg = utils.SimpleLoader(data_path,file_name,use_SR=flags.SR)
    parts_sig,jets_sig,mjj_sig = utils.SimpleLoader(data_path,file_name,use_SR=flags.SR,
                                                    load_signal=True)
    
    
    #flatten particles
    parts_bkg = parts_bkg[:nbkg].reshape(parts_bkg[:nbkg].shape[0],-1,parts_bkg.shape[-1])
    mjj_bkg = mjj_bkg[:nbkg]
    jets_bkg = jets_bkg[:nbkg]
    
    if nsignal>0:
        parts_sig = parts_sig[:nsignal].reshape(nsignal,-1,parts_sig.shape[-1])
        mjj_sig = mjj_sig[:nsignal]
        jets_sig = jets_sig[:nsignal]

    
        labels = np.concatenate([np.zeros_like(mjj_bkg),np.ones_like(mjj_sig)])
        particles = np.concatenate([parts_bkg,parts_sig],0)
        jets = np.concatenate([jets_bkg,jets_sig],0)
        mjj = np.concatenate([mjj_bkg,mjj_sig],0)

    else:
        labels = np.zeros_like(mjj_bkg)
        particles = parts_bkg
        jets = jets_bkg
        mjj = mjj_bkg
        
    return particles,jets,mjj,labels


def get_classifier(max_epoch,batch_size,learning_rate,nevts):
    #Define the model
    inputs_mask = Input((None,1))        
    inputs,outputs = DeepSetsClass(
        num_feat = 4,
        num_transformer = 4,
        projection_dim = 128,
        mask = inputs_mask,
    )

    model = keras.Model(inputs=[inputs,inputs_mask],
                        outputs=outputs)
    
    lr_schedule = keras.experimental.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max_epoch*nevts/batch_size
    )
    
    opt = keras.optimizers.Adamax(learning_rate=lr_schedule)
    #opt = keras.optimizers.Adam(learning_rate=LR)

    model.compile(            
        optimizer=opt,
        #run_eagerly=True,
        loss="binary_crossentropy",
        experimental_run_tf_function=False,
        weighted_metrics=[])
    return model


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/LHCO/', help='Folder containing data and MC files')    
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--file_name', default='events_anomalydetection_v2.features_with_jet_constituents.h5', help='File to load')

    parser.add_argument('--config', default='config_jet.json', help='Training parameters')    
    parser.add_argument('--SR', action='store_true', default=False,help='Load signal region background events')
    parser.add_argument('--nsig', type=int,default=2500,help='Number of injected signal events')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    MAX_EPOCH = 30
    BATCH_SIZE = 256
    LR = 1e-4

    _,data,_,labels = class_loader(flags.data_folder,
                                   flags.file_name,
                                   use_SR=flags.SR,
                                   nsignal = flags.nsig)

    sample_name = config['MODEL_NAME']    
    if flags.SR:
        sample_name += '_SR'

    with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
        #bkg = h5f['particle_features'][:]
        bkg = h5f['jet_features'][:]
        mjj = h5f['mjj'][:]


    semi_labels = np.concatenate([np.zeros(bkg.shape[0]),np.ones(data.shape[0])],0)
    sample = np.concatenate([bkg.reshape(bkg.shape[0],-1,bkg.shape[-1]),data],0)
    mask = sample[:,:,0]>0

    sample,semi_labels = shuffle(sample,semi_labels, random_state=10)
    
    model = get_classifier(MAX_EPOCH,BATCH_SIZE,LR,sample.shape[0])
        
    model.fit([sample,sample[:,:,0]>0],
              semi_labels,
              batch_size=BATCH_SIZE,
              epochs=MAX_EPOCH,shuffle=True,)

    if flags.SR:    
        pred = model.predict([data,data[:,:,0]>0])
        fpr, tpr, _ = roc_curve(labels,pred, pos_label=1)
    
        auc_res =auc(fpr, tpr)
        print("AUC: {}".format(auc_res))
        print("Max SIC: {}".format(np.max(np.ma.divide(tpr,np.sqrt(fpr)).filled(0))))
        nsig = np.sum(labels)*1.0
        nbkg = data.shape[0]*1.0 - nsig
        
        print("s/b(%): {}, s/sqrt(b): {}, s: {}, b: {}".format(nsig/nbkg*100,
                                                               nsig/nbkg,
                                                               nsig,nbkg
        ))


        plt.figure(figsize=(10,8))
        plt.plot(1.0/fpr, tpr/np.sqrt(fpr),"-", label='SIC', linewidth=1.5)
        plt.xlabel("1/FPR")
        plt.ylabel("TPR/sqrt(FPR)")
        plt.semilogx()
        
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('{}/sic{}.pdf'.format(flags.plot_folder,"_SR" if flags.SR else ""))

    else:
        pred = model.predict([sample,sample[:,:,0]>0])
        fpr, tpr, _ = roc_curve(semi_labels,pred, pos_label=1)
        auc_res =auc(fpr, tpr)
        print("AUC: {}".format(auc_res))
