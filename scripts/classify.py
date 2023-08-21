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
from deepsets_cond import DeepSetsClass
import time
import gc
import sys

from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras import  Input
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

def combine_part_jet(particle,jet,npart=100):
    #Recover the particle information

    new_j = np.copy(jet)
    new_p = np.copy(particle)


    #Flatten
    new_j = np.reshape(new_j,(-1,jet.shape[-1]))
    new_p = np.reshape(new_p,(-1,particle.shape[-1]))
    
    mask = new_p[:,0]!=0    
    data_dict = utils.LoadJson('preprocessing_{}.json'.format(npart))
        
    new_j = np.ma.divide(new_j-data_dict['mean_jet'],data_dict['std_jet']).filled(0)
    new_p = np.ma.divide(new_p-data_dict['mean_particle'],data_dict['std_particle']).filled(0)
    new_p *=np.expand_dims(mask,-1)

    #Reshape it back
    new_j = np.reshape(new_j,jet.shape)
    new_p = np.reshape(new_p,particle.shape)
    return new_j, new_p



def class_loader(data_path,
                 file_name,
                 npart=100,
                 use_SR=False,
                 nsig=15000,
                 nbkg=60671,
                 mjjmin=2300,
                 mjjmax=5000
                 
):

    if not use_SR:
        nsig = 0

    parts_bkg,jets_bkg,mjj_bkg = utils.SimpleLoader(data_path,file_name,use_SR=flags.SR,
                                                    npart=npart,
                                                    mjjmax=mjjmax,mjjmin=mjjmin)
    parts_sig,jets_sig,mjj_sig = utils.SimpleLoader(data_path,file_name,use_SR=flags.SR,
                                                    npart=npart,load_signal=True,
                                                    mjjmax=mjjmax,mjjmin=mjjmin)
    
    
    #flatten particles
    parts_bkg = parts_bkg[:nbkg]
    mjj_bkg = mjj_bkg[:nbkg]
    jets_bkg = jets_bkg[:nbkg]
    
    if nsig>0:
        parts_sig = parts_sig[:nsig]
        mjj_sig = mjj_sig[:nsig]
        jets_sig = jets_sig[:nsig]
    
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


def get_classifier(max_epoch,batch_size,learning_rate,nevts,SR=False):
    #Define the model
    inputs_mask = Input((2,None,1))
    inputs_jet = Input((None,5))
    inputs_particle = Input((2,None,3))
    if SR:
        outputs = DeepSetsClass(
            inputs_jet,
            inputs_particle,
            num_heads = 2,
            num_transformer = 6,
            projection_dim = 128,
            mask = inputs_mask,
        )

        model = keras.Model(inputs=[inputs_jet,inputs_particle,inputs_mask],
                            outputs=outputs)

    else:
        inputs_cond = Input((1))
        outputs = DeepSetsClass(
            inputs_jet,
            inputs_particle,
            num_heads = 2,
            num_transformer = 6,
            projection_dim = 128,
            mask = inputs_mask,
            use_cond=True,
            cond_embedding = inputs_cond,
        )
        model = keras.Model(inputs=[inputs_jet,inputs_particle,inputs_mask,inputs_cond],
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
    parser.add_argument('--file_name', default='events_anomalydetection_v2.features_with_jet_constituents_100.h5', help='File to load')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--npart', default=100, type=int, help='Maximum number of particles')
    parser.add_argument('--config', default='config_jet.json', help='Training parameters')    
    parser.add_argument('--SR', action='store_true', default=False,help='Load signal region background events')
    parser.add_argument('--nsig', type=int,default=2500,help='Number of injected signal events')
    parser.add_argument('--nbkg', type=int,default=101214,help='Number of injected signal events')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    MAX_EPOCH = 200
    BATCH_SIZE = 128
    LR = 1e-4

    data_part,data_jet,data_mjj,labels = class_loader(flags.data_folder,
                                                      flags.file_name,
                                                      npart=flags.npart,
                                                      use_SR=flags.SR,
                                                      nsig = flags.nsig,
                                                      nbkg=flags.nbkg,
                                                      mjjmax=config['MJJMAX'],
                                                      mjjmin=config['MJJMIN']
                                                      )

    data_j,data_p = combine_part_jet(data_part,data_jet)
    sample_name = config['MODEL_NAME']    
    if flags.SR:
        sample_name += '_SR'

    if flags.test:
        bkg_part,bkg_jet,bkg_mjj = utils.SimpleLoader(flags.data_folder,flags.file_name,
                                                      use_SR=flags.SR,npart=flags.npart,
                                                      mjjmax=config['MJJMAX'],
                                                      mjjmin=config['MJJMIN']
                                                      )
        bkg_part = bkg_part[:flags.nbkg]
        bkg_jet = bkg_jet[:flags.nbkg]
        bkg_mjj = bkg_mjj[:flags.nbkg]
    else:
        with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
            bkg_part = h5f['particle_features'][:]
            bkg_jet = h5f['jet_features'][:]
            bkg_mjj = h5f['mjj'][:]
            

        
    bkg_j,bkg_p = combine_part_jet(bkg_part,bkg_jet)
    print("Loading {} generated samples and {} data samples".format(bkg_j.shape[0],data_j.shape[0]))
    semi_labels = np.concatenate([np.zeros(bkg_j.shape[0]),np.ones(data_j.shape[0])],0)
    sample_j = np.concatenate([bkg_j,data_j],0)
    sample_p = np.concatenate([bkg_p,data_p],0)
    mjj = np.concatenate([bkg_mjj,data_mjj],0)
    mjj = utils.prep_mjj(mjj,mjjmin=config['MJJMIN'],mjjmax=config['MJJMAX'])


    mask = sample_p[:,:,:,0]!=0        
    model = get_classifier(MAX_EPOCH,BATCH_SIZE,LR,sample_j.shape[0],flags.SR)
    checkpoint_folder = '../{}_class/checkpoint'.format(config['MODEL_NAME'])
    callbacks = [EarlyStopping(patience=20,restore_best_weights=True),]
    
    if flags.SR:
        
        if flags.test:
            weights = np.ones(sample_j.shape[0])
        else:
            print("Loading weights...")
            model_weight = get_classifier(MAX_EPOCH,BATCH_SIZE,LR,sample_j.shape[0],SR=False)
            model_weight.load_weights(checkpoint_folder).expect_partial()
            weights = utils.reweight(bkg_j,bkg_p,model_weight,
                                     utils.prep_mjj(bkg_mjj,mjjmin=config['MJJMIN'],mjjmax=config['MJJMAX']),                                     
                                     )
            weights = np.concatenate([weights,np.ones(data_j.shape[0])])
            
            callbacks.append(ModelCheckpoint('../{}_nsig_{}_nbkg_{}/checkpoint'.format(config['MODEL_NAME'],flags.nsig,flags.nbkg),
                                             mode='auto',save_best_only=True,
                                             period=1,save_weights_only=True))
    else:
        callbacks.append(ModelCheckpoint(checkpoint_folder,mode='auto',
                        save_best_only=True,
                        period=1,save_weights_only=True))
        weights = np.ones(sample_j.shape[0])

    sample_j,sample_p,semi_labels,mask,mjj,weights = shuffle(
        sample_j,sample_p,semi_labels,mask,mjj,weights, random_state=10)

    model.fit([sample_j,sample_p,mask] if flags.SR else [sample_j,sample_p,mask,mjj],
              semi_labels,
              batch_size=BATCH_SIZE,
              validation_split = 0.1,
              callbacks=callbacks,
              sample_weight = weights,
              epochs=MAX_EPOCH,shuffle=True,)

    if flags.SR:
        mask_data = data_p[:,:,:,0]!=0
        pred = model.predict([data_j,data_p,mask_data])
        fpr, tpr, _ = roc_curve(labels,pred, pos_label=1)
    
        auc_res =auc(fpr, tpr)
        print("AUC: {}".format(auc_res))
        print("Max SIC: {}".format(np.max(np.ma.divide(tpr,np.sqrt(fpr)).filled(0))))
        nsig = np.sum(labels)*1.0
        nbkg = data_j.shape[0]*1.0 - nsig
        
        print("s/b(%): {}, s/sqrt(b): {}, s: {}, b: {}".format(nsig/nbkg*100,
                                                               nsig/np.sqrt(nbkg),
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
        pred = model.predict([sample_j,sample_p,mask,mjj])
        fpr, tpr, _ = roc_curve(semi_labels,pred, pos_label=1)
        auc_res =auc(fpr, tpr)
        print("AUC: {}".format(auc_res))
