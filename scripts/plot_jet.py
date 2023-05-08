import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
from GSGM_lhco import GSGM
from GSGM_distill import GSGM_distill
import time
import gc
import sys
from plot_class import PlottingConfig

    
def plot(jet1,jet2,nplots,title,plot_folder):
    for ivar in range(nplots):
        config = PlottingConfig(title,ivar)
        
            
        feed_dict = {
            'true':jet1[:,ivar],
            'gen': jet2[:,ivar]
        }
            

        fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                     binning=config.binning,
                                     plot_ratio=True,
                                     reference_name='true',
                                     ylabel= 'Normalized entries',logy=config.logy)
        
        ax0 = plt.subplot(gs[0])     
        ax0.set_ylim(top=config.max_y)
        if config.logy == False:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/FPCD_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')


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
    
    parser.add_argument('--distill', action='store_true', default=False,help='Use the distillation model')
    parser.add_argument('--sample', action='store_true', default=False,help='Sample from the generative model')
    parser.add_argument('--factor', type=int,default=1, help='Step reduction for distillation model')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--SR', action='store_true', default=False,help='Load signal region background events')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    npart = 30

    
    particles,jets,logmjj,_ = utils.DataLoader(flags.data_folder,
                                                flags.file_name,
                                                npart=npart,
                                                make_tf_data=False,use_SR=flags.SR)
    
    if flags.test:
        particles_gen,jets_gen,mjj_gen = utils.SimpleLoader(flags.data_folder,flags.file_name,use_SR=flags.SR)
    else:
    
        model_name = config['MODEL_NAME']
        sample_name = model_name
        if flags.SR:
            sample_name += '_SR'
        if flags.distill:
            sample_name += '_d{}'.format(flags.factor)

        if flags.sample:            
            model = GSGM(config=config,factor=flags.factor,npart=npart)
            checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
            if flags.distill:
                checkpoint_folder = '../checkpoints_{}_d{}/checkpoint'.format(model_name,flags.factor)
                model = GSGM_distill(model.ema_jet,model.ema_part,config=config,
                                     factor=flags.factor,npart=npart)
                print("Loading distilled model from: {}".format(checkpoint_folder))
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()

            particles_gen = []
            jets_gen = []

            nsplit = 2
            noversample = 1
            logmjj = np.repeat(logmjj,noversample,0)
            #split_part = np.array_split(jets,nsplit)
            for i,split in enumerate(np.array_split(logmjj,nsplit)):
                # if i>0:break
                #,split_part[i]
                p,j = model.generate(split)
                particles_gen.append(p)
                jets_gen.append(j)
    
            particles_gen = np.concatenate(particles_gen)
            jets_gen = np.concatenate(jets_gen)
            
            particles_gen,jets_gen= utils.ReversePrep(particles_gen,
                                                      jets_gen,
                                                      npart=npart,
                                                  
            )

            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"w") as h5f:
                dset = h5f.create_dataset("particle_features", data=particles_gen)
                dset = h5f.create_dataset("jet_features", data=jets_gen)
                dset = h5f.create_dataset("mjj", data=np.exp(logmjj))
                
        else:
            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
                jets_gen = h5f['jet_features'][:]
                particles_gen = h5f['particle_features'][:]
                mjj_gen = h5f['mjj'][:]

        
            #assert np.all(mjj_gen == np.exp(logmjj)), 'The order between the particles dont match'

    particles,jets= utils.ReversePrep(particles,jets,npart=npart)

    jets = jets.reshape(-1,config['NUM_JET'])
    jets_gen = jets_gen.reshape(-1,config['NUM_JET'])
    plot(jets,jets_gen,title='jet' if flags.SR==False else 'jet_SR',
         nplots=3,plot_folder=flags.plot_folder)
    
        
    particles_gen=particles_gen.reshape((-1,config['NUM_FEAT']))
    mask_gen = particles_gen[:,0]>0.
    particles_gen=particles_gen[mask_gen]
    particles=particles.reshape((-1,config['NUM_FEAT']))
    mask = particles[:,0]>0.
    particles=particles[mask]
    plot(particles,particles_gen,
         title='part' if flags.SR==False else 'part_SR',
         nplots=3,
         plot_folder=flags.plot_folder)


