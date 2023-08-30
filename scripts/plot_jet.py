import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
from GSGM_lhco import GSGM
import energyflow as ef

import sys
from plot_class import PlottingConfig

def get_mjj(particle,jet):
    #Recover the particle information
    
    new_p = np.copy(particle)
    new_p[:,:,:,0]*=np.expand_dims(jet[:,:,0],-1)
    new_p[:,:,:,1]+=np.expand_dims(jet[:,:,1],-1)
    new_p[:,:,:,2]+=np.expand_dims(jet[:,:,2],-1)
    
    #fix phi
    # new_p[:,:,:,2][new_p[:,:,:,2]>np.pi] -= 2*np.pi
    # new_p[:,:,:,2][new_p[:,:,:,2]<-np.pi] += 2*np.pi
    mask = np.expand_dims(new_p[:,:,:,0]!=0,-1)
    new_p*=mask
    #Flatten particles
    #new_p = np.reshape(new_p,(-1,new_p.shape[1]*new_p.shape[2],new_p.shape[-1]))

    new_p = ef.p4s_from_ptyphims(new_p)
    mjj = ef.ms_from_p4s(np.sum(new_p,(1,2)))
    
    return mjj
    
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
        # ax0.set_ylim(top=config.max_y)
        if config.logy == False:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/LHCO/', help='Folder containing data and MC files')    
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--file_name', default='processed_data_background_rel.h5', help='File to load')
    parser.add_argument('--npart', default=279, type=int, help='Maximum number of particles')
    parser.add_argument('--config', default='config_jet.json', help='Training parameters')
    parser.add_argument('--nsplit', default=20,type = int, help='Number of batches to generate')
    
    parser.add_argument('--sample', action='store_true', default=False,help='Sample from the generative model')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--SR', action='store_true', default=False,help='Load signal region background events')
    parser.add_argument('--hamb', action='store_true', default=False,help='Load hamburg team files')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)

    model_name = config['MODEL_NAME']
    sample_name = model_name
    if flags.SR:
        sample_name += '_SR'
    if flags.hamb:
        sample_name += '_Hamburg'
    
    particles,jets,logmjj,_ = utils.DataLoader(flags.data_folder,
                                               flags.file_name,
                                               npart=flags.npart,
                                               norm=config['NORM'],
                                               make_tf_data=False,use_SR=flags.SR)
    
        
    if flags.test:
        particles_gen,jets_gen,mjj_gen = utils.SimpleLoader(flags.data_folder,
                                                            flags.file_name,
                                                            use_SR=flags.SR,
                                                            npart=flags.npart)
    else:        
        if flags.sample:            
            model = GSGM(config=config,npart=flags.npart)
            checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()

            particles_gen = []
            jets_gen = []

            if flags.SR:
                logmjj = utils.LoadMjjFile(flags.data_folder,'generated_data_all_200k.h5',use_SR=flags.SR)[:200000]
    
            
            for i,split in enumerate(np.array_split(logmjj,flags.nsplit)):
                #if i>2:break
                p,j = model.generate(split)
                particles_gen.append(p)
                jets_gen.append(j)
    
            particles_gen = np.concatenate(particles_gen)
            jets_gen = np.concatenate(jets_gen)
            
            particles_gen,jets_gen= utils.ReversePrep(particles_gen,
                                                      jets_gen,
                                                      mjj = utils.revert_mjj(logmjj),
                                                      npart=flags.npart,
                                                      norm=config['NORM'],
                                                      
                                                  
            )

            mjj_gen = utils.revert_mjj(logmjj)
            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"w") as h5f:
                dset = h5f.create_dataset("particle_features", data=particles_gen)
                dset = h5f.create_dataset("jet_features", data=jets_gen)
                dset = h5f.create_dataset("mjj", data=mjj_gen)
                
        elif flags.hamb:
            assert flags.SR, "ERROR: Hamburg files available only at SR"
            with h5.File(os.path.join(flags.data_folder,'generated_data_datacond_both_jets.h5'),"r") as h5f:
                particles_gen = np.stack([
                    h5f['particle_data_rel_x'][:],
                    h5f['particle_data_rel_y'][:]],1)

                jets_gen = np.stack([
                    h5f['jet_features_x'][:],
                    h5f['jet_features_y'][:]],1)

                mjj_gen = h5f['mjj'][:]
                
        else:
            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
                jets_gen = h5f['jet_features'][:]
                particles_gen = h5f['particle_features'][:]
                mjj_gen = h5f['mjj'][:]

    particles,jets= utils.ReversePrep(particles,jets,mjj = utils.revert_mjj(logmjj),
                                      npart=flags.npart,norm=config['NORM'])


    feed_dict = {
        'true':get_mjj(particles,jets),
        'gen': get_mjj(particles_gen,jets_gen)
    }
    
    fig,gs,_ = utils.HistRoutine(feed_dict,xlabel="mjj GeV",
                                 binning=np.linspace(2800,4200,100),
                                 plot_ratio=True,
                                 reference_name='true',
                                 ylabel= 'Normalized entries',logy=True)
        
    fig.savefig('{}/mjj_{}.pdf'.format(flags.plot_folder,sample_name),bbox_inches='tight')

    #Flatten and plot other features
    
    jets = jets.reshape(-1,config['NUM_JET'])
    jets_gen = jets_gen.reshape(-1,config['NUM_JET'])
    title = 'jet' if flags.SR==False else 'jet_SR'
    if flags.hamb: title+='_Hamburg'
    
    plot(jets,jets_gen,title=title,
         nplots=config['NUM_JET'],plot_folder=flags.plot_folder)
    
        
    particles_gen=particles_gen.reshape((-1,config['NUM_FEAT']))
    mask_gen = particles_gen[:,0]!=0.
    particles_gen=particles_gen[mask_gen]
    particles=particles.reshape((-1,config['NUM_FEAT']))
    mask = particles[:,0]!=0.
    particles=particles[mask]
    title = 'part' if flags.SR==False else 'part_SR'
    if flags.hamb: title+='_Hamburg'
    
    plot(particles,particles_gen,
         title=title,
         nplots=config['NUM_FEAT'],
         plot_folder=flags.plot_folder)
