import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd

#import energyflow as ef

np.random.seed(0) #fix the seed to keep track of validation split

line_style = {
    'true':'dotted',
    'gen':'-',    
}

colors = {
    'true':'black',
    'gen':'#7570b3',
}

name_translate={
    'true':'True backgound distribution',
    'gen':'Generated background distribution',
}

nevts = 100000

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    
    # hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs



        
def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-100,100])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.1f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,
                xlabel='',ylabel='',
                reference_name='Geant',
                logy=False,binning=None,
                fig = None, gs = None,
                plot_ratio= True,
                idx = None,
                label_loc='best'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    if fig is None:
        fig,gs = SetGrid(plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)
        
    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),20)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    maxy = np.max(reference_hist)
    
    for ip,plot in enumerate(feed_dict.keys()):
        dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=name_translate[plot],linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
        if plot_ratio:
            if reference_name!=plot:
                ratio = 100*np.divide(reference_hist-dist,reference_hist)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc=label_loc,fontsize=12,ncol=5)

    if logy:
        ax0.set_yscale('log')



    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100,100])
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0)
    
    return fig,gs, binning


def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


def SimpleLoader(data_path,file_name,use_SR=False,load_signal=False):
    lhco = pd.read_hdf(
        os.path.join(data_path,file_name)).to_numpy().astype(np.float32)[:]
    
    parts = lhco[:,14:14+2*30*3].reshape(-1,2*30,3)
    mjj = lhco[:,-2]
    label = lhco[:,-1]
    
    jet1 = lhco[:,:4]
    jet2 = lhco[:,7:11]

    if load_signal:
        mask_label = label==1
    else:
        mask_label = label==0
    #Ensure there are 2 jets
    mask_mass = (jet1[:,-1]>0.1) & (jet2[:,-1]>0.1)

    # train using only the sidebands
    if use_SR:
        mask_region = (mjj>3300) & (mjj<3700)
    else:
        #mask_region = ((mjj<3300) & (mjj>3000)) | ((mjj>3700) & (mjj<4000))
        mask_region = (mjj<3300)  | (mjj>3700) 

    parts = parts[(mask_label) & (mask_region) & (mask_mass)]
    mjj = mjj[(mask_label) & (mask_region) & (mask_mass)]
    jet1 = jet1[(mask_label) & (mask_region) & (mask_mass)]
    jet2 = jet2[(mask_label) & (mask_region) & (mask_mass)]

    particles,jets = convert_inputs(parts,jet1,jet2)
    particles,jets,mjj = shuffle(particles,jets, mjj,random_state=0)

    mask = np.sqrt(particles[:,:,:,1]**2 + particles[:,:,:,2]**2) < 1.1 #eta looks off
    particles*=np.expand_dims(mask,-1)
    
    mask = np.expand_dims(particles[:,:,:,-1],-1)
    return particles[:,:,:,:-1]*mask,jets,mjj

        

def revert_npart(npart,max_npart):
    #Revert the preprocessing to recover the particle multiplicity
    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(max_npart))
    x = npart*data_dict['std_jet'][-1] + data_dict['mean_jet'][-1]
    x = revert_logit(x)
    x = x * (data_dict['max_jet'][-1]-data_dict['min_jet'][-1]) + data_dict['min_jet'][-1]
    #x = np.exp(x)
    return np.round(x).astype(np.int32)
     
def revert_logit(x):
    alpha = 1e-6
    exp = np.exp(x)
    x = exp/(1+exp)
    return (x-alpha)/(1 - 2*alpha)                

def ReversePrep(particles,jets,npart):
    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(npart))
    num_part = particles.shape[2]
    batch_size = particles.shape[0]
    particles=particles.reshape(-1,particles.shape[-1])
    jets=jets.reshape(-1,jets.shape[-1])
    mask = np.expand_dims(particles[:,0]!=0,-1)
    def _revert(x,name='jet'):    
        x = x*data_dict['std_{}'.format(name)] + data_dict['mean_{}'.format(name)]
        x = revert_logit(x)
        #print(data_dict['max_{}'.format(name)],data_dict['min_{}'.format(name)])
        x = x * (np.array(data_dict['max_{}'.format(name)]) -data_dict['min_{}'.format(name)]) + data_dict['min_{}'.format(name)]
        return x
        
    particles = _revert(particles,'particle')
    jets = _revert(jets,'jet')
    jets[:,-1] = np.round(jets[:,-1])
    particles[:,0] = 1.0 - particles[:,0]
    return (particles*mask).reshape(batch_size,2,num_part,-1),jets.reshape(batch_size,2,-1)


def convert_inputs(parts,jet1,jet2,npart=30):

    def _convert_to_polar(jet):
        new_jet = np.zeros_like(jet)
        new_jet[:,0] = np.sqrt(jet[:,0]**2 + jet[:,1]**2) #pt
        new_jet[:,1] = np.arcsinh(np.ma.divide(jet[:,2],new_jet[:,0]).filled(0))
        new_jet[:,2] = np.arctan2(jet[:,1],jet[:,0])
        new_jet[:,3] = jet[:,3]
        return new_jet
    
    jet1 = _convert_to_polar(jet1)
    jet2 = _convert_to_polar(jet2)

    #particles
    pt = np.sqrt(parts[:,:,0]**2 + parts[:,:,1]**2)
    mask = (pt>0).astype(np.float32)
    
    eta = np.arcsinh(np.ma.divide(parts[:,:,2],pt).filled(0)) 
    phi = np.arctan2(parts[:,:,1],parts[:,:,0])*mask


    pt[:,:npart]  /=jet1[:,0].reshape(-1,1)
    eta[:,:npart] -=jet1[:,1].reshape(-1,1)
    phi[:,:npart] -=jet1[:,2].reshape(-1,1)
    
    pt[:,npart:]  /=jet2[:,0].reshape(-1,1)
    eta[:,npart:] -=jet2[:,1].reshape(-1,1)
    phi[:,npart:] -=jet2[:,2].reshape(-1,1)
    
    phi[phi>np.pi] -= 2*np.pi
    phi[phi<-np.pi] += 2*np.pi 

    

    # print(np.min(phi),np.max(phi))
    # input()
    #print(jet2[:,2].reshape(-1,1)[np.any(np.abs(phi[:,npart:])>3)])
    
    
    particles = np.stack([pt,eta*mask,phi*mask,mask],-1).reshape(-1,2,npart,4)
    
    npart1 = np.sum(mask[:,:npart],-1)
    npart2 = np.sum(mask[:,npart:],-1)

    #delete phi and add mask
    jet1 = np.concatenate([np.delete(jet1,2,1),npart1.reshape(-1,1)],-1)
    jet2 = np.concatenate([np.delete(jet2,2,1),npart2.reshape(-1,1)],-1)
    
    
    jets = np.stack([jet1,jet2],1)

    # #Use most energetic jet as the reference
    # condition = jet1[:,0] > jet2[:,0]
    # ref = np.where(condition.reshape(-1,1),jet1,jet2)
    
    # pt = np.sqrt(parts[:,:,0]**2 + parts[:,:,1]**2)
    # eta = np.arcsinh(np.ma.divide(parts[:,:,2],pt).filled(0)) - ref[:,1].reshape(-1,1)

    # phi = np.arctan2(parts[:,:,1],parts[:,:,0]) - ref[:,2].reshape(-1,1)
    # cosphi = np.cos(phi)
    # sinphi = np.sin(phi)
    # # phi[phi>np.pi] -= 2*np.pi
    # # phi[phi<-np.pi] += 2*np.pi 
    
    # mask = (pt>0).astype(np.float32)
    # npart = np.sum(mask,-1)

    # jets = np.concatenate([ref[:,:2],npart.reshape(-1,1)],-1)
    # particles = np.stack([pt/ref[:,0].reshape(-1,1),eta,sinphi,cosphi,mask],-1)
    
    return particles,jets

    
def DataLoader(data_path,file_name,
               npart,
               rank=0,size=1,
               batch_size=64,
               make_tf_data=True,
               use_SR=False,
               #train_jet=True,
):
    particles = []
    jets = []

    def _preprocessing(particles,jets,save_json=False):
        num_part = particles.shape[2]
        batch_size = particles.shape[0]
        
        mask = np.sqrt(particles[:,:,:,1]**2 + particles[:,:,:,2]**2) < 1.1 #eta looks off
        # print(np.sum(mask)/np.prod(mask.shape))
        # input()
        particles*=np.expand_dims(mask,-1)

        particles=particles.reshape(-1,particles.shape[-1]) #flatten
        jets=jets.reshape(-1,jets.shape[-1]) #flatten

        def _logit(x):                            
            alpha = 1e-6
            x = alpha + (1 - 2*alpha)*x
            return np.ma.log(x/(1-x)).filled(0)

        #Transformations
        particles[:,0] = 1.0 - particles[:,0]

        if save_json:
            data_dict = {
                'max_jet':np.max(jets,0).tolist(),
                'min_jet':np.min(jets,0).tolist(),
                'max_particle':np.max(particles[:,:-1],0).tolist(),
                'min_particle':np.min(particles[:,:-1],0).tolist(),
            }                
            
            SaveJson('preprocessing_{}.json'.format(npart),data_dict)
        else:
            data_dict = LoadJson('preprocessing_{}.json'.format(npart))

        #normalize
        jets = np.ma.divide(jets-data_dict['min_jet'],np.array(data_dict['max_jet'])- data_dict['min_jet']).filled(0)        
        particles[:,:-1]= np.ma.divide(particles[:,:-1]-data_dict['min_particle'],np.array(data_dict['max_particle'])- data_dict['min_particle']).filled(0)

        jets = _logit(jets)
        particles[:,:-1] = _logit(particles[:,:-1])
        if save_json:
            mask = particles[:,-1]
            mean_particle = np.average(particles[:,:-1],axis=0,weights=mask)
            data_dict['mean_jet']=np.mean(jets,0).tolist()
            data_dict['std_jet']=np.std(jets,0).tolist()
            data_dict['mean_particle']=mean_particle.tolist()
            data_dict['std_particle']=np.sqrt(np.average((particles[:,:-1] - mean_particle)**2,axis=0,weights=mask)).tolist()                        
            SaveJson('preprocessing_{}.json'.format(npart),data_dict)
        
            
        jets = np.ma.divide(jets-data_dict['mean_jet'],data_dict['std_jet']).filled(0)
        particles[:,:-1]= np.ma.divide(particles[:,:-1]-data_dict['mean_particle'],data_dict['std_particle']).filled(0)
        
        particles = particles.reshape(batch_size,2,num_part,-1)
        jets = jets.reshape(batch_size,2,-1)
        return particles.astype(np.float32),jets.astype(np.float32)
            

    # lhco = pd.read_hdf(
    #     os.path.join(data_path,file_name)).columns.values
    # print(lhco)
    # input()
    
    lhco = pd.read_hdf(
        os.path.join(data_path,file_name)).to_numpy().astype(np.float32)[rank::size]

    
    parts = lhco[:,14:14+2*30*3].reshape(-1,2*30,3)
    mjj = lhco[:,-2]    
    label = lhco[:,-1]
    jet1 = lhco[:,:4]
    jet2 = lhco[:,7:11]
    #keep background only
    mask_label = label==0
    mask_mass = (jet1[:,-1]>0.1) & (jet2[:,-1]>0.1)

    

    # train using only the sidebands
    if use_SR:
        mask_region = (mjj>3300) & (mjj<3700)
    else:
        #mask_region = ((mjj<3300) & (mjj>3000)) | ((mjj>3700) & (mjj<4000))
        mask_region = (mjj<3300)  | (mjj>3700)

        
    parts = parts[(mask_label) & (mask_region) & (mask_mass)]
    mjj = mjj[(mask_label) & (mask_region) & (mask_mass)]
    jet1 = jet1[(mask_label) & (mask_region) & (mask_mass)]
    jet2 = jet2[(mask_label) & (mask_region) & (mask_mass)]
    
    
    
    particles,jets = convert_inputs(parts,jet1,jet2)
    particles,jets,mjj = shuffle(particles,jets,mjj, random_state=0)

    data_size = jets.shape[0]

    particles,jets = _preprocessing(particles,jets)
    # input("done")
    if make_tf_data:
        train_particles = particles[:int(0.8*data_size)]
        train_jets = jets[:int(0.8*data_size)]
        train_mjj = mjj[:int(0.8*data_size)]
        
        test_particles = particles[int(0.8*data_size):]
        test_jets = jets[int(0.8*data_size):]
        test_mjj = mjj[int(0.8*data_size):]
        
        data_size = int(0.8*data_size)
    
        def _prepare_batches(particles,jets,mjj):
            # print(np.min(mjj),np.max(mjj))
            # input()
            tf_cond = tf.data.Dataset.from_tensor_slices(mjj/5000.)
            tf_jet = tf.data.Dataset.from_tensor_slices(jets)
            mask = np.expand_dims(particles[:,:,:,-1],-1)
            masked = particles[:,:,:,:-1]*mask
            

            # print(mask)
            # input()
            tf_part = tf.data.Dataset.from_tensor_slices(masked)
            tf_mask = tf.data.Dataset.from_tensor_slices(mask)
            tf_zip = tf.data.Dataset.zip((tf_part, tf_jet,tf_cond,tf_mask))
            
            return tf_zip.shuffle(data_size).repeat().batch(batch_size)
    
        train_data = _prepare_batches(train_particles,train_jets,train_mjj)
        test_data  = _prepare_batches(test_particles,test_jets,test_mjj)

        return data_size, train_data,test_data
    
    else:
        nevts = 10000
        mask = particles[:nevts,:,:,-1].reshape(nevts,2,-1,1)
        return particles[:nevts,:,:,:-1]*mask,jets[:nevts],mjj[:nevts]/5000., mask
