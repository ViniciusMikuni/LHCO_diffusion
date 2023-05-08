import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
#import horovod.tensorflow.keras as hvd
import argparse
import utils
from GSGM_lhco import GSGM
from GSGM_distill import GSGM_distill
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
import horovod.tensorflow.keras as hvd

tf.random.set_seed(1233)
#tf.keras.backend.set_floatx('float64')
if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='config_jet.json', help='Config file with training parameters')
    parser.add_argument('--file_name', default='events_anomalydetection_v2.features_with_jet_constituents.h5', help='File to load')
    parser.add_argument('--data_path', default='/global/cfs/cdirs/m3929/LHCO/', help='Path containing the training files')
    parser.add_argument('--distill', action='store_true', default=False,help='Use the distillation model')
    parser.add_argument('--load', action='store_true', default=False,help='Load trained model')
    #parser.add_argument('--train_jet', action='store_true', default=False,help='Train jet model alone')
    parser.add_argument('--factor', type=int,default=1, help='Step reduction for distillation model')


    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    npart = 30
    
    data_size,training_data,test_data = utils.DataLoader(flags.data_path,
                                                         flags.file_name,
                                                         npart,
                                                         hvd.rank(),hvd.size(),
                                                         config['BATCH'],)

    model = GSGM(config=config,npart=npart)

    model_name = config['MODEL_NAME']
    checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)

    
    
    if flags.distill:
        if flags.factor>2:
            checkpoint_folder = '../checkpoints_{}_d{}/checkpoint'.format(model_name,flags.factor//2)
            model = GSGM_distill(model.ema_jet,model.ema_part,factor=flags.factor//2,config=config)
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
            #previous student, now teacher
        else:
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
        model = GSGM_distill(model.ema_jet,model.ema_part,factor=flags.factor,
                             config=config,npart=npart)

        if hvd.rank()==0:print("Loading Teacher from: {}".format(checkpoint_folder))
        checkpoint_folder = '../checkpoints_{}_d{}/checkpoint'.format(model_name,flags.factor)
        
        
    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=config['LR']*hvd.size(),
        decay_steps=config['MAXEPOCH']*int(data_size/config['BATCH'])
    )

    #opt = tf.keras.optimizers.Adam(learning_rate=config['LR']*hvd.size())
    opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)

        
    model.compile(            
        optimizer=opt,
        #run_eagerly=True,
        experimental_run_tf_function=False,
        weighted_metrics=[])

    if flags.load:
        model.load_weights(checkpoint_folder).expect_partial()

    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        EarlyStopping(patience=100,restore_best_weights=True),
    ]

        
    if hvd.rank()==0:
        checkpoint = ModelCheckpoint(checkpoint_folder,mode='auto',
                                     save_best_only=True,
                                     period=1,save_weights_only=True)
        callbacks.append(checkpoint)
        
    
    history = model.fit(
        training_data,
        epochs=config['MAXEPOCH'],
        callbacks=callbacks,
        steps_per_epoch=int(data_size/config['BATCH']),
        validation_data=test_data,
        validation_steps=int(data_size*0.1/config['BATCH']),
        verbose=1 if hvd.rank()==0 else 0,
        #steps_per_epoch=1,
    )

    
