from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np




def DeepSetsAtt(
        #num_part,
        num_feat,
        time_embedding,
        num_heads=4,
        num_transformer = 4,
        projection_dim=32,
        mask = None,
        use_dist = False,
):


    inputs = Input((None,num_feat))
    if mask is None:
        mask = tf.ones_like(inputs[:,:,:1])

    
    masked_inputs = layers.Masking(mask_value=0.0,name='Mask')(inputs)
    masked_features = Dense(projection_dim,activation=None)(masked_inputs)
    masked_features = layers.LeakyReLU(alpha=0.01)(masked_features)
    
    #Include the time information as an additional feature fixed for all particles
    time = layers.Dense(2*projection_dim,activation=None)(time_embedding)
    time = layers.LeakyReLU(alpha=0.01)(time)
    time = layers.Dense(projection_dim)(time)

    time = layers.Reshape((1,-1))(time)
    time = tf.tile(time,(1,tf.shape(inputs)[1],1))
        
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event
    concat = layers.Concatenate(-1)([masked_features,time])
    tdd = TimeDistributed(Dense(projection_dim,activation=None))(concat)
    tdd = TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = TimeDistributed(Dense(projection_dim))(tdd)

    mask_matrix = tf.matmul(mask,tf.transpose(mask,perm=[0,2,1]))
    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = TimeDistributed(layers.LayerNormalization(epsilon=1e-6))(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,key_dim=projection_dim//num_heads
        )(x1, x1, attention_mask=tf.cast(mask_matrix,tf.bool))
        
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.        
        x3 = TimeDistributed(layers.LayerNormalization(epsilon=1e-6))(x2)        
        x3 = TimeDistributed(layers.Dense(4*projection_dim,activation="gelu"))(x3)
        x3 = TimeDistributed(layers.Dense(projection_dim,activation="gelu"))(x3)

        encoded_patches = layers.Add()([x3, x2])
        

    representation = TimeDistributed(layers.LayerNormalization(epsilon=1e-6))(encoded_patches)

    representation_mean = layers.GlobalAvgPool1D()(representation)
    representation_mean = layers.Concatenate(-1)([representation_mean,time_embedding])
    representation_mean = layers.Reshape((1,-1))(representation_mean)
    representation_mean = tf.tile(representation_mean,(1,tf.shape(inputs)[1],1))

    add = layers.Concatenate(-1)([tdd,representation,representation_mean])
    representation =  TimeDistributed(Dense(2*projection_dim,activation=None))(add)    
    representation =  TimeDistributed(layers.LeakyReLU(alpha=0.01))(representation)
    outputs = TimeDistributed(Dense(num_feat,activation=None,kernel_initializer="zeros"))(representation)

    
    return  inputs, outputs


def make_patches(inputs,projection_dim):
    tdd = Dense(projection_dim,activation=None)(inputs)
    tdd = layers.LeakyReLU(alpha=0.01)(tdd)
    encoded_patches = Dense(projection_dim)(tdd)
    return encoded_patches

def encode(inputs,projection_dim):
    masked_inputs = layers.Masking(mask_value=0.0)(inputs)
    masked_features = Dense(projection_dim,activation=None)(masked_inputs)
    masked_features = layers.LeakyReLU(alpha=0.01)(masked_features)
    return masked_features

def transformer(encoded_patches,num_transformer,num_heads,
                projection_dim,mask_matrix=None,attention_axes=(1,2)):
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads,
            attention_axes = attention_axes,
            dropout=0.1)(x1, x1, attention_mask=tf.cast(mask_matrix,tf.bool) if mask_matrix is not None else None)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)        
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return representation



def DeepSetsClass(
        inputs_jet,
        inputs_particle,
        num_heads=1,
        num_transformer = 8,
        projection_dim=256,
        mask = None,
        use_cond = False,
        cond_embedding = None
):
    
    
    masked_features = encode(inputs_particle,projection_dim)
    jet_features = encode(inputs_jet,projection_dim)

    mask_matrix = tf.matmul(mask,tf.transpose(mask,perm=[0,1,3,2]))
        
    if use_cond:
        pass
        cond = layers.Dense(2*projection_dim,activation=None)(cond_embedding)
        cond = layers.LeakyReLU(alpha=0.01)(cond)
        cond = layers.Dense(projection_dim)(cond)        
        cond = layers.Reshape((1,1,-1))(cond)
        cond = tf.tile(cond,(1,2,tf.shape(inputs_particle)[2],1))        
        masked_features = layers.Concatenate(-1)([masked_features,cond])


    encoded_patches = make_patches(masked_features,projection_dim)
    representation = transformer(encoded_patches,num_transformer,num_heads,      
                                 projection_dim,mask_matrix,attention_axes=(2))
    representation = layers.Reshape((-1,projection_dim))(representation)
    representation = layers.GlobalAvgPool1D()(representation)
    
    encoded_patches_jet = make_patches(jet_features,projection_dim)
    representation_jet = transformer(encoded_patches_jet,num_transformer,num_heads,      
                                     projection_dim,attention_axes=(1))
    representation_jet = layers.GlobalAvgPool1D()(representation_jet)

    
    merged = layers.Concatenate(-1)([representation,representation_jet])
    merged = Dense(2*projection_dim,activation=None)(merged)    
    merged = layers.LeakyReLU(alpha=0.01)(merged)
    #merged = layers.GlobalAvgPool1D()(merged)
    
    merged = layers.LeakyReLU(alpha=0.01)(layers.Dense(2*projection_dim)(merged))
    merged = layers.Dropout(0.1)(merged)
    merged = layers.LeakyReLU(alpha=0.01)(layers.Dense(projection_dim)(merged))    
    outputs = Dense(1,activation='sigmoid')(merged)
    
    return   outputs






def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
):

    
    act = layers.LeakyReLU(alpha=0.01)
    #act = swish

    def resnet_dense(input_layer,hidden_size,nlayers=2):
        layer = input_layer
        residual = layers.Dense(hidden_size)(layer)
        for _ in range(nlayers):
            layer=act(layers.Dense(hidden_size,activation=None)(layer))
            layer = layers.Dropout(0.1)(layer)
        return residual + layer
    
    embed = layers.Dense(mlp_dim)(time_embedding)
    residual = act(layers.Dense(2*mlp_dim)(tf.concat([inputs,embed],-1)))    
    residual = layers.Dense(mlp_dim)(residual)
    layer = residual
    for _ in range(num_layer-1):
        cond = layers.Dense(mlp_dim)(embed)
        layer =  resnet_dense(tf.concat([layer,cond],-1),mlp_dim)

    layer = act(layers.Dense(mlp_dim)(residual+layer))
    outputs = layers.Dense(end_dim,kernel_initializer="zeros")(layer)
    
    return outputs

