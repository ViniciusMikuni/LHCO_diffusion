from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def GetLocalFeat(pc,outsize):
    '''Return local features from embedded point cloud
    Input: point cloud shaped as (B,N,k,NFEAT)
    '''

    features = layers.Conv2D(outsize, kernel_size=[1,1],activation=None)(pc)
    features = layers.LeakyReLU(alpha=0.01)(features) 
    features = layers.Conv2D(outsize, kernel_size=[1,1],activation=None)(features)
    features = layers.LeakyReLU(alpha=0.01)(features) 
    features = tf.reduce_mean(features, axis=-2)    
    return features



def GetEdgeFeat(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
    point_cloud: (batch_size, num_points, 1, num_dims) 
    nn_idx: (batch_size, num_points, k)
    k: int
    Returns:
    edge features: (batch_size, num_points, k, num_dims)
    """



    point_cloud_central = point_cloud

    batch_size = tf.shape(point_cloud)[0]
    num_points = tf.shape(point_cloud)[1]
    num_dims = point_cloud.get_shape()[2]

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature

def pairwise_distance(point_cloud,mask): 
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1]) 
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose) # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True) # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])

    if mask != None:
        zero_mask = 10000*mask
        zero_mask_transpose = tf.transpose(zero_mask, perm=[0, 2, 1])
        zero_mask = zero_mask + zero_mask_transpose
        zero_mask = tf.where(tf.equal(zero_mask,20000),tf.zeros_like(zero_mask),zero_mask)
        point_cloud_square += zero_mask
        
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.math.top_k(neg_adj, k=k)  # values, indices
    return nn_idx


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
    
    #Include the time information as an additional feature fixed for all particles
    time = layers.Dense(2*projection_dim,activation=None)(time_embedding)
    time = layers.LeakyReLU(alpha=0.01)(time)
    time = layers.Dense(projection_dim)(time)

    time = layers.Reshape((1,-1))(time)
    time = tf.tile(time,(1,tf.shape(inputs)[1],1))

    if use_dist:
        k=7
        adj = pairwise_distance(masked_inputs[:,:,1:3],mask) #only eta-phi
        nn_idx = knn(adj, k=k)
        edge_feature = GetEdgeFeat(masked_inputs, nn_idx=nn_idx, k=k)
        masked_features = GetLocalFeat(edge_feature,projection_dim)
    else:
        masked_features = TimeDistributed(Dense(projection_dim,activation=None))(masked_inputs)

    
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event
    tdd = TimeDistributed(Dense(projection_dim,activation=None))(tf.concat([masked_features,time],-1))
    tdd = TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = TimeDistributed(Dense(projection_dim))(tdd)


    mask_matrix = tf.matmul(mask,tf.transpose(mask,perm=[0,2,1]))
    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #x1 =encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim//num_heads)(x1, x1, attention_mask=tf.cast(mask_matrix,tf.bool))
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.
        time_cond = layers.Dense(projection_dim,activation=None)(time)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)        
        x3 = layers.Dense(2*projection_dim,activation="gelu")(tf.concat([x3,time_cond],-1))
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    representation = TimeDistributed(Dense(2*projection_dim,activation=None))(tdd+representation)    
    representation =  TimeDistributed(layers.LeakyReLU(alpha=0.01))(representation)
    outputs = TimeDistributed(Dense(num_feat,activation=None,kernel_initializer="zeros"))(representation)
    
    return  inputs, outputs


def DeepSetsClass(
        #num_part,
        num_feat,
        num_heads=1,
        num_transformer = 8,
        projection_dim=256,
        mask = None,
):


    inputs = Input((None,num_feat))
    masked_inputs = layers.Masking(mask_value=0.0,name='Mask')(inputs)
        
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event
    tdd = TimeDistributed(Dense(projection_dim,activation=None))(masked_inputs)
    tdd = TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = TimeDistributed(Dense(projection_dim))(tdd)

    mask_matrix = tf.matmul(mask,tf.transpose(mask,perm=[0,2,1]))
    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #x1 =encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads,
            dropout=0.1)(x1, x1, attention_mask=tf.cast(mask_matrix,tf.bool))
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)        
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    representation = TimeDistributed(Dense(projection_dim,activation=None))(tdd+representation)    
    representation =  TimeDistributed(layers.LeakyReLU(alpha=0.01))(representation)

    merged = tf.reduce_mean(representation,1)
    merged = layers.LeakyReLU(alpha=0.01)(layers.Dense(projection_dim)(merged))    
    outputs = Dense(1,activation='sigmoid')(merged)
    
    return  inputs, outputs






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

