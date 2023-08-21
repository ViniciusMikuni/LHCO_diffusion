import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
#import horovod.tensorflow.keras as hvd
import utils
from deepsets_cond import DeepSetsAtt, Resnet
from tensorflow.keras.activations import swish, relu

# tf and friends
tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,name='SGM',npart=100,config=None):
        super(GSGM, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")



        self.activation = swish
        # self.activation = relu
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.num_steps = self.config['MAX_STEPS']
        self.ema=0.999
        # self.train_jet = True #Train only the jet generation first before training the particle generation
        
        
                
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank

        
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None,1)) #mask to identify zero-padded objects
        
        #Time embedding
        graph_conditional = self.Embedding(inputs_time,self.projection)
        jet_conditional = self.Embedding(inputs_time,self.projection)
        #embedding_conditional = self.Embedding(inputs_cond,self.projection)

        #Conditional jet inputs
        jet_dense = layers.Dense(2*self.num_embed)(inputs_jet)
        jet_dense = self.activation(layers.Dense(self.num_embed)(jet_dense))

        #Conditional mjj values
        ff_cond = self.FF(inputs_cond)
        cond_dense = layers.Dense(2*self.num_embed)(ff_cond)
        cond_dense = self.activation(layers.Dense(self.num_embed)(cond_dense))
        
        
        graph_conditional = layers.Dense(3*self.num_embed,activation=None)(tf.concat(
            [graph_conditional,jet_dense,cond_dense],-1))
        graph_conditional=self.activation(graph_conditional)


        jet_conditional = layers.Dense(2*self.num_embed,activation=None)(tf.concat(
            [jet_conditional,cond_dense],-1))
        jet_conditional=self.activation(jet_conditional)

        
        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding=graph_conditional,
            num_heads=2,
            num_transformer = 6,
            projection_dim = 128,
            mask = inputs_mask,
        )
        

        self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_jet,inputs_cond,inputs_mask],
                                      outputs=outputs)

        

        inputs,outputs = DeepSetsAtt(
            num_feat=self.num_jet,
            time_embedding=jet_conditional,
            num_heads=2,
            num_transformer = 6,
            projection_dim = 128,
            mask = None,
        )
        
        
        self.model_jet = keras.Model(inputs=[inputs,inputs_time,inputs_cond],outputs=outputs)

            
        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_part = keras.models.clone_model(self.model_part)
        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        #half_dim = 16
        half_dim = self.num_embed // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq


    def Embedding(self,inputs,projection):
        angle = inputs*projection*1000
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return self.activation(embedding)

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    
    def FF(self,features):
        #Gaussian features to the inputs
        max_proj = 8
        min_proj = 6
        freq = tf.range(start=min_proj, limit=max_proj, dtype=tf.float32)
        freq = 2.**(freq) * 2 * np.pi        

        x = features
        freq = tf.tile(freq[None, :], ( 1, tf.shape(x)[-1]))  
        h = tf.repeat(x, max_proj-min_proj, axis=-1)
        angle = h*freq
        h = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        return tf.concat([features,h],-1)

    @tf.function
    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))
    
    @tf.function
    def get_logsnr_alpha_sigma(self,time):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        
        return logsnr, tf.reshape(alpha,self.shape), tf.reshape(sigma,self.shape)


    # def train_step(self, inputs):
    #     if self.train_jet:
    #         return self.train_jet_step(inputs)
    #     else:
    #         return self.train_part_step(inputs)

    # def test_step(self, inputs):
    #     if self.train_jet:
    #         return self.test_jet_step(inputs)
    #     else:
    #         return self.test_part_step(inputs)



    @tf.function
    def train_step(self, inputs):
        part,jet,cond,mask = inputs
        part = part*mask
        
        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)


        with tf.GradientTape() as tape:
            #jet
            z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
            perturbed_x = alpha*jet + z * sigma
            pred = self.model_jet([perturbed_x, random_t,cond])
            v = alpha * z - sigma * jet
            losses = tf.square(pred - v)
            loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))

        trainable_variables = self.model_jet.trainable_variables
        g = tape.gradient(loss_jet, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))                    
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        #Split the 2 jets apart
        num_part = tf.shape(part)[2]        
        part = tf.reshape(part,(-1,num_part,self.num_feat))
        mask = tf.reshape(mask,(-1,num_part,1))
        jet = tf.reshape(jet,(-1,self.num_jet))
        cond = tf.concat([cond,cond],0)

        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        
        # print(part.shape,mask.shape,jet.shape,cond.shape)
            
        with tf.GradientTape() as tape:
            #part
            z = tf.random.normal((tf.shape(part)),dtype=tf.float32)*mask
            # print(alpha.shape,part.shape , z.shape , sigma.shape)
            perturbed_x = alpha*part + z * sigma
            pred = self.model_part([perturbed_x, random_t,jet,cond,mask])
            
            v = alpha * z - sigma * part
            losses = tf.square(pred - v)*mask
            
            loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        trainable_variables = self.model_part.trainable_variables
        g = tape.gradient(loss_part, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        

            
        self.loss_tracker.update_state(loss_jet + loss_part)
        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }


    @tf.function
    def test_step(self, inputs):
        part,jet,cond,mask = inputs
        part = part*mask
        
        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)



        #jet
        z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        perturbed_x = alpha*jet + z * sigma
        pred = self.model_jet([perturbed_x, random_t,cond])
        v = alpha * z - sigma * jet
        losses = tf.square(pred - v)
        loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))

        #Split the 2 jets apart
        num_part = tf.shape(part)[2]        
        part = tf.reshape(part,(-1,num_part,self.num_feat))
        mask = tf.reshape(mask,(-1,num_part,1))
        jet = tf.reshape(jet,(-1,self.num_jet))
        cond = tf.concat([cond,cond],0)

        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        

            
        #part
        z = tf.random.normal((tf.shape(part)),dtype=tf.float32)*mask
        # print(alpha.shape,part.shape , z.shape , sigma.shape)
        perturbed_x = alpha*part + z * sigma
        pred = self.model_part([perturbed_x, random_t,jet,cond,mask])
        
        v = alpha * z - sigma * part
        losses = tf.square(pred - v)*mask
        
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        self.loss_tracker.update_state(loss_jet + loss_part)
        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }


            
    @tf.function
    def call(self,x):        
        return self.model(x)

    def generate(self,cond,jets):
        start = time.time()
        jets = self.DDPMSampler(cond,self.ema_jet,
                                data_shape=[cond.shape[0],2,self.num_jet],
                                const_shape = self.shape).numpy()
        # end = time.time()
        # print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))

        particles = []
        for ijet in range(2):
            jet_info = jets[:,ijet]
            nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1],self.max_part,norm=self.config['NORM']),
                                            1,self.max_part),-1)
            #print(np.unique(nparts))
            mask = np.expand_dims(
                np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)
        
            assert np.sum(np.sum(mask.reshape(mask.shape[0],-1),-1,keepdims=True)-nparts)==0, 'ERROR: Particle mask does not match the expected number of particles'

            #start = time.time()
            parts = self.DDPMSampler(tf.convert_to_tensor(cond,dtype=tf.float32),
                                     self.ema_part,
                                     data_shape=[cond.shape[0],self.max_part,self.num_feat],
                                     jet=tf.convert_to_tensor(jet_info, dtype=tf.float32),
                                     const_shape = self.shape,
                                     mask=tf.convert_to_tensor(mask, dtype=tf.float32),
                                     ).numpy()
            particles.append(parts*mask)
            # parts = np.ones(shape=(cond.shape[0],self.max_part,3))
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))

            
        return np.stack(particles,1),jets



    @tf.function
    def second_order_correction(self,time_step,x,pred_images,pred_noises,
                                alphas,sigmas,
                                cond,model,jet=None,mask=None,
                                second_order_alpha=0.5):
        step_size = 1.0/self.num_steps
        _, alpha_signal_rates, alpha_noise_rates = self.get_logsnr_alpha_sigma(time_step - second_order_alpha * step_size)
        alpha_noisy_images = alpha_signal_rates * pred_images + alpha_noise_rates * pred_noises

        if jet is None:
            score = model([alpha_noisy_images, time_step - second_order_alpha * step_size,
                           cond],training=False)
        else:
            alpha_noisy_images *= mask
            score = model([alpha_noisy_images, time_step - second_order_alpha * step_size,
                           jet,cond,mask],training=False)*mask

        alpha_pred_noises = alpha_noise_rates * alpha_noisy_images + alpha_signal_rates * score

        # linearly combine the two noise estimates
        pred_noises = (1.0 - 1.0 / (2.0 * second_order_alpha)) * pred_noises + 1.0 / (
            2.0 * second_order_alpha
        ) * alpha_pred_noises

        mean = (x - sigmas * pred_noises) / alphas        
        eps = pred_noises

        
        return mean,eps    
        
    

    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    jet=None,mask=None,
                    clip=False,second_order=True):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        
        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)

        
        for time_step in tf.range(self.num_steps, 0, delta=-1):
            random_t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps)

            
            if jet is None:
                score = model([x, random_t,cond],training=False)
            else:
                x *= mask
                score = model([x, random_t,jet,cond,mask],training=False)*mask
                            
            mean = alpha * x - sigma * score
            eps = sigma * x + alpha * score            

            mean,eps = self.second_order_correction(random_t,x,mean,eps,
                                                    alpha,sigma,
                                                    cond,model,jet,mask
                                                    )
            
            x = alpha_ * mean + sigma_ * eps
            
        # The last step does not include any noise
        return mean        


