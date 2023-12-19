# Define the RNN model and task related function

import os
import numpy as np
import tensorflow as tf
import scipy.io


# define the RNN class respecting Dale's law
class RNN_Dale:
    def __init__(self, hp):
        # load the hyperparameters
        self.task = hp['task']        
        self.with_STP_flag = hp['with_STP_flag']
        self.train_w_flag = hp['train_w_flag']
        self.train_STP_flag = hp['train_STP_flag']
        self.train_wout_flag = hp['train_wout_flag']
        self.train_win_flag = hp['train_win_flag'] 
        
        self.N = hp['N']
        self.num_in = hp['num_In']
        self.num_out = hp['num_Out']
        self.P_inh = hp['P_inh']
        self.P_rec = hp['P_rec']
        self.w_dist = hp['w_dist']
        self.gain = hp['gain']
        self.apply_dale = hp['apply_dale']
        
        self.sigma = hp['sigma']
        
        self.b_out0 = hp['b_out0']
        self.b_rec0 = hp['b_rec0']
        
        # initialize STP parameters
        if hp['sameIni_flag']:
            self.tau_f0 = np.ones([self.N,1])*1.0
            self.tau_d0 = np.ones([self.N,1])*1.0
            self.U0 = np.ones([self.N,1])*0.5
            
            self.tau_f_i0 = np.ones([self.num_in,1])*1.0
            self.tau_d_i0 = np.ones([self.num_in,1])*1.0
            self.U_i0 = np.ones([self.num_in,1])*0.5
        else:
            self.tau_f0 = np.random.randn(self.N,1)*hp['stp_mean']*hp['stp_var'] + hp['stp_mean']
            self.tau_f0 = np.clip(self.tau_f0, 0.1, 3) # 0.1-3 for default
            self.tau_d0 = np.random.randn(self.N,1)*hp['stp_mean']*hp['stp_var'] + hp['stp_mean']
            self.tau_d0 = np.clip(self.tau_d0, 0.1, 3)
            self.U0 = np.random.randn(self.N,1)*0.5*hp['stp_var'] +0.5
            self.U0 = np.clip(self.U0, 0.001, 0.99)            
        
        self.batch_size = hp['batch_size']
                
        self.T = hp['T']
        self.dt = hp['dt']
        self.tau_rnn = hp['tau_rnn']
        self.activation = hp['activation'] 
        
        # Assign each unit as excitatory or inhibitory
        inh, exc, NI, NE,  = self.assign_exc_inh()
        self.inh = inh
        self.exc = exc
        self.NI = NI
        self.NE = NE
        
        self.normal_variable = []
        
        # Initialize the weight matrix
        self.w_in0, self.w0, self.w_out0, self.mask = self.initialize_W()
        

        # define variables
        self.w = tf.Variable(self.w0, dtype=tf.float32, trainable=self.train_w_flag)
        self.m = tf.Variable(self.mask, dtype=tf.float32, trainable=False)
        self.w_in = tf.Variable(self.w_in0, dtype=tf.float32, trainable=self.train_win_flag)        
        self.w_out = tf.Variable(self.w_out0, dtype=tf.float32, trainable=self.train_wout_flag)
        
        self.b_out = tf.Variable(self.b_out0, dtype=tf.float32, name='b_out', trainable=True)
        self.b_rec = tf.Variable(self.b_rec0, dtype=tf.float32, name='b_rec', trainable=False)        
        self.tau_f = tf.Variable(self.tau_f0, dtype=tf.float32, name='tau_f', trainable=self.train_STP_flag, constraint=lambda x:tf.clip_by_value(x, 0.1, 3))
        self.tau_d = tf.Variable(self.tau_d0, dtype=tf.float32, name='tau_d', trainable=self.train_STP_flag, constraint=lambda x:tf.clip_by_value(x, 0.1, 3))
        self.U = tf.Variable(self.U0, dtype=tf.float32, name='U', trainable=self.train_STP_flag,constraint=lambda x:tf.clip_by_value(x, 0.001, 0.99))
                  
        self.train_var = [self.b_out]
        if self.train_w_flag:
            self.train_var.append(self.w)
            
        if self.train_win_flag:
            self.train_var.append(self.w_in)
            
        if self.train_wout_flag:
            self.train_var.append(self.w_out)
            
        if self.train_STP_flag:
            self.train_var.append(self.tau_f)
            self.train_var.append(self.tau_d)
            self.train_var.append(self.U)            

        self.tf_var = [self.w,self.w_out,self.b_out,self.tau_f,self.tau_d,self.U,self.m,self.w_in]

    def assign_exc_inh(self):

        # Apply Dale's principle
        if self.apply_dale == True:
            exc = np.zeros((self.N,1), dtype = int)
            exc[0:np.int32(self.N*(1-self.P_inh)),0] = 1
            inh = 1 - exc
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI
        else:
            inh = np.random.rand(self.N, 1) < 0 # no separate inhibitory units
            exc = ~inh
            NI = len(np.where(inh == True)[0])
            NE = self.N - NI
        return inh, exc, NI, NE

    def initialize_W(self):
        # Weight matrix
        w0 = np.zeros((self.N, self.N), dtype = np.float32)
        idx = np.where(np.random.rand(self.N, self.N) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w0[idx[0], idx[1]] = np.random.gamma(0.1, 1, len(idx[0]))*1 # *0.25 for without stp
        elif self.w_dist.lower() == 'gaus':
            w0[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0]))
            if self.P_rec > 0:
                w0 = w0/np.sqrt(self.N*self.P_rec)*self.gain # scale by a gain to make it chaotic

        if self.apply_dale == True:
            w0 = np.abs(w0)
        w0[np.diag_indices(self.N,2)] = 0.
        # Mask matrix
        mask = np.eye((self.N), dtype=np.float32)
        mask[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = -1
        
        mask_ini = np.eye((self.N), dtype=np.float32)
        mask_ini[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = ((1-self.P_inh)/self.P_inh)
        w0 = np.matmul(w0,mask_ini) # set E/I balance to w
        
        w_out0 = np.float32(np.random.randn(self.num_out,self.N))/np.sqrt(self.N)*0
        w_in0 = np.float32(np.random.gamma(0.1, 1, [self.N, self.num_in])) 
                       
        return w_in0, w0, w_out0,  mask
      
    def save(self, savepath, filename):
        file = savepath +'\\'+ filename +'.npy'
        VAR = []
        for i in range(len(self.tf_var)):
            VAR.append(self.tf_var[i].numpy())
        np.save(file,VAR)

    def load(self, loadpath, filename):
        file = loadpath +'\\'+ filename +'.npy'
        temp = np.load(file,allow_pickle=True)
        for i in range(len(self.tf_var)):
            self.tf_var[i].assign(temp[i])
            
        return self

    def __call__(self,stim,label, target, target_mask):

        x = [] # synaptic currents
        r = [] # firing-rates
    
        out = [] #output
        out_net = [] # output currents
    
        Syn_d = [] # depression variable x
        Syn_f = [] # facilitation variable u
        
        
        x.append(tf.random.normal([self.N, self.batch_size], dtype=tf.float32)*self.sigma)
        if self.activation == 'sigmoid':
            r.append(tf.sigmoid(x[0]))
        elif self.activation == 'relu':
            r.append(tf.nn.relu(x[0] + self.b_rec))
        elif self.activation == 'softplus':
            r.append(tf.clip_by_value(tf.nn.softplus(x[0]+ self.b_rec), 0, 20))              
            
        out_net.append(tf.matmul(self.w_out, r[0])) 
        out.append(out_net[0] + self.b_out)
                
        #initialize the STP variable
        syn_d = tf.ones([self.N,self.batch_size], dtype=tf.float32)
        syn_f = self.U*label[-1]*tf.ones([self.N,self.batch_size], dtype=tf.float32)
        Syn_d.append(syn_d)
        Syn_f.append(syn_f)
        
        if self.apply_dale == True:
            ww = tf.nn.relu(self.w) # Parametrize the weight matrix to enforce exc/inh synaptic currents
        else:
            ww = self.w
        ww_in =  tf.nn.relu(self.w_in)             
        ww = tf.matmul(ww, self.m)
        tau_ff = self.tau_f*1000 # change to ms
        tau_dd = self.tau_d*1000 # change to ms
        
        # time loop        
        for t in range(1, self.T):            
            if self.with_STP_flag:
                syn_d += (1 - Syn_d[t-1])*self.dt/tau_dd - self.dt/1000*Syn_f[t-1] * Syn_d[t-1] * r[t - 1]     
                syn_f += (self.U*label[-1] - Syn_f[t-1])*self.dt/tau_ff + self.dt/1000*self.U*label[-1] * (1 - Syn_f[t-1]) * r[t - 1]
                
                next_x = tf.multiply((1 - self.dt/self.tau_rnn), x[t-1]) + \
                        tf.multiply( self.dt/self.tau_rnn, tf.matmul(ww_in, stim[:, :, t]) +  \
                        tf.matmul(ww, Syn_d[t-1]*Syn_f[t-1]*r[t-1]) + \
                        tf.random.normal([self.N, self.batch_size], dtype=tf.float32)*tf.math.sqrt(2*self.tau_rnn/self.dt)*self.sigma)
            else:
                next_x = tf.multiply((1 - self.dt/self.tau_rnn), x[t-1]) + \
                        tf.multiply( self.dt/self.tau_rnn, tf.matmul(ww_in, stim[:, :, t]) +  \
                        tf.matmul(ww, r[t-1]*self.U*label[-1]) + \
                        tf.random.normal([self.N, self.batch_size], dtype=tf.float32)*tf.math.sqrt(2*self.tau_rnn/self.dt)*self.sigma)                
    
            x.append(next_x)
            Syn_d.append(syn_d)
            Syn_f.append(syn_f)
            
            if self.activation == 'sigmoid':
                r.append(tf.sigmoid(next_x))
            elif self.activation == 'relu':
                r.append(tf.nn.relu(next_x + self.b_rec))
            elif self.activation == 'softplus':
                r.append(tf.clip_by_value(tf.nn.softplus(next_x + self.b_rec), 0, 20))

            next_o = tf.matmul(self.w_out, r[t])
                
            out_net.append(next_o)
            out.append(next_o + self.b_out)
            
        out = tf.stack(out, axis=2)
        out_net = tf.stack(out_net, axis=2);
        
        Syn_d= tf.stack(Syn_d, axis=2)
        Syn_f= tf.stack(Syn_f, axis=2)
                
        r= tf.stack(r, axis=2)
    
        loss = tf.reduce_mean(tf.square(out - target)*target_mask)
        loss = tf.sqrt(loss)
        
        return r, out, out_net, loss, Syn_d, Syn_f




'''
Task-related input signals
'''

def generate_input_stim_readysetgoContext(hp):
    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    delays = hp['delays']
    
    if np.random.rand()>0.5:
        context = hp['context'][0]
    else:
        context = hp['context'][1]
            
    input = np.zeros((num_In,batch_size, T))
    label = []
    for i in range(batch_size):
        # delay = np.random.rand()*(delay_max - delay_min) + delay_min
        delay = np.random.choice(delays,1)
        delay = np.int(delay)
        input[0,i,stim_on:stim_on+stim_dur] = 1
        input[0,i,stim_on+stim_dur + delay:stim_on+stim_dur + delay + stim_dur] = 1
        label.append(delay)
        
    label.append(context)

    return np.float32(input), np.float32(np.array(label))


def generate_input_stim_iafc(hp):

    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    delays = hp['delays']

    input = np.zeros((num_In,batch_size, T))
    label = []
    for i in range(batch_size):
        delay = delays[np.random.choice(len(delays))]
        input[0,i,stim_on:stim_on+stim_dur] = 1
        input[0,i,stim_on+stim_dur + delay:stim_on+stim_dur + delay + stim_dur] = 1
        label.append(delay)
        
    label.append(hp['context'])        
    return np.float32(input), label

def generate_input_stim_motorTraj(hp):

    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    N = hp['N']
    time_ind = hp['time_ind']
    space_ind = hp['space_ind']
    
    temp = np.zeros((N,batch_size),dtype=np.float32)
  
    input = np.zeros((num_In,batch_size, T))
    label = []
        
    for i in range(batch_size):        
        Context = np.ones((N),dtype=np.float32)   
        digit = np.random.choice(10)
        input[digit,i,stim_on:stim_on+stim_dur] = 1
        
        if np.random.rand() < 0.5:
            Context[time_ind] = hp['context_time'][0]
            if np.random.rand() < 0.5:
                Context[space_ind] = hp['context_space'][1]
                label.append(['sh_sm',digit])
            else:
                Context[space_ind] = hp['context_space'][0]
                label.append(['sh_la',digit]) 
        else:
            Context[time_ind] = hp['context_time'][1]
            if np.random.rand() < 0.5:
                Context[space_ind] = hp['context_space'][1]
                label.append(['lo_sm',digit])
            else:
                Context[space_ind] = hp['context_space'][0]
                label.append(['lo_la',digit])
        temp[:,i] = Context  
                      
    label.append(temp)
    return np.float32(input), label


def generate_input_stim_motorTraj_time(hp):

    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    N = hp['N']
    time_ind = hp['time_ind']

    temp = np.zeros((N,batch_size),dtype=np.float32)
  
    input = np.zeros((num_In,batch_size, T))
    label = []
            
    for i in range(batch_size):
        digit = np.random.choice(10)
        input[digit,i,stim_on:stim_on+stim_dur] = 1 # initial state        
        Context = np.ones((N),dtype=np.float32)                    
        if np.random.rand() < 0.5:
            Context[time_ind] = hp['context_time'][0]
            label.append(['sh',digit])
        else:
            Context[time_ind] = hp['context_time'][1]
            label.append(['lo',digit])
        temp[:,i] = Context  
                      
    label.append(temp)
    return np.float32(input), label

def generate_input_stim_motorTraj_space(hp):

    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    N = hp['N']
    space_ind = hp['space_ind']

    temp = np.zeros((N,batch_size),dtype=np.float32)
  
    input = np.zeros((num_In,batch_size, T))
    label = []
        
    for i in range(batch_size):
        digit = np.random.choice(10)
        input[digit,i,stim_on:stim_on+stim_dur] = 1        
        
        Context = np.ones((N),dtype=np.float32)        
        
        if np.random.rand() < 0.5:
            Context[space_ind] = hp['context_space'][1]
            label.append(['sm',digit])
        else:
            Context[space_ind] = hp['context_space'][0]
            label.append(['la',digit]) 

        temp[:,i] = Context  
                      
    label.append(temp)
    return np.float32(input), label

def generate_input_stim_motorTraj_digit(hp):

    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    N = hp['N']
    time_ind = hp['time_ind']
    space_ind = hp['space_ind']
    digit_ind = hp['digit_ind']

    temp = np.zeros((N,batch_size),dtype=np.float32)
  
    input = np.zeros((num_In,batch_size, T))
    label = []
        
    for i in range(batch_size):
        input[0,i,stim_on:stim_on+stim_dur] = 1 # initial state        
        Context = np.ones((N),dtype=np.float32)
        Context[digit_ind] = hp['context_digit'][0]        
        digit = np.random.choice(10)
        Context[digit_ind[np.int32(digit*np.size(digit_ind)/10):np.int32((digit+1)*np.size(digit_ind)/10)]] = hp['context_digit'][1]
        
        if np.random.rand() < 0.5:
            Context[time_ind] = hp['context_time'][0]
            if np.random.rand() < 0.5:
                Context[space_ind] = hp['context_space'][1]
                label.append(['sh_sm',digit])
            else:
                Context[space_ind] = hp['context_space'][0]
                label.append(['sh_la',digit]) 
        else:
            Context[time_ind] = hp['context_time'][1]
            if np.random.rand() < 0.5:
                Context[space_ind] = hp['context_space'][1]
                label.append(['lo_sm',digit])
            else:
                Context[space_ind] = hp['context_space'][0]
                label.append(['lo_la',digit])
        temp[:,i] = Context  
                      
    label.append(temp)
    return np.float32(input), label


def generate_input_stim_motorTraj_time_inputScale(hp):

    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    N = hp['N']
    time_ind = hp['time_ind']

    temp = np.zeros((N,batch_size),dtype=np.float32)
  
    input = np.zeros((num_In,batch_size, T))
    label = []
            
    for i in range(batch_size):
        digit = np.random.choice(10)
        input[digit,i,stim_on:stim_on+stim_dur] = 1 # initial state        
        Context = np.ones((N),dtype=np.float32)                    
        if np.random.rand() < 0.5:
            Context[time_ind] = 0.85
            label.append(['sh',digit])
            input[10,i,:] = hp['context_time'][0]

        else:
            Context[time_ind] = 0.85
            label.append(['lo',digit])
            input[10,i,:] = hp['context_time'][1]
        temp[:,i] = Context  
                      
    label.append(temp)
    return np.float32(input), label

def generate_input_stim_motorTraj_space_inputScale(hp):

    num_In      = hp['num_In']
    batch_size  = hp['batch_size']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    N = hp['N']
    space_ind = hp['space_ind']

    temp = np.zeros((N,batch_size),dtype=np.float32)
  
    input = np.zeros((num_In,batch_size, T))
    label = []
        
    for i in range(batch_size):
        digit = np.random.choice(10)
        input[digit,i,stim_on:stim_on+stim_dur] = 1        
        
        Context = np.ones((N),dtype=np.float32)        
        
        if np.random.rand() < 0.5:
            Context[space_ind] = 0.85
            label.append(['sm',digit])
            input[10,i,:] = hp['context_space'][1]
        else:
            Context[space_ind] = 0.85
            label.append(['la',digit])
            input[10,i,:] = hp['context_space'][0]

        temp[:,i] = Context  
                      
    label.append(temp)
    return np.float32(input), label




'''
Task-related target signals
'''

def generate_target_continuous_readysetgoContext(hp, label):
    batch_size  = hp['batch_size']
    num_Out = hp['num_Out']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    g = hp['g']    
    target      = np.zeros((num_Out,batch_size, T))
    target_mask = np.zeros((num_Out,batch_size, T))
    
    if label[-1] == hp['context'][0]:
        for i in range(batch_size):
            delay = np.int32(label[i])     
            target[0,i, stim_on + stim_dur*2 + delay:stim_on + stim_dur*2 + delay*2 + np.int32(delay* hp['resp_ratio'])] = np.linspace(0,1, delay + np.int32(delay* hp['resp_ratio']))            
            target_mask[0,i,0:stim_on + stim_dur*2 + delay*2 + np.int32(delay* hp['resp_ratio'])] =1 
            
    else:
        for i in range(batch_size):
            delay = np.int32(label[i])
            delay2 = np.int32(delay*g)
            target[0,i, stim_on + stim_dur*2 + delay:stim_on + stim_dur*2 + delay +delay2 + np.int32(delay2* hp['resp_ratio'])] = np.linspace(0,1, delay2 + np.int32(delay2* hp['resp_ratio']))            
            target_mask[0,i,0:stim_on + stim_dur*2 + delay +delay2 + np.int32(delay2* hp['resp_ratio'])] =1           

    return np.float32(np.squeeze(target)), np.float32(np.squeeze(target_mask))

def generate_target_continuous_iafc(hp, label):
    """
    """
    batch_size  = hp['batch_size']
    num_Out = hp['num_Out']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    resp_dur = hp['resp_dur']
    Thr = hp['Thr']
    
    target      = np.zeros((num_Out,batch_size, T))
    target_mask = np.zeros((num_Out,batch_size, T))
    
    for i in range(batch_size):
        delay = label[i]
        if delay < Thr:
            target[0,i, stim_on + stim_dur*2 + delay:stim_on + stim_dur*2 + delay + resp_dur] = 1
            target_mask[0,i,0:stim_on + stim_dur*2 + delay-1] =1
            target_mask[0,i,stim_on + stim_dur*2 + delay+2:stim_on + stim_dur*2 + delay + resp_dur] =1
            target_mask[1,i,:stim_on + stim_dur*2 + delay + resp_dur] =1
            
        else:
            target[1,i, stim_on + stim_dur*2 + delay:stim_on + stim_dur*2 + delay + resp_dur] = 1
            target_mask[1,i,0:stim_on + stim_dur*2 + delay-1] =1
            target_mask[1,i,stim_on + stim_dur*2 + delay+2:stim_on + stim_dur*2 + delay + resp_dur] =1
            target_mask[0,i,:stim_on + stim_dur*2 + delay + resp_dur] =1            

    return np.float32(np.squeeze(target)), np.float32(np.squeeze(target_mask))

def generate_target_continuous_motorTraj(hp, label):
    batch_size  = hp['batch_size']
    num_out = hp['num_Out']
    interval1 = hp['interval1']
    interval2 = hp['interval2']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    dt = hp['dt']
    
    target      = np.zeros((num_out,batch_size, T))
    target_mask = np.zeros((num_out,batch_size, T))
    
    file = os.path.join(os.getcwd(), 'digitMaster.mat')
    temp = scipy.io.loadmat(file)
    temp = temp['digitMaster']
        
    for i in range(batch_size):
        digit = label[i][1]
        temp1 = temp[digit*2:digit*2+2,:]
        if label[i][0] == 'sh_sm':
            interval = interval1
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,::dt*2]
            target_mask[:,i,:stim_on + stim_dur + interval] =1 
        elif label[i][0] == 'sh_la':
            interval = interval1
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,::dt*2]*hp['space_scale']
            target_mask[:,i,:stim_on + stim_dur + interval] =1  
        elif label[i][0] == 'lo_sm':
            interval = interval2
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,np.int32(np.linspace(0,2000-1,num = np.int32(hp['interval1']*hp['time_scale'])))]
            target_mask[:,i,:stim_on + stim_dur + interval] =1 
        elif label[i][0] == 'lo_la':
            interval = interval2
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,np.int32(np.linspace(0,2000-1,num = np.int32(hp['interval1']*hp['time_scale'])))]*hp['space_scale']
            target_mask[:,i,:stim_on + stim_dur + interval] =1  
                
    return np.float32(np.squeeze(target)), np.float32(np.squeeze(target_mask))

def generate_target_continuous_motorTraj_time(hp, label):
    batch_size  = hp['batch_size']
    num_out = hp['num_Out']
    interval1 = hp['interval1']
    interval2 = hp['interval2']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    dt = hp['dt']
    
    target      = np.zeros((num_out,batch_size, T))
    target_mask = np.zeros((num_out,batch_size, T))
    
    file = os.path.join(os.getcwd(), 'digitMaster.mat')
    temp = scipy.io.loadmat(file)
    temp = temp['digitMaster']
    
    
    for i in range(batch_size):
        digit = label[i][1]
        temp1 = temp[digit*2:digit*2+2,:]
        if label[i][0] == 'sh':
            interval = interval1
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,::dt*2]
            target_mask[:,i,:stim_on + stim_dur + interval] =1 
        elif label[i][0] == 'lo':
            interval = interval2
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,np.int32(np.linspace(0,2000-1,num = np.int32(hp['interval1']*hp['time_scale'])))]
            target_mask[:,i,:stim_on + stim_dur + interval] =1 
                
    return np.float32(np.squeeze(target)), np.float32(np.squeeze(target_mask))

def generate_target_continuous_motorTraj_space(hp, label):
    batch_size  = hp['batch_size']
    num_out = hp['num_Out']
    interval1 = hp['interval1']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    dt = hp['dt']
    
    target      = np.zeros((num_out,batch_size, T))
    target_mask = np.zeros((num_out,batch_size, T))
    
    file = os.path.join(os.getcwd(), 'digitMaster.mat')
    temp = scipy.io.loadmat(file)
    temp = temp['digitMaster']
       
    for i in range(batch_size):
        digit = label[i][1]
        temp1 = temp[digit*2:digit*2+2,:]
        if label[i][0] == 'sm':
            interval = interval1
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,::dt*2]
            target_mask[:,i,:stim_on + stim_dur + interval] =1 
        elif label[i][0] == 'la':
            interval = interval1
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,::dt*2]*hp['space_scale']
            target_mask[:,i,:stim_on + stim_dur + interval] =1  
                
    return np.float32(np.squeeze(target)), np.float32(np.squeeze(target_mask))

def generate_target_continuous_motorTraj_digit(hp, label):
    batch_size  = hp['batch_size']
    num_out = hp['num_Out']
    interval1 = hp['interval1']
    interval2 = hp['interval2']
    T = hp['T']
    stim_on = hp['stim_on']
    stim_dur = hp['stim_dur']
    dt = hp['dt']
    
    target      = np.zeros((num_out,batch_size, T))
    target_mask = np.zeros((num_out,batch_size, T))
    
    file = os.path.join(os.getcwd(), 'digitMaster.mat')
    temp = scipy.io.loadmat(file)
    temp = temp['digitMaster']
        
    for i in range(batch_size):
        digit = label[i][1]
        temp1 = temp[digit*2:digit*2+2,:]
        if label[i][0] == 'sh_sm':
            interval = interval1
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,::dt*2]
            target_mask[:,i,:stim_on + stim_dur + interval] =1 
        elif label[i][0] == 'sh_la':
            interval = interval1
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,::dt*2]*hp['space_scale']
            target_mask[:,i,:stim_on + stim_dur + interval] =1  
        elif label[i][0] == 'lo_sm':
            interval = interval2
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,np.int32(np.linspace(0,2000-1,num = np.int32(hp['interval1']*hp['time_scale'])))]
            target_mask[:,i,:stim_on + stim_dur + interval] =1 
        elif label[i][0] == 'lo_la':
            interval = interval2
            target[:,i,  stim_on + stim_dur:stim_on + stim_dur + interval] =  temp1[:,np.int32(np.linspace(0,2000-1,num = np.int32(hp['interval1']*hp['time_scale'])))]*hp['space_scale']
            target_mask[:,i,:stim_on + stim_dur + interval] =1  
                
    return np.float32(np.squeeze(target)), np.float32(np.squeeze(target_mask))