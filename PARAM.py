# codes to train RNNs to do temporal and spatial scaling task as in the paper 
#'Unified control of temporal and spatial scales of sensorimotor behavior through neuromodulation of short-term synaptic plasticity'
# By Shanglin Zhou and Dean Buonomano
# This code is meant to incorprate the hyperparameter search. For a specific set of hyperparameters, just run the code after the second 'for' loop with hyperparameter specified in the 'hp' variable



import numpy as np
import tensorflow as tf

import time
import datetime
import scipy.io
import os



numRep = 10# number of RNNs for a given set of parameter
Para = [0] # candidate hyperparamter; set to 0 for a preconfigured set of hyperparameters
numPara = len(Para)

for paraInd in range(numPara):
    para = Para[paraInd]
    for repInd in range(0,numRep):
        
        # Import the RNN model
        from model import RNN_Dale
        
        # Import the tasks        
        from model import generate_input_stim_readysetgoContext  # for ready-set-go task
        from model import generate_target_continuous_readysetgoContext
        
        from model import generate_input_stim_iafc  # for IAFC task
        from model import generate_target_continuous_iafc       
        
        from model import generate_input_stim_motorTraj # for joint control temporal and spatial scales with input-cued-digit setting
        from model import generate_target_continuous_motorTraj  
        
        from model import generate_input_stim_motorTraj_time # for temporal scaling task with input-cued-digit setting
        from model import generate_target_continuous_motorTraj_time   
        
        from model import generate_input_stim_motorTraj_space # for spatial scaling task with input-cued-digit setting
        from model import generate_target_continuous_motorTraj_space 
        
        from model import generate_input_stim_motorTraj_digit # for joint control temporal and spatial scales with alpha-cued-digit setting
        from model import generate_target_continuous_motorTraj_digit 
        
        from model import generate_input_stim_motorTraj_time_inputScale # for temporal scaling task  with input-cued-digit setting and input-ampplitude-cued-scaling
        
        from model import generate_input_stim_motorTraj_space_inputScale # for spatial scaling task with input-cued-digit setting and input-ampplitude-cued-scaling
        
        # GPU
        tf.config.set_visible_devices([], 'GPU') # comment to use gpu
           
        # set mode to train
        Mode = 'train'  
        
        #Hyperparameter Dict
        hp = {
            'task':     'motorTraj_time',  #readysetgoContext, iafc, motorTraj_time, motorTraj_space, motorTraj, motorTraj_digit, motorTraj_time_inputScale, motorTraj_space_inputScale
            
            'with_STP_flag': True, # True to implement STP in RNN
            'train_STP_flag': False, # False to not train STP param
            'train_w_flag': True, # True to train recurrent weights
            'train_wout_flag': True, # True to output recurrent weights
            'train_win_flag': True, # True to input recurrent weights
            'sameIni_flag': False, # False to allow STP parameters from random distributions
            
            'n_trials': 500000,
            'N':        200,
            'num_In':   2, # number of inputs
            'num_Out':  1,# number of outputs
            'gain':     0.5,  #gain for the connectivity weight initialization, 0.5 defult
            'apply_dale': True, # Ture to allow Exc and Inh weights
            'P_rec':    1,  #connectivity probability
            'P_inh':    0.2,  #proportion of inhibitory units
            'w_dist': 'gamma', #  weight distribution (Gaussian or Gamma)
        
            'activation': 'relu',  #activation function (sigmoid, relu)
        
            'dt':       10,  # discretization time step, ms
            'tau_rnn':  100.0,  # decay time-constants, ms
        
        
            'learning_rate':    0.001,  # learning rate 0.01
            'loss_thr':   0.02,  #loss threshold (when to stop training)
            'perf_thr': 0.85, # task performace threshold, needed for task with decisions, such as in IAFC task
            'eval_freq':        100,  # how often to evaluate task performance
            'eval_tr':          20,  # number of trials for evaluation
        }
        
        
        
        # task dependent parameters        
            
        if hp['task'] == 'readysetgoContext':
            hp['g'] = 1.5 #
            hp['context'] = np.float32([0.9, 0.8])
            hp['num_In']    = 1 
            hp['stim_on'] = np.int32(600/hp['dt'])
            hp['stim_dur'] = np.int32(100/hp['dt'])
            hp['delays'] = np.int32(np.linspace(500,1000,7)/hp['dt'])
            hp['thr_alpha'] =  0.2
            hp['thr_beta'] =  np.int32(25.0/hp['dt'])
            hp['eval_amp_threh'] = 0.75 # amplitude threshold during response window
            hp['resp_ratio'] = ((1-hp['eval_amp_threh'])/hp['eval_amp_threh'])
            
            hp['T']= np.int32(600/hp['dt']) + hp['stim_dur']*2 +hp['delays'][-1]+ np.int32(hp['delays'][-1]*hp['g'] + np.int32(hp['delays'][-1]*hp['g']* hp['resp_ratio'])) # 
            hp['sigma'] = 0.05
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = 0
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 10000
            hp['perf_thr'] = 0.98

        elif hp['task'] == 'iafc':
            hp['num_In']    = 1
            hp['num_Out']    = 2
            hp['stim_on'] = round(600/hp['dt'])
            hp['stim_dur'] = round(150/hp['dt'])
            hp['delays'] = np.int16(np.array([0.6, 1.05, 1.26,1.38,1.62,1.74,1.95, 2.4])*1000/hp['dt'])
            hp['resp_dur'] = round(200/hp['dt'])
            hp['Thr'] = int(1.5*1000/hp['dt'])
            hp['T']= round(600/hp['dt']) + hp['stim_dur']*2 + hp['delays'][-1] + hp['resp_dur'] # trial duration for long interval (in steps)
            hp['sigma'] = 1
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = 0
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 100000
            hp['perf_thr'] = 0.9
            
            hp['context'] = 0.8
            
        elif hp['task'] == 'motorTraj':
            hp['N']  = 400
            hp['context_time'] = np.float32([0.9, 0.8])
            hp['context_space'] = np.float32([0.9, 0.8])
            # hp['context_digit'] = np.float32([1, 0.6])
            
            hp['time_scale'] = 1.5
            hp['space_scale'] = 1.5
            
            hp['time_ind'] = np.random.choice(hp['N'],np.int32(hp['N']/2),replace=False)
            hp['space_ind'] = np.random.choice(np.setdiff1d(np.arange(hp['N']),  hp['time_ind']),np.int32(hp['N']/2),replace=False)

            hp['num_In']    = 10            
            hp['num_Out']    = 2
            hp['stim_on'] = np.int32(600/hp['dt'])
            hp['stim_dur'] = np.int32(100/hp['dt'])
            hp['interval1'] = np.int32(1000/hp['dt'])
            hp['interval2'] = np.int32(1000*hp['time_scale']/hp['dt'])
            
            hp['T']= np.int32(600/hp['dt']) + hp['stim_dur'] + hp['interval2']# trial duration for long interval (in steps)
            
            hp['sigma'] = 0.01
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = np.zeros((hp['num_Out'],1))
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 0.02
            hp['perf_thr'] = 0

        elif hp['task'] == 'motorTraj_time':
            hp['N']  = 200
            hp['context_time'] = np.float32([0.9, 0.8]) # for congruent condition
#            hp['context_time'] = np.float32([0.8, 0.9]) # for incongruent condition         
            hp['time_scale'] = 1.5
#            hp['space_scale'] = 1.5                                    
            hp['time_ind'] = np.random.choice(hp['N'],np.int32(hp['N']),replace=False)
            
            
            hp['num_In']    = 10            
            hp['num_Out']    = 2
            hp['stim_on'] = np.int32(600/hp['dt'])
            hp['stim_dur'] = np.int32(100/hp['dt'])
            hp['interval1'] = np.int32(1000/hp['dt'])
            hp['interval2'] = np.int32(1000*hp['time_scale']/hp['dt'])
            
            hp['T']= np.int32(600/hp['dt']) + hp['stim_dur'] + hp['interval2']# trial duration for long interval (in steps)
            
            hp['sigma'] = 0.01
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = np.zeros((hp['num_Out'],1))
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 0.02
            hp['perf_thr'] = 0
            
        elif hp['task'] == 'motorTraj_space':
            hp['N']  = 200
            hp['context_space'] = np.float32([0.9, 0.8])# for congruent condition
#            hp['context_space'] = np.float32([0.8, 0.9])# for incongruent condition  
            
            hp['space_scale'] = 1.5
                                    
            hp['space_ind'] = np.random.choice(hp['N'],np.int32(hp['N']),replace=False)

            hp['num_In']    = 10            
            hp['num_Out']    = 2
            hp['stim_on'] = np.int32(600/hp['dt'])
            hp['stim_dur'] = np.int32(100/hp['dt'])
            hp['interval1'] = np.int32(1000/hp['dt'])
            
            hp['T']= np.int32(600/hp['dt']) + hp['stim_dur'] + hp['interval1']# trial duration for long interval (in steps)
            
            hp['sigma'] = 0.01
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = np.zeros((hp['num_Out'],1))
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 0.02
            hp['perf_thr'] = 0

        elif hp['task'] == 'motorTraj_digit':
            hp['N']  = 400
            hp['context_time'] = np.float32([0.9, 0.8])
            hp['context_space'] = np.float32([0.9, 0.8])
            hp['context_digit'] = np.float32([1, 0.6])
            
            hp['time_scale'] = 1.5
            hp['space_scale'] = 1.5
            
            hp['time_ind'] = np.random.choice(hp['N'],np.int32(hp['N']/4),replace=False)
            hp['space_ind'] = np.random.choice(np.setdiff1d(np.arange(hp['N']),  hp['time_ind']),np.int32(hp['N']/4),replace=False)
            hp['digit_ind'] = np.setdiff1d(np.arange(hp['N']),  np.union1d(hp['time_ind'],hp['space_ind']))
                        
            hp['num_In']    = 1            
            hp['num_Out']    = 2
            hp['stim_on'] = np.int32(600/hp['dt'])
            hp['stim_dur'] = np.int32(100/hp['dt'])
            hp['interval1'] = np.int32(1000/hp['dt'])
            hp['interval2'] = np.int32(1000*hp['time_scale']/hp['dt'])
            
            hp['T']= np.int32(600/hp['dt']) + hp['stim_dur'] + hp['interval2']# trial duration for long interval (in steps)
            
            hp['sigma'] = 0.01
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = np.zeros((hp['num_Out'],1))
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 0.02
            hp['perf_thr'] = 0
            
        elif hp['task'] == 'motorTraj_time_inputScale':
            hp['N']  = 200
            hp['context_time'] = np.float32([0.9, 0.85]) # 
#            hp['context_time'] = np.float32([0.8, 0.9]) # 
            
            hp['time_scale'] = 1.5
                        
            hp['time_ind'] = np.random.choice(hp['N'],np.int32(hp['N']),replace=False)
            
            hp['num_In']    = 10 + 1           
            hp['num_Out']    = 2
            hp['stim_on'] = np.int32(600/hp['dt'])
            hp['stim_dur'] = np.int32(100/hp['dt'])
            hp['interval1'] = np.int32(1000/hp['dt'])
            hp['interval2'] = np.int32(1000*hp['time_scale']/hp['dt'])
            
            hp['T']= np.int32(600/hp['dt']) + hp['stim_dur'] + hp['interval2']# trial duration for long interval (in steps)
            
            hp['sigma'] = 0.01
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = np.zeros((hp['num_Out'],1))
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 0.02
            hp['perf_thr'] = 0
            
        elif hp['task'] == 'motorTraj_space_inputScale':
            hp['N']  = 200
            hp['context_space'] = np.float32([0.9, 0.7])#
            
            hp['time_scale'] = 1.5
            hp['space_scale'] = 1.5
                        
            hp['space_ind'] = np.random.choice(hp['N'],np.int32(hp['N']),replace=False)
                       
            hp['num_In']    = 10 + 1           
            hp['num_Out']    = 2
            hp['stim_on'] = np.int32(600/hp['dt'])
            hp['stim_dur'] = np.int32(100/hp['dt'])
            hp['interval1'] = np.int32(1000/hp['dt'])
            hp['interval2'] = np.int32(1000*hp['time_scale']/hp['dt'])
            
            hp['T']= np.int32(600/hp['dt']) + hp['stim_dur'] + hp['interval1']# trial duration for long interval (in steps)
            
            hp['sigma'] = 0.01
            hp['b_rec0'] = np.zeros((hp['N'],1))
            hp['b_out0'] = np.zeros((hp['num_Out'],1))
            hp['stp_mean'] = 1
            hp['stp_var'] = 1/3 # times of the stp_mean            
            hp['batch_size'] = 16
            
            hp['loss_thr'] = 0.02
            hp['perf_thr'] = 0  
            
        print(hp)
        
        
        
        rnn = RNN_Dale(hp)
        
        
            
        #Train the network
#        wrap everthing into one function to speed up training
        def train_step(stim,label,target,target_mask):
            with tf.GradientTape(persistent=False) as tape:
                t_r,t_o, t_o_net, t_loss, Syn_d, Syn_f = rnn(stim,label,target,target_mask)
           
            grads = tape.gradient(t_loss, rnn.train_var)
            clipmin = -10
            clipmax = 10
            for i in range(len(grads)):
                grads[i] = tf.where(tf.math.is_nan(grads[i]), 0., grads[i])
                grads[i] = tf.where(tf.math.is_inf(grads[i]), 0., grads[i])                  
                grads[i] = tf.clip_by_value(grads[i],clipmin,clipmax)
        
            optimizer.apply_gradients(zip(grads,rnn.train_var))
            
            del tape
            return  t_r, t_o, t_o_net, t_loss, Syn_d, Syn_f
        
        
        if Mode == 'train':
                print('TRAINING STARTED ...')
                optimizer = tf.keras.optimizers.Adam(learning_rate = hp['learning_rate'])
                # optimizer = tf.keras.optimizers.SGD(learning_rate = hp['learning_rate'])
                training_success = False
        
                losses = np.zeros((hp['n_trials'],))
                perfEvals = np.zeros((hp['n_trials'],));
                lossEvals = np.zeros((hp['n_trials'],));
                evalCount = 0;

                Out = np.zeros((hp['T'], ))
                start_time = time.time()
                
                for tr in range(hp['n_trials']):         
                    if hp['task'] == 'readysetgoContext':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_readysetgoContext(hp)
                        target, target_mask = generate_target_continuous_readysetgoContext(hp,label)  
                    elif hp['task'] == 'iafc':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_iafc(hp)
                        target, target_mask = generate_target_continuous_iafc(hp,label)                         
                    elif hp['task'] == 'motorTraj':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_motorTraj(hp)
                        target, target_mask = generate_target_continuous_motorTraj(hp,label)                         
                    elif hp['task'] == 'motorTraj_time':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_motorTraj_time(hp)
                        target, target_mask = generate_target_continuous_motorTraj_time(hp,label)
                    elif hp['task'] == 'motorTraj_space':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_motorTraj_space(hp)
                        target, target_mask = generate_target_continuous_motorTraj_space(hp,label)
                    elif hp['task'] == 'motorTraj_digit':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_motorTraj_digit(hp)
                        target, target_mask = generate_target_continuous_motorTraj_digit(hp,label)  
                    elif hp['task'] == 'motorTraj_time_inputScale':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_motorTraj_time_inputScale(hp)
                        target, target_mask = generate_target_continuous_motorTraj_time(hp,label) # the target is the same as motorTraj_time
                    elif hp['task'] == 'motorTraj_space_inputScale':
                        hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                        stim,label = generate_input_stim_motorTraj_space_inputScale(hp)
                        target, target_mask = generate_target_continuous_motorTraj_space(hp,label) # the target is the same as motorTraj_space                       
                        
                    # Train using backprop
                    t_r, t_o, t_o_net, t_loss, Syn_d, Syn_f = train_step(stim,label,target,target_mask)
                    
                    losses[tr] = t_loss
               
                    '''
                    Evaluate the model and determine if the training termination criteria are met
                    # '''
                    if (tr-1)%hp['eval_freq'] == 0:
                        eval_perfs = np.zeros((hp['eval_tr'],1))
                        eval_losses = np.zeros((hp['eval_tr'],1))
                        eval_os = np.zeros((hp['eval_tr'],hp["num_Out"],hp['batch_size'], hp['T']))
                        eval_o_nets = np.zeros((hp['eval_tr'],hp["num_Out"],hp['batch_size'], hp['T']))
                        eval_us = np.zeros((hp['eval_tr'],hp["num_In"],hp['batch_size'],hp['T']))
                        eval_zs = np.zeros((hp['eval_tr'],hp["num_Out"], hp['batch_size'],hp['T']))
                        
                        eval_rs = np.zeros((hp['eval_tr'],hp['N'],hp['batch_size'],hp['T']))
                        eval_syn_d_is = np.zeros((hp['eval_tr'],hp['N'],hp['num_In'],hp['batch_size'],hp['T']))
                        eval_syn_f_is = np.zeros((hp['eval_tr'],hp['N'],hp['num_In'],hp['batch_size'],hp['T']))                        
                        
                        for ii in range(hp['eval_tr']):                                                                    
                            if hp['task'] == 'readysetgoContext':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_readysetgoContext(hp)
                                target, target_mask = generate_target_continuous_readysetgoContext(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask)                       

                                stim_on = hp['stim_on']
                                stim_dur = hp['stim_dur']
                              
                                t_perf = np.zeros((1,hp['batch_size']))
                                temp = t_out.numpy()
                                
                                if label[-1] == hp['context'][0]:
                                    for iii in range(hp['batch_size']):
                                        delay = np.int32(label[iii])
                                        thr =  np.int32(hp['thr_alpha']*delay + hp['thr_beta'])
                                        resp = np.squeeze(np.where(temp[0,iii,stim_on:]> hp['eval_amp_threh']))
                                        if resp.size > 1:
                                            if resp[0]>(stim_dur*2 + delay*2) - thr and resp[0]< (stim_dur*2 + delay*2) + thr:
                                                t_perf[0, iii] = 1
                                else:
                                    for iii in range(hp['batch_size']):
                                        delay = np.int32(label[iii])
                                        delay2 = np.int32(delay*hp['g'])
                                        thr =  np.int32(hp['thr_alpha']*delay2 + hp['thr_beta'])
                                        resp = np.squeeze(np.where(temp[0,iii,stim_on:]> hp['eval_amp_threh']))
                                        if resp.size > 1:
                                            if resp[0]>(stim_dur*2 + delay + delay2) - thr and resp[0]< (stim_dur*2 + delay + delay2) + thr:
                                                t_perf[0, iii] = 1                       
                                                
                            elif hp['task'] == 'iafc':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_iafc(hp)
                                target, target_mask = generate_target_continuous_iafc(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask) 
                                
                                stim_on = hp['stim_on']
                                stim_dur = hp['stim_dur']
                                T = hp['T']      
                                resp_dur = hp['resp_dur']
                                Thr = hp['Thr']
                                
                                t_perf = np.zeros((1,hp['batch_size']))
                                temp = np.squeeze(t_out.numpy())
                                
                                for iii in range(hp['batch_size']):
                                    delay = label[iii]                                    
                                    temp1 = np.squeeze(temp[0,iii,stim_on + stim_dur*2 + delay:stim_on + stim_dur*2 + delay + resp_dur])
                                    temp2 = np.squeeze(temp[1,iii,stim_on + stim_dur*2 + delay:stim_on + stim_dur*2 + delay + resp_dur])
                                    resp1 = np.mean(temp1)
                                    resp2 = np.mean(temp2)                                    
                                    if delay < Thr:
                                        if resp1 > resp2:
                                                t_perf[0, iii] = 1
                                    else:
                                        if resp1 < resp2:
                                                t_perf[0, iii] = 1
                                                                                                                                
                            elif hp['task'] == 'motorTraj':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_motorTraj(hp)
                                target, target_mask = generate_target_continuous_motorTraj(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask)   
                                
                                t_perf = np.ones((1,hp['batch_size']))                                
                            elif hp['task'] == 'motorTraj_time':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_motorTraj_time(hp)
                                target, target_mask = generate_target_continuous_motorTraj_time(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask)   
                                
                                t_perf = np.ones((1,hp['batch_size']))                                

                            elif hp['task'] == 'motorTraj_space':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_motorTraj_space(hp)
                                target, target_mask = generate_target_continuous_motorTraj_space(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask)   
                                
                                t_perf = np.ones((1,hp['batch_size']))

                            elif hp['task'] == 'motorTraj_digit':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_motorTraj_digit(hp)
                                target, target_mask = generate_target_continuous_motorTraj_digit(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask)   
                                
                                t_perf = np.ones((1,hp['batch_size']))
                                
                            elif hp['task'] == 'motorTraj_time_inputScale':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_motorTraj_time_inputScale(hp)
                                target, target_mask = generate_target_continuous_motorTraj_time(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask)   
                                
                                t_perf = np.ones((1,hp['batch_size']))                                

                            elif hp['task'] == 'motorTraj_space_inputScale':
                                hp['stim_on'] = np.int32(np.random.random_sample()*400/hp['dt'] + 200/hp['dt'])
                                stim,label = generate_input_stim_motorTraj_space_inputScale(hp)
                                target, target_mask = generate_target_continuous_motorTraj_space(hp,label)
                                t_r, t_out, t_out_net, t_loss,t_syn_d, t_syn_f  = rnn(stim,label,target,target_mask)   
                                
                                t_perf = np.ones((1,hp['batch_size']))
                                
                                
                            eval_losses[ii,0] = t_loss.numpy()
                            eval_perfs[ii,0] = np.mean(t_perf)
                            eval_os[ii,:,:,:] = t_out.numpy()
                            eval_o_nets[ii,:,:,:] = t_out_net.numpy()
                            eval_us[ii,:,:,:] = stim
                            eval_zs[ii,:,:,:] = target                            
                            eval_rs[ii,:,:,:] = t_r.numpy()
     
                                                
                        eval_loss_mean = np.nanmean(eval_losses, 0)
                        eval_perf_mean = np.nanmean(eval_perfs, 0)
                        perfEvals[evalCount] = eval_perf_mean
                        lossEvals[evalCount] = eval_loss_mean
                        evalCount = evalCount + 1 
        
                        print("Trial = %5d, Loss: %.4f, Perf: %.4f"%(tr, eval_loss_mean, eval_perf_mean ))
                        
                        if eval_loss_mean < hp['loss_thr'] and eval_perf_mean >= hp['perf_thr']:
                            training_success = True
                            break
        
                elapsed_time = time.time() - start_time
                print("elapsed_time=",elapsed_time,"s")
        
        # Save the trained params in a .mat file
        var = {}
        var['hp'] = hp
        var['w0'] = rnn.w0
        var['b_out0'] = rnn.b_out0
        var['b_rec0'] = rnn.b_rec0
        var['stim'] = stim
        var['w']    = rnn.w.numpy()
        var['target'] = target
        var['target_mask'] = target_mask
        var['w_out']  = rnn.w_out.numpy()
        var['w_out0'] = rnn.w_out0
        var['m'] =rnn.m.numpy()
        var['N'] = rnn.N
        var['exc']    = rnn.exc
        var['inh']    = rnn.inh
        var['w_in']   = rnn.w_in.numpy()
        var['w_in0']  = rnn.w_in0
        var['b_out']  = rnn.b_out.numpy()
        var['b_rec']  = rnn.b_rec.numpy()
        var['losses'] = losses[0:tr]        
        var['tr'] = tr        
        var['tau_d'] = rnn.tau_d.numpy()
        var['tau_f'] = rnn.tau_f.numpy()
        var['U']     = rnn.U.numpy()        
        var['tau_d0'] = rnn.tau_d0
        var['tau_f0'] = rnn.tau_f0
        var['U0']     = rnn.U0        
        var['eval_us'] = eval_us
        var['eval_zs'] = eval_zs
        var['eval_os'] = eval_os
        var['eval_o_nets'] = eval_o_nets
        var['eval_rs'] = eval_rs                
        var['perfEvals'] = perfEvals[0:np.int32(tr/hp['eval_freq'])]
        var['lossEvals'] = lossEvals[0:np.int32(tr/hp['eval_freq'])]
                        
        fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
                
        out_dir = os.path.join(os.getcwd(), 'models', hp['task'],'context_90_85') 
        if os.path.exists(out_dir)==False:
            os.makedirs(out_dir)
                  
        fname = 'withSTP{}_trainSTP{}_trainW{}_para{}_rep{}.mat'.format(int(hp['with_STP_flag']),int(hp['train_STP_flag']),int(hp['train_w_flag']),paraInd,repInd)
        
        
        scipy.io.savemat(os.path.join(out_dir, fname), var)
        
        tf.compat.v1.reset_default_graph()
        
        
        
        
        
        
        
        
        
        
        
