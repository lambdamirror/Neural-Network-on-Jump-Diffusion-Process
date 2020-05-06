import time, sys, math 
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.python.training import moving_averages
import tensorflow.python.framework.dtypes

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

from jump_mc import jump_diff_mc

# parameter for the PDE
r = 0.05
sigma = 0.25
T = 1
K = 20
d = 10
Sinit = np.array([25, 20]*int(d/2))

# Poisson Jump parameters
lamb = 1
eta, delta = 0.2, 0.2
		
# parameter for the algorithm
n_layer = 4
n_neuron = [d, d, d, d]
batch_size = 500
n_maxstep = 200

def g_tf(S):
	return math.exp(-r*T)*(tf.reduce_max(S, axis=1) - K)*tf.cast(tf.greater_equal(tf.reduce_max(S, axis=1) - K, 0), dtype=tf.float64)

def _one_time_net (x, name):
	with tf.compat.v1.variable_scope ( name ):
		layer1 = _one_layer ( x , n_neuron [1], name ='layer1')
		layer2 = _one_layer ( layer1 , n_neuron [2], name ='layer2')
		z = _one_layer ( layer2, n_neuron [3] , activation_fn =None , name ='final')
	return z

def _one_layer ( input_ , output_size , activation_fn =tf.nn.relu , stddev =1.0 , name ='linear'):
	with tf.compat.v1.variable_scope ( name ):
		shape = input_.get_shape ().as_list ()
		w = tf.compat.v1.get_variable ('Matrix', [ shape [1] , output_size ], tf.float64 , \
							tf.random_normal_initializer (stddev = stddev /np.sqrt ( shape [1]+ output_size )))
		hidden = tf.matmul ( input_ , w)
	if activation_fn :
		return activation_fn ( hidden )
	else :
		return hidden

def main(arglist):
    n_time = int(arglist[0])
    y_mc = jump_diff_mc(Sinit, K, r, sigma, T, n_time, lamb, eta, delta)   
    dt = T/n_time
    sqdt = np.sqrt(dt)
    
    start_time = time.time()
    
    with tf.compat.v1.Session() as sess:
        dW = tf.random.normal(shape=[batch_size, d], stddev=1, dtype=tf.float64)
        dN = tf.random.poisson(shape=[batch_size, d], lam=(lamb*dt), dtype=tf.float64)
        logj = tf.random.normal(shape=[batch_size, d], mean=(eta-delta*delta/2), stddev=delta, dtype=tf.float64)
        J = tf.exp(logj)

        S = tf.Variable(np.ones([batch_size, d])*Sinit, dtype=tf.float64, trainable=False)
        Y0 = tf.Variable(25, dtype=tf.float64)
        Z0 = tf.Variable(tf.random.uniform([1, d], minval=-.5, maxval=.5, dtype=tf.float64))
        
        allones = tf.ones(shape=[batch_size, 1], dtype=tf.float64)
        Y = allones*Y0
        Z = tf.matmul(allones, Z0)
        
        with tf.compat.v1.variable_scope('forward'):
            for t in range(0, n_time-1):
                dS = r*S*dt + sqdt*sigma*S*tf.tile(tf.reshape(tf.reduce_sum(dW, axis=1), [batch_size, 1]), [1, d]) \
                                                                        + S*(J-1)*dN #- S*lamb*(math.exp(eta)-1)*dt		
                Y = Y + tf.reduce_sum(Z*dW, axis=1)
                S = S + dS

                Z = _one_time_net(S, "Z" + str(t+1))

                dW = tf.random.normal(shape=[batch_size, d], stddev=1, dtype=tf.float64)
                dN = tf.random.poisson(shape=[batch_size, d], lam=(lamb*dt), dtype=tf.float64)
                logj = tf.random.normal(shape=[batch_size, d], mean=(eta-delta*delta/2), stddev=delta, dtype=tf.float64)
                J = tf.exp(logj)
            
            dA = S*dt
            dS = r*S*dt + sqdt*sigma*S*tf.tile(tf.reshape(tf.reduce_sum(dW, axis=1), [batch_size, 1]), [1, d]) \
                                                                        + S*(J-1)*dN #- S*lamb*(math.exp(eta)-1)*dt
            Y = Y + tf.reduce_sum(Z*dW, axis=1)
            S = S + dS

            loss = tf.reduce_mean(tf.square(Y-g_tf(S)))
		
        global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
        boundaries = [n_maxstep]
        values = [0.1, 0.0]
        learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries, values)

        train_vars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(loss, train_vars)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        apply_op = optimizer.apply_gradients(zip (grads , train_vars ), global_step=global_step, name='train_step')
        train_ops = [ apply_op ]
        train_op = tf.group(* [ train_ops ])

        y0_values = []
        losses = []
        convergence_rates = [0]
        errors = []
        running_time = []
        steps = []
        sess.run(tf.compat.v1.global_variables_initializer())

        print("\n\n***Neural Network Scheme's Performance Summary:\n")
        print("\tr = %1.2f, sigma = %1.2f, S = [%1.2f, %1.2f], K = %d, N_time = %d, dt = %1.2f \n" % (r, sigma, Sinit[0], Sinit[1], K, n_time, dt))
        print("\tm = %d, n_training = %d, gamma = %1.2f \n" % (n_layer, n_maxstep, values[0]))
        print("\tlambda = %1.2f, eta = %1.2f, delta = %1.2f \n" % (lamb, eta, delta))

        try:
            for i in range(n_maxstep+1):
                sess.run(train_op)
                step, y0_value, currentLoss, curlearningrate = sess.run([global_step, Y0, loss, learning_rate])

                steps.append(step)
                losses.append(currentLoss)
                if len(y0_values) > 0:
                    convergence_rates.append(abs(y0_value-y0_values[-1])/y0_values[-1])
                    
                y0_values.append(y0_value)
                errors.append(abs((y0_value-y_mc)/y_mc))
                running_time.append(time.time() - start_time)
                
                if (step % (n_maxstep/10) == 0):
                    print ("\tstep : %d, \tloss : %2.5f, \tY0: %2.5f, \tconvergence : %1.5f, \terror : %1.5f" % (step, currentLoss, y0_value, convergence_rates[i] , errors[i]))

            mean_y0 = np.mean(y0_values[-50:])
            mean_error = abs((mean_y0 - y_mc)/y_mc)
            print("\n\tmean Y0 : %2.5f, \tmax convergence : %1.5f, \terror : %1.5f" % (mean_y0, max(convergence_rates[-50:]), mean_error))
            
            end_time = time.time()
            print ("\n\tNN Elapsed time : %f" % (end_time - start_time) )	
            print("\n**********\n")

        except KeyboardInterrupt:
            print("manually disengaged")
			

	# fileout = arglist[0]
	# with open(fileout, "a") as f:
		# #content = f.read()
		# content = str(n_time) + "\t" + str(mean_y0) + "\n"
		# f.write(content)
		# """
		# for i in range(len(Y_list)):
			# f.write(str(i) + "\t" + str(Y_list[i]) + "\n") #+ "\t" + str(convergence_rates[i]) + "\n")
		# """
		# f.close()
		
if __name__== '__main__':
	main(sys.argv[1:])		