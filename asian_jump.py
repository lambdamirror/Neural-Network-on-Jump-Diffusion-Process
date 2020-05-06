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
S0 = 35
A0 = 0

# Poisson Jump parameters
lamb = 5
eta, delta = 0.2, 0.5

start_time = time.time()
		
# parameter for the algorithm
n_layer = 4
n_neuron = [1, 1, 1, 1]
batch_size = 10000
n_maxstep = 200

def g_tf(S, A):
	return math.exp(-r*T)*(A/T - K)*tf.cast(tf.greater_equal((A/T - K), 0), tf.float64)

def _one_time_net (x, name):
	with tf.compat.v1.variable_scope ( name ):
		#layer1 = _one_layer ( x_norm , (1- isgamma )* n_neuron [1] + isgamma * n_neuron [1] , name ='layer1')
		#layer2 = _one_layer ( layer1 , (1- isgamma )* n_neuron [2] + isgamma * n_neuron [2] , name ='layer2')
		z = _one_layer ( x, n_neuron [3] , activation_fn =None , name ='final')
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
    y_mc = jump_diff_mc(S0, K, r, sigma, T, n_time, lamb, eta, delta)   
    dt = T/n_time
    sqdt = np.sqrt(dt)
    with tf.compat.v1.Session() as sess:
        dW = tf.random.normal(shape=[batch_size, 1], stddev=1, dtype=tf.float64)
        dN = tf.random.poisson(shape=[batch_size, 1], lam=(lamb*dt), dtype=tf.float64)
        logj = tf.random.normal(shape=[batch_size, 1], mean=(eta-delta*delta/2), stddev=delta, dtype=tf.float64)
        J = tf.exp(logj)

        S = tf.Variable(np.ones([batch_size, 1])*S0, dtype=tf.float64, trainable=False)
        A = tf.Variable(np.ones([batch_size, 1])*A0, dtype=tf.float64, trainable=False)
        Y0 = tf.Variable(20, dtype=tf.float64)
        allones = tf.ones(shape=[batch_size, 1], dtype=tf.float64)
        Y = allones*Y0

        Z = tf.Variable(np.ones([batch_size, 1])*0, dtype=tf.float64)

        with tf.compat.v1.variable_scope('forward'):
            for t in range(0, n_time-1):
                dA = S*dt
                dS = r*S*dt + sqdt*sigma*S*dW + S*(J-1)*dN - S*lamb*(math.exp(eta)-1)*dt		
                Y = Y + Z*dW
                S = S + dS 
                A = A + dA
                Z = _one_time_net(S, "Z" + str(t+1))

                dW = tf.random.normal(shape=[batch_size, 1], stddev=1, dtype=tf.float64)
                dN = tf.random.poisson(shape=[batch_size, 1], lam=(lamb*dt), dtype=tf.float64)
                logj = tf.random.normal(shape=[batch_size, 1], mean=(eta-delta*delta/2), stddev=delta, dtype=tf.float64)
                J = tf.exp(logj)
            
            dA = S*dt
            dS = r*S*dt + sqdt*sigma*S*dW  + S*(J-1)*dN - S*lamb*(math.exp(eta)-1)*dt
            Y = Y + Z*dW
            S = S + dS
            A = A + dA
            loss = tf.reduce_mean(tf.square(Y-g_tf(S, A)))
		
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

        print("\n\n***The calculation has finished. The performance summary is following:\n")
        print("\tr = %1.2f, sigma = %1.2f, S0 = %d, K = %d, N_time = %d, dt = %1.2f \n" % (r, sigma, S0, K, n_time, dt))
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
                    print ("\tstep : %d, \tloss : %2.5f, \tY0: %2.5f, \tconvergence : %1.5f, \t error : %1.5f" % (step, currentLoss, y0_value, convergence_rates[i], errors[i]))

            end_time = time.time()
            print ("\n\tElapsed time : %f" % (end_time - start_time) )	
            print("\n**********\n")

        except KeyboardInterrupt:
            print("manually disengaged")
			
	# mean_y0 = sum(y0_values[151:])/50
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