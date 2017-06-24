import argparse
import numpy as np 
import scipy.stats as stats
import tensorflow as tf 


def linear(input, output_dim, scope=None, stddev=1.0):
	norm = tf.random_normal_initializer(stddev=stddev)
	const = tf.constant_initializer(0.0)
	with tf.variable_scope(scope or 'linear'):
		w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
		b = tf.get_variable('b', [output_dim], initializer=const)
		return tf.matmul(input, w) + b


def minibatch(input, num_kernels=5, kernal_dim=3):
	x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
	activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
	diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
	abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
	minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
	return tf.concat(1, [input, minibatch_features])


def discriminator(input, h_dim, minibatch_layer=True):
	h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
	h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))

	# without the minibatch layer, the discriminator needs an additional layer
	# to have enough capacity to separate the two distributions correctly
	if minibatch_layer:
		h2 = minibatch(h1)
	else:
		h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

	h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
	return h3


def generator(input, h_dim):
	h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
	h1 = linear(h0, 1, 'g1')
	return  h1


def optimizer(loss, var_list, initial_learning_rate):
	decay = 0.95
	num_decay_steps = 150
	batch = tf.Variable(0)
	learning_rate = tf.train.exponential_decay( initial_learning_rate,
												batch,
												num_decay_steps,
												decay,
												staircase=True
												)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minize(loss,
																		global_step=batch,
																		var_list=var_list
																		)
	return optimizer


class GAN(object):
	def __init__(self, data, gen, num_steps, batch_size, minibatch, log_every, anim_path):
		self.data = data
		self.gen = gen
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.minibatch = minibatch
		self.log_every = log_every
		self.mlp_hidden_size = 4
		self.anim_path = anim_path
		self.anim_frames = []


		# use a higher learning rate when not using the minibatch layer
		if self.minibatch:
			self.learning_rate = 0.005
		else:
			self.learning_rate = 0.03

		self._create_model()

	def _create_model(self):
		# In order to make sure that the discriminator is providing useful gradient
		# information to the generator from the start, we're going to pretrain the 
		# discriminator using a maximum likelihood objective. We define the network 
		# for this pretraining step scoped as D_pre
		with tf.variable_scope('D_pre'):
			self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
			self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
			D_pre = discriminator(self.pre_input, self.mlp_hidden_size, self.minibatch)
			self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
			self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

		# This defines the generator network - it takes samples from a noise 
		# distribution as input, and passes them through an MLP.
		with tf.variable_scope('Gen'):
			self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
			self.G = generator(self.z, self.mlp_hidden_size)

		# The discriminator tries to tell the difference between samples from the 
		# true data distribution (self.x) and the generated samples (self.z).
		#
		# Here we create two copies of the discriminator network (that share parameters)
		# as you cannot use the same network with different inputs in TensorFlow.
		with tf.variable_scope('Disc') as scope:
			self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
			self.D1 = discriminator(self.x, self.mlp_hidden_size, self.minibatch)
			scope.resuse_variables()
			self.D2 = discriminator(self.G, self.mlp_hidden_size, self.minibatch)

		# Define the loss for discriminator and generator networks (see the original 
		# paper for details), and create optimizers for both
		self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
		self.loss_g = tf.reduce_mean(-tf.log(self.D2))

		self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
		self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
		self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

		self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
		self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)


	def train(self):
		with tf.Session() as session:
			tf.global_variables_initialzer().run()

			# pretraining discriminator
			num_pretrain_steps = 1000
			for step in range(num_pretrain_steps):
				d = (np.random.random(self.batch_size) - 0.5) * 10.0
				labels = stats.norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
				pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
					self.pre_input: np.reshape(d, (self.batch_size, 1)),
					self.pre_labels: np.reshape(d, (self.batch_size, 1))
					})
			self.weightsD = session.run(self.d_pre_params)

			# copy weights from pre-training over to new D network
			for i, v in enumerate(self.d_params):
				session.run(v.assign(self.weightsD[i]))

			for step in range(self.num_steps):
				# update discriminator
				x = self.data.sample(self.batch_size)
				z = self.gen.sample(self.batch_size)
				loss_d, _ = session.run([self.loss_d, self.opt_d], {
					self.x: np.reshape(x, (self.batch_size, 1)),
					self.z: np.reshape(z, (self.batch_size, 1))
					})

				# update generator
				z = self.gen.sample(self.batch_size)
				loss_g, _ = session.run([self.loss_g, self.opt_g], {
					self.z: np.reshape(z, (self.batch_size, 1))
					})

				if step % self.log_every == 0:
					print('{}: loss_d = {}\t loss_g = {}'.format(step, loss_d, loss_g))

				if self.anim_path:
					self.anim_frames.append(self._samples(session))

			if self.anim_path:
				self._save_animation()
			else:
				self._plot_distributions(session)
				
def main(args):
	model = GAN(
		DataDistribution(),
		GeneratorDistribution(range=8),
		args.num_steps,
		args.batch_size,
		args.minibatch,
		args.log_every,
		args.anim
		)
	model.train()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-steps', type=int, default=1200,
						help='the number of training steps to take')
	parser.add_argument('--batch-size', type=int, default=12,
						help='the batch size')
	parser.add_argument('--minibatch', type=bool, default=False,
						help='use minibatch discrimination')
	parser.add_argument('--log-every', type=int, default=10,
						help='print loss after this many steps')
	parser.add_argument('--anim', type=str, default=None,
						help='name of the output animation file (default: none)')
	return parser.parse_args()

if __name__ == '__main__':
	main(parse_args())