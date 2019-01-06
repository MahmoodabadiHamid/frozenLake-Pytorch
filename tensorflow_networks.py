import tensorflow as tf #1.8.0
import numpy as np #1.14.5

state_size=len(env.observation_space.sample())


class Actor:
    def __init__(self, env, lamb=1e-5, learning_rate=0.01, scope="policy_estimator"):
        self.env = env
        self.lamb = lamb
        self.learning_rate = learning_rate

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, state_size, name="state")
        
        self.l1= tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=400,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer()
        )

        self.mu = tf.contrib.layers.fully_connected(
            inputs=self.l1,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.mu = tf.squeeze(self.mu)

        self.sigma = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.norm_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0], self.env.action_space.high[0])

    def _build_train_op(self):
        self.action_train = tf.placeholder(tf.float32, name="action_train")
        self.advantage_train = tf.placeholder(tf.float32, name="advantage_train")

        self.loss = -tf.log(
            self.norm_dist.prob(self.action_train) + 1e-5) * self.advantage_train - self.lamb * self.norm_dist.entropy()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        feed_dict = {self.state: state}
        return sess.run(self.action, feed_dict=feed_dict)

    def update(self, state, action, advantage, sess):
        feed_dict = {
            self.state: state,
            self.action_train: action,
            self.advantage_train: advantage
        }
        sess.run([self.train_op], feed_dict=feed_dict)


class Critic:
    def __init__(self, env, learning_rate=0.01, scope="value_estimator"):
        self.env = env
        self.learning_rate = learning_rate

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, state_size, name="state")
        
        self.l1= tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=400,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal_initializer()
        )

        
        self.value = tf.contrib.layers.fully_connected(
            inputs=self.l1,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.value = tf.squeeze(self.value)

    def _build_train_op(self):
        self.target = tf.placeholder(tf.float32, name="target")
        self.loss = tf.reduce_mean(tf.squared_difference(self.value, self.target))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: state})

    def update(self, state, target, sess):
        feed_dict = {
            self.state: state,
            self.target: target
        }
        sess.run([self.train_op], feed_dict=feed_dict)


def main(episodes=100, gamma=0.95, display=False, lamb=1e-5, policy_lr=0.001, value_lr=0.1):
    tf.reset_default_graph()
    policy_estimator = Actor(env, lamb=lamb, learning_rate=policy_lr)
    value_estimator = Critic(env, learning_rate=value_lr)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    stats = []
    for i_episode in range(episodes):
        state = env.reset()
        reward_total = 0
        while True:
            action = policy_estimator.predict(state, sess)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward
            target = reward + gamma * value_estimator.predict(next_state, sess)
            td_error = target - value_estimator.predict(state, sess)

            policy_estimator.update(state, action, advantage=td_error, sess=sess)
            value_estimator.update(state, target, sess=sess)

            if done:
                if reward_total<90:
                    reward=-100
                    target = reward + gamma * value_estimator.predict(next_state, sess)
                    td_error = target - value_estimator.predict(state, sess)

                    policy_estimator.update(state, action, advantage=td_error, sess=sess)
                    value_estimator.update(state, target, sess=sess)

                break
            state = next_state
            #env.render()
        stats.append(reward_total)
        if np.mean(stats[-100:]) > 90 and len(stats) >= 101:
            print(np.mean(stats[-100:]))
            print("Solved.")
        print("Episode: {}, reward: {}.".format(i_episode, reward_total))
    return np.mean(stats[-100:])


if __name__ == "__main__":
    policy_lr, value_lr, lamb, gamma = [0.0001, 0.0046415888336127773, 2.782559402207126e-05, 0.999]
    loss = main(episodes=1000, gamma=gamma, display=False, lamb=lamb, policy_lr=policy_lr, value_lr=value_lr)
    print(-loss)
    env.close()
