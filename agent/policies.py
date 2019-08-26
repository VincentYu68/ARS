'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


class FeedForwardNNPolicy(Policy):
    """
    Feedforward neural network policy.
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.hidden_layer_sizes = policy_params["hidden_layer_sizes"]

        self._num_weights = 0
        self._weight_start_id = []
        self._weight_end_id = []
        self._layer_sizes = [self.ob_dim]
        for l in range(len(self.hidden_layer_sizes)):
            self._weight_start_id.append(self._num_weights)
            self._num_weights += self._layer_sizes[-1] * self.hidden_layer_sizes[l]
            self._weight_end_id.append(self._num_weights)
            self._layer_sizes.append(self.hidden_layer_sizes[l])
        self._weight_start_id.append(self._num_weights)
        self._num_weights += self._layer_sizes[-1] * self.ac_dim
        self._weight_end_id.append(self._num_weights)
        self._layer_sizes.append(self.ac_dim)

        self.weights = np.zeros(self._num_weights, dtype=np.float64)

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)

        ith_layer_result = ob
        for l in range(len(self._layer_sizes)-1):
            weight_mat = np.reshape(self.weights[self._weight_start_id[l] : self._weight_end_id[l]],
                                    (self._layer_sizes[l+1], self._layer_sizes[l]))
            ith_layer_result = np.tanh(np.dot(weight_mat, ith_layer_result))

        return ith_layer_result

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux