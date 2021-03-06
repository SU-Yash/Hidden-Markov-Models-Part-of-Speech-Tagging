from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        for j in range(S):
            alpha[j, 0] = self.pi[j] * self.B[j, self.obs_dict[Osequence[0]]]
  
        for t in range(1, L):
            for j in range(S):
                alpha[j, t] = self.B[j, self.obs_dict[Osequence[t]]] * sum([self.A[i, j] * alpha[i, t - 1] for i in range(S)])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        for j in range(S):
            beta[j, L - 1] = 1
          
        for t in reversed(range(L - 1)):
            for i in range(S):
                beta[i, t] = sum([beta[j, t + 1] * self.A[i, j] * self.B[j, self.obs_dict[Osequence[t + 1]]] for j in range(S)])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        alpha = self.forward(Osequence)
        prob = sum(alpha[:,-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sequence = self.sequence_prob(Osequence)

        prob = (alpha * beta)/sequence
        ###################################################
        return prob

    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sequence = self.sequence_prob(Osequence)

        for i in range(S):
            for j in range(S):
                for t in range(L-1):
                    prob[i,j,t] = (alpha[i, t] * self.A[i,j] * self.B[j, self.obs_dict[Osequence[t+1]]] * beta[j, t+1])/sequence
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)

        delta = np.zeros([S, L])
        Delta = np.zeros([S, L], dtype="int")
        delta[:,0] = self.pi * self.B[:,self.obs_dict[Osequence[0]]]

        for t in range(1,L):
           for i in range(S):
                delta[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * np.max(self.A[:, i] * delta[:, t-1])
                Delta[i, t] = np.argmax(self.A[:, i] * delta[:, t-1])
        z = np.argmax(delta[:, L-1])
        path.append(list(self.state_dict.keys())[list(self.state_dict.values()).index(z)])

        for t in range(L-1,0,-1):
           z = Delta[z,t]
           path.append(list(self.state_dict.keys())[list(self.state_dict.values()).index(z)])
        path = path[::-1]
        ###################################################
        return path
