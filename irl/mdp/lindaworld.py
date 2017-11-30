"""
Implements the objectworld MDP described in Levine et al. 2011.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import pickle

import math
from itertools import product

import numpy as np
import numpy.random as rn

class Lindaworld():
    """
    Lindaworld MDP.
    """

    def __init__(self):
        """
        -> Lindaworld
        """
        traj_file = None
        try:
            traj_file = open("trajectories.dat", "rb")
        except:
            print "Trajectories file not found!"
        else:
            trajectories = pickle.load(traj_file)

        # Create the sets with unique states and actions
        state_set = set()
        action_set = set()
        max_length_traj = 0
        for traj in trajectories:
            if len(traj) > max_length_traj:
                max_length_traj = len(traj)
            for (state, action) in traj:
                state_set.add(" ".join(state))
                action_set.add(" ".join(action))
                #print action

        # Get transition frequencies and generate trajectories with indexes
        self.trajectories = []
        transition_probability = np.zeros((len(state_set), len(action_set), len(state_set)))
        state_list = list(state_set)
        action_list = list(action_set)
        for traj in trajectories:
            tmp_traj = []
            prev_state = None
            prev_action = None
            for (state, action) in traj:
                state = state_list.index(" ".join(state))
                action = action_list.index(" ".join(action))
                tmp_traj.append([state, action, 0]) # reward 0 (we don't need it...)
                if prev_state:
                    transition_probability[prev_state]\
                            [prev_action][state] += 1
                prev_state = state
                prev_action = action
            # fill with the missing steps
            if prev_state:
                for i in range(max_length_traj - len(traj)):
                    no_action = action_list.index("NO_ACTION")
                    tmp_traj.append([prev_state, no_action, 0])

                self.trajectories.append(tmp_traj)

        self.trajectories = np.array(self.trajectories)
        print self.trajectories.shape

        # normalize to get probabilities
        for i in range(len(state_list)):
            for j in range(len(action_list)):
                norm = np.linalg.norm(transition_probability[i][j])
                if norm > 0:
                    #print norm
                    transition_probability[i][j] /= norm

        self.transition_probability = transition_probability
        self.n_states = len(state_list)
        self.n_actions = len(action_list)
        self.state_list = state_list
        self.action_list = action_list
        print self.transition_probability.shape

    def feature_vector(self, i, discrete=True):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """

        state = [0] * self.n_states
        state[i] = 1

        return state

    def feature_matrix(self, discrete=True):
        """
        Get the feature matrix for this objectworld.

        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> NumPy array with shape (n_states, n_states).
        """

        return np.array([self.feature_vector(i, discrete)
                         for i in range(self.n_states)])

    def reward(self, state_int):
        """
        Get the reward for a state int.

        state_int: State int.
        -> reward float
        """

        return 0

    def generate_trajectories(self, n_trajectories, trajectory_length, policy):
        """
        Generate n_trajectories trajectories with length trajectory_length.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        -> [[(state int, action int, reward float)]]
        """

        raise NotImplementedError()

    def optimal_policy(self, state_int):
        raise NotImplementedError(
            "Optimal policy is not implemented for Objectworld.")
    def optimal_policy_deterministic(self, state_int):
        raise NotImplementedError(
            "Optimal policy is not implemented for Objectworld.")
