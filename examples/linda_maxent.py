"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.maxent as maxent
import irl.mdp.lindaworld as lindaworld

import sys
import rospy
import random
from visualization_msgs.msg import *
from strands_navigation_msgs.msg import TopologicalMap


def main(discount, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the objectworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_objects: Number of objects. int.
    n_colours: Number of colours. int.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    #wind = 0.3
    #trajectory_length = 8

    lw = lindaworld.Lindaworld()
    #ground_r = np.array([lw.reward(s) for s in range(lw.n_states)])
    #policy = find_policy(lw.n_states, lw.n_actions, lw.transition_probability,
    #                     ground_r, lw.discount, stochastic=False)
    #trajectories = lw.generate_trajectories(n_trajectories,
    #                                        trajectory_length,
    #                                        lambda s: policy[s])
    feature_matrix = lw.feature_matrix(discrete=False)
    reward = maxent.irl(feature_matrix, lw.n_actions, discount,
        lw.transition_probability, lw.trajectories, epochs, learning_rate)

    policy = maxent.find_policy(lw.n_states, reward, lw.n_actions, discount,
                               lw.transition_probability)

    ## Save the policy
    print "Saving policy file...",
    sys.stdout.flush()
    policy_file = open("linda_policy.pnpo", "w")
    for i, a_prob in enumerate(policy):
        state = lw.state_list[i]
        actions = np.array(lw.action_list)[np.where(a_prob==np.amax(a_prob))]
        #print set(actions.tolist())
        action = random.choice(list(set(actions.tolist()) - set(["NO_ACTION"])))
        policy_file.write(state + "\t" + action + "\n")
    policy_file.close()
    print "DONE"

    #for i, ap in enumerate(policy):
    #    print lw.state_list[i], np.array(lw.action_list)[np.where(ap==np.amax(ap))]

    #for i, state in enumerate(lw.state_list):
    #    print reward[i], state

    rospy.init_node("reward_visualizer")

    ## Visualize the reward on the rviz markers
    # initialize interactive marker server
    rew_markers_publisher = rospy.Publisher("reward_visualizer", MarkerArray, latch=True, queue_size=10)

    # take the current markers
    top_map = rospy.wait_for_message("/topological_map", TopologicalMap, timeout=10)

    max_reward = max(reward)
    min_reward = min(reward)
    map_v = "/map"
    marker_array = MarkerArray()
    for index, node in enumerate(top_map.nodes):

        # get the corresponding state index
        currentstate_index = None
        closeststate_index = None
        current_marker_id = None
        closest_marker_id = None
        for i, state in enumerate(lw.state_list):
            if "CurrentNode_" + node.name in state.split(" "):
                currentstate_index = i
                current_marker_id = int(node.name.replace("WayPoint", ""))
                print "current>>>> ", state
            if "ClosestNode_" + node.name in state.split(" "):
                closeststate_index = i
                closest_marker_id = int(node.name.replace("WayPoint", "")) * 100
                print "closest>>>> ", state

        # Current state marker
        if current_marker_id is not None:
            current_box_marker = Marker()
            if currentstate_index is not None:
                # get heatmap color
                r, g, b = rgb(min_reward, max_reward, reward[currentstate_index])
                current_box_marker.text = str(reward[currentstate_index])
            else:
                current_box_marker.text = "0"
                r = g = b = 0.0
            current_box_marker.header.frame_id = map_v
            current_box_marker.type = Marker.CYLINDER
            current_box_marker.action = Marker.ADD
            current_box_marker.id = current_marker_id
            current_box_marker.scale.x = 0.5
            current_box_marker.scale.y = 0.5
            current_box_marker.scale.z = 0.1
            current_box_marker.pose = node.pose
            current_box_marker.color.r = r
            current_box_marker.color.g = g
            current_box_marker.color.b = b
            current_box_marker.color.a = 1.0

            marker_array.markers.append(current_box_marker)

        if closest_marker_id is not None:
            # Closest state marker
            closest_box_marker = Marker()
            if closeststate_index is not None:
                # get heatmap color
                r, g, b = rgb(min_reward, max_reward, reward[closeststate_index])
                closest_box_marker.text = str(reward[closeststate_index])
                #print reward[closeststate_index]
            else:
                closest_box_marker.text = "0"
                r = g = b = 0.0
            closest_box_marker.header.frame_id = map_v
            closest_box_marker.type = Marker.CYLINDER
            closest_box_marker.action = Marker.ADD
            closest_box_marker.id = closest_marker_id
            closest_box_marker.scale.x = 2
            closest_box_marker.scale.y = 2
            closest_box_marker.scale.z = 0.01
            closest_box_marker.pose = node.pose
            closest_box_marker.color.r = r
            closest_box_marker.color.g = g
            closest_box_marker.color.b = b
            closest_box_marker.color.a = 0.7

            marker_array.markers.append(closest_box_marker)

    #print marker_array
    rew_markers_publisher.publish(marker_array)

    rospy.spin()

def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = max(0, 1 - ratio)
    r = max(0, ratio - 1)
    g = 1 - b - r
    return r, g, b

if __name__ == '__main__':
    main(0.9, 50, 0.01)
