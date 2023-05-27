from connect_four.connectfour.dummy_policies import SmartHeuristic
import numpy as np
 


def test_horizontal():
    smart = SmartHeuristic()
    observation = np.zeros((6, 7, 2))
    observation[5, 0, 0] = 1
    observation[5, 1, 0] = 1
    observation[5, 2, 0] = 1
    legal_action = np.array([1, 1, 1, 1, 1, 1, 1])
    action = smart.select_action(legal_action, observation)
    assert action


def test_vertical():
    smart = SmartHeuristic()
    observation = np.zeros((6, 7, 2))
    observation[5, 0, 0] = 1
    observation[4, 0, 0] = 1
    observation[3, 0, 0] = 1
    legal_action = np.array([1, 1, 1, 1, 1, 1, 1])
    action = smart.select_action(legal_action, observation)
    assert action


def test_positive_diag():
    smart = SmartHeuristic()
    observation = np.zeros((6, 7, 2))
    observation[5, 0, 0] = 1
    observation[4, 1, 0] = 1
    observation[3, 2, 0] = 1
    observation[3, 3, 1] = 1
    observation[4, 3, 1] = 1
    observation[5, 3, 1] = 1
    legal_action = np.array([1, 1, 1, 1, 1, 1, 1])
    action = smart.select_action(legal_action, observation)
    assert action


def test_negative_diag():
    smart = SmartHeuristic()
    observation = np.zeros((6, 7, 2))
    observation[3, 1, 0] = 1
    observation[4, 2, 0] = 1
    observation[5, 3, 0] = 1
    observation[3, 0, 1] = 1
    observation[4, 0, 1] = 1
    observation[5, 0, 1] = 1
    legal_action = np.array([1, 1, 1, 1, 1, 1, 1])
    action = smart.select_action(legal_action, observation)
    assert action


def test_other():
    smart = SmartHeuristic()

    observation_1 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
        ]
    )
    observation_2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
        ]
    )

    observation = np.stack([observation_1, observation_2], axis=2)
    legal_action = np.array([1, 1, 1, 1, 1, 1, 1])
    action = smart.select_action(legal_action, observation)
    assert action
