import numpy as np


class SmartHeuristic:
    row_count = 6
    column_count = 7
    win_count = 4

    def check_window(self, window, num_discs, mark):
        return (
            window.count(mark) == num_discs
            and window.count(0) == self.win_count - num_discs
        )

    def count_windows(self, grid, num_discs, mark):
        num_windows = 0
        # horizontal
        for row in range(self.row_count):
            for col in range(self.column_count - self.win_count - 1):
                window = list(grid[row, col : col + self.win_count])
                if self.check_window(window, num_discs, mark):
                    num_windows += 1
        # vertical
        for row in range(self.row_count - self.win_count + 1):
            for col in range(self.column_count):
                window = list(grid[row : row + self.win_count, col])
                if self.check_window(window, num_discs, mark):
                    num_windows += 1

        # diagonals
        for row in range(self.row_count - self.win_count + 1):
            for col in range(self.column_count - self.win_count + 1):
                negative_window = list(
                    grid[
                        range(row, row + self.win_count),
                        range(col, col + self.win_count),
                    ]
                )
                if self.check_window(negative_window, num_discs, mark):
                    num_windows += 1

                positive_window = list(
                    grid[
                        range(
                            self.row_count - row - 1,
                            self.row_count - row - self.win_count - 1,
                            -1,
                        ),
                        range(
                            self.column_count - col - self.win_count,
                            self.column_count - col,
                        ),
                    ]
                )
                if self.check_window(positive_window, num_discs, mark):
                    num_windows += 1

        return num_windows

    def get_heuristic(self, grid, mark):
        num_threes = self.count_windows(grid, 3, mark)
        num_fours = self.count_windows(grid, 4, mark)
        num_threes_opp = self.count_windows(grid, 3, mark % 2 + 1)
        score = num_threes - 1e2 * num_threes_opp + 1e6 * num_fours
        return score

    def drop_piece(self, grid, col, mark):
        next_grid = grid.copy()
        for row in range(self.row_count - 1, -1, -1):
            if next_grid[row, col] == 0:
                break
        next_grid[row, col] = mark
        return next_grid

    def select_action(self, legal_action, observation):
        legal_choices = np.arange(len(legal_action))[legal_action == 1]

        my_grid = observation[:, :, 0] + observation[:, :, 1] * 2

        mark_1 = np.sum(my_grid == 1)
        mark_2 = np.sum(my_grid == 2)

        mark = 1 if mark_1 == mark_2 else 2
        scores = []
        for i in range(7):
            future_grid = self.drop_piece(my_grid, i, mark)
            scores.append(self.get_heuristic(future_grid, mark))

        scores = np.array(scores, dtype=int)
        if np.all(scores == 0):
            return np.random.choice(legal_choices)

        desired_action = np.random.choice(
            list(range(self.column_count)), p=scores / sum(scores)
        )

        if desired_action.size > 1:
            desired_action = np.random.choice(desired_action)

        if desired_action in legal_choices:
            return desired_action
        else:
            return np.random.choice(legal_choices)


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


# [[0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0]
#  [0 2 0 0 0 0 0]
#  [0 2 0 0 0 0 0]
#  [0 2 1 0 2 0 0]
#  [0 1 1 0 1 2 0]]
