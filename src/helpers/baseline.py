"""
RANDOM MAZE GENERATOR WITH DIFFERENT COIN FLIP PROBABILITY
Calculating the baseline for a fixed baseline against percentage of correct mazes
"""
import os
import csv
import numpy as np
from maze_utils import check_maze, draw_maze

def random_maze(mx, my, q):
    count = 0.0
    all_counts = 0.0
    while all_counts < 1000000:
        all_counts+=1
        # p = [pbt(0) = q, pbt(1) = 1-q]
        rand_maze = np.random.choice([0, 1], size=(mx, my), p=[q, 1-q])
        if check_maze(rand_maze):
            # print(rand_maze)
            count+=1
    return count, all_counts

if __name__ == "__main__":
    baseline = csv.writer(open("basline.csv", "w", newline=""), delimiter=",")
    baseline.writerow(["maze_size", "0_density",
    "1_density", "correct", "all_counts"])
    mx = 5
    pbt_choice = np.arange(0.1, 1.0, 0.1)
    for q in pbt_choice:
        correct_counts, all_counts = random_maze(mx, mx, q)
        print("{}/{}: {}/{}".format(q, 1-q, correct_counts, all_counts))
        baseline.writerow([mx, q, 1-q, correct_counts, all_counts])

#    maze = np.array([[0, 1, 1, 1, 1, 1],
#  [1, 1, 0, 1, 0, 1],
#  [0, 1, 0, 0, 0, 1],
#  [1, 0, 1, 1, 1, 1],
#  [1, 1, 1, 0, 0, 1],
#  [1, 0, 1, 1, 1, 0]]
# )
