# Random Maze Generator using Depth-first Search
# http://en.wikipedia.org/wiki/Maze_generation_algorithm

# Code from http://code.activestate.com/recipes/578356-random-maze-generator/
# FB - 20121214
# Modified by Peter O'Conor
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label
imgx = 500; imgy = 500

def check_maze(maze, mx, my):
    # single connected-component
    labeled_array, num_features = label(maze)
    if num_features > 1:
        return False
    # no loops
    maze = 1 - maze
    s = [[1,1,1],
         [1,1,1],
         [1,1,1]]
    labeled_array, num_features = label(maze, structure=s)
    for feat in range(1, num_features+1):
        indexes = np.array(np.where(labeled_array==feat))

        if np.all(indexes[:, :] != 0) and np.all(indexes[0, :] != mx-1) and np.all(indexes[1, :] != my-1):
            return False
    return True

def generate_maze(mx, my):
    """
    Generate a shape (mx, my) maze.  1's represent "hallways" and 0's represent "walls".
    All "hallways" will be connected into a single component with no loops.

    This does not take care of setting appropriate (start/stop) pixels.  Any white pixel could be used as a start/end
    point.

    :param int mx: Number of horizontal units
    :param int my: Number of vertical units
    :return: Array(my,mx)[bool]
    """
    maze = np.zeros((my, mx), dtype=np.bool)
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    stack = [(np.random.randint(0, mx ), np.random.randint(0, my ))]

    while len(stack) > 0:
        (cx, cy) = stack[-1]
        maze[cy][cx] = 1
        nlst = [] # list of available neighbors
        for i in range(4):
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                if maze[ny][nx] == 0:
                    # of occupied neighbors must be 1
                    ctr = 0
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey][ex] == 1: ctr += 1
                    if ctr == 1: nlst.append(i)
        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = nlst[np.random.randint(0, len(nlst))]
            cx += dx[ir]; cy += dy[ir]
            stack.append((cx, cy))
        else:
            stack.pop()

    if check_maze(maze, mx, my):
        correct_maze = np.array(maze, dtype = np.int32)
    else: correct_maze = np.array(maze, dtype = np.int32)

    return correct_maze

def demo_generate_maze(mx, my): # width and height of the maze
    maze = generate_maze(mx, my)
    if check_maze(maze, mx, my):
        draw_maze(maze)

def draw_maze(maze):
    plt.figure()
    plt.imshow(maze, cmap='gray')
    plt.show()

def maze_data(mx, my, N):
    data = []
    for i in range(N):
        maze = generate_maze(mx, my)
        if i%100 == 0:
            print("Generated {}/{} mazes...".format(i, N))
        data.append(maze)
    print(len(data))
    return np.array(data)

 # you can add to the list all the transformations you need.

# if __name__ == '__main__':
#     demo_generate_maze(16, 16)
