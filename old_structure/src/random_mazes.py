# Generate random "mazes" (any combination of black and white fields)
from maze_gen import check_maze, draw_maze
import random
import numpy as np

def gen_random(w, h, num):
    correct = 0
    for _ in range(num):
        maze = [[random.getrandbits(1) for x in range(w)] for y in range(h)]
        maze = np.array(maze)
#        for x in range(w):
#            for y in range(h):
#                maze[x,y] = random.getrandbits(1)
        if check_maze(maze):
            correct += 1
    return correct

if __name__ == '__main__':
    print(gen_random(3, 3, 100000))
    print(gen_random(4, 4, 100000))
    print(gen_random(5, 5, 100000))
    print(gen_random(6, 6, 100000))
    print(gen_random(7, 7, 100000))
    print(gen_random(8, 8, 100000))
    

    
