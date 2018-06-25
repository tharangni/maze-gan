'''
RANDOM MAZE GENERATOR WITH DIFFERENT COIN FLIP PROBABILITY
'''

import numpy as np
from maze_gen import check_maze, draw_maze

def random_maze(mx, my, q):
    count = 0.0
    pbt = 0.0
    while pbt == 0:
    # rand_maze = np.random.randint(2, size = (mx, mx))
    # p = [pbt(0) = q, pbt(1) = 1-q]
        rand_maze = np.random.choice([0, 1], size=(mx, my), p=[q, 1-q])
        count+=1
        if check_maze(rand_maze):
            # draw_maze(rand_maze)
            pbt = 1/count
    return count, pbt

if __name__ == '__main__':
    iter_list = [100, 1000, 10000, 100000]
    pbt_choice = np.arange(0.1, 1.0, 0.1)
    for each_item in iter_list:
        print("\nGenerating experiment {} times :".format(each_item))
        for q in pbt_choice:
            print("0/1 probability = {}, {} respectively".format(q, 1-q))
            for i in range(4, 6):
                count_list = []
                pbt_list = []
                for j in range(each_item):
                    count, pbt = random_maze(i, i, q)
                    count_list.append(count)
                    pbt_list.append(pbt)
                    # print('j: ', j, '--', i, '--counts: ', count, '--pbt: ', pbt)
                # print('Maze size: ', i, '--counts avg', sum(count_list)/len(count_list), '--pbt avg', sum(pbt_list)/len(pbt_list))
                print("Maze size: {} --- Avg Counts: {} ; --- Avg PBT: {} ".format(i, round(sum(count_list)/len(count_list), 1), sum(pbt_list)/len(pbt_list) ))

'''
Generating 100 mazes:
I:  4 --counts avg 6.52 --pbt avg 0.3714994409467029
I:  5 --counts avg 14.98 --pbt avg 0.2000841455435066
I:  6 --counts avg 48.75 --pbt avg 0.10404906933470884
I:  7 --counts avg 169.09 --pbt avg 0.02096400497606488
I:  8 --counts avg 733.47 --pbt avg 0.009810979697465914
I:  9 --counts avg 2776.06 --pbt avg 0.012153732242238667
Generating 1000 mazes:
I:  4 --counts avg 6.065 --pbt avg 0.36425326687489656
I:  5 --counts avg 16.514 --pbt avg 0.18128738651358262
I:  6 --counts avg 53.546 --pbt avg 0.07343366152093467
I:  7 --counts avg 186.969 --pbt avg 0.02440007602616886
I:  8 --counts avg 671.398 --pbt avg 0.015262429880208693
I:  9 --counts avg 3109.4 --pbt avg 0.001657790319786124
Generating 10000 mazes:
I:  4 --counts avg 5.9484 --pbt avg 0.36159486272997504
I:  5 --counts avg 16.7136 --pbt avg 0.17998367721135905
I:  6 --counts avg 53.3037 --pbt avg 0.07510550740221213
I:  7 --counts avg 179.4268 --pbt avg 0.028627469690049822
I:  8 --counts avg 694.7302 --pbt avg 0.008922275443594821
I:  9 --counts avg 2952.4277 --pbt avg 0.002279048227042786
Generating 100000 mazes:
I:  4 --counts avg 5.96908 --pbt avg 0.3596355439989614
I:  5 --counts avg 16.83446 --pbt avg 0.1784856831840889
I:  6 --counts avg 52.64865 --pbt avg 0.07728001673135114
I:  7 --counts avg 181.99948 --pbt avg 0.028839144138198193
I:  8 --counts avg 694.74992 --pbt avg 0.009304571628381676

'''
