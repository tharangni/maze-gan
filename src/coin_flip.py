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
            # print(rand_maze)
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
Generating experiment 100 times :
0/1 probability = 0.1, 0.9 respectively
Maze size: 4 --- Avg Counts: 500.1 ; --- Avg PBT: 0.008977260458025514
Maze size: 5 --- Avg Counts: 25353.7 ; --- Avg PBT: 0.00029392514728404624
0/1 probability = 0.2, 0.8 respectively
Maze size: 4 --- Avg Counts: 57.9 ; --- Avg PBT: 0.06634389777270984
Maze size: 5 --- Avg Counts: 755.8 ; --- Avg PBT: 0.00509742422667849
0/1 probability = 0.30000000000000004, 0.7 respectively
Maze size: 4 --- Avg Counts: 29.5 ; --- Avg PBT: 0.11279012366785254
Maze size: 5 --- Avg Counts: 170.0 ; --- Avg PBT: 0.0238873870770751
0/1 probability = 0.4, 0.6 respectively
Maze size: 4 --- Avg Counts: 24.9 ; --- Avg PBT: 0.09157504161637657
Maze size: 5 --- Avg Counts: 99.1 ; --- Avg PBT: 0.03429504694731769
0/1 probability = 0.5, 0.5 respectively
Maze size: 4 --- Avg Counts: 21.3 ; --- Avg PBT: 0.13462867103473747
Maze size: 5 --- Avg Counts: 106.1 ; --- Avg PBT: 0.03259282516132395
0/1 probability = 0.6, 0.4 respectively
Maze size: 4 --- Avg Counts: 18.2 ; --- Avg PBT: 0.15977551251415828
Maze size: 5 --- Avg Counts: 173.6 ; --- Avg PBT: 0.02342705899804209
0/1 probability = 0.7000000000000001, 0.29999999999999993 respectively
Maze size: 4 --- Avg Counts: 11.7 ; --- Avg PBT: 0.23424087038692717
Maze size: 5 --- Avg Counts: 95.5 ; --- Avg PBT: 0.049825702742547294
0/1 probability = 0.8, 0.19999999999999996 respectively
Maze size: 4 --- Avg Counts: 5.0 ; --- Avg PBT: 0.4251272565154145
Maze size: 5 --- Avg Counts: 21.1 ; --- Avg PBT: 0.15919429444159655
0/1 probability = 0.9, 0.09999999999999998 respectively
Maze size: 4 --- Avg Counts: 1.7 ; --- Avg PBT: 0.7823333333333334
Maze size: 5 --- Avg Counts: 3.0 ; --- Avg PBT: 0.5401075036075038

Generating experiment 1000 times :
0/1 probability = 0.1, 0.9 respectively
Maze size: 4 --- Avg Counts: 505.0 ; --- Avg PBT: 0.009733530604587188
Maze size: 5 --- Avg Counts: 27538.6 ; --- Avg PBT: 0.00045947665959149856
0/1 probability = 0.2, 0.8 respectively
Maze size: 4 --- Avg Counts: 59.5 ; --- Avg PBT: 0.07220992985149276
Maze size: 5 --- Avg Counts: 756.3 ; --- Avg PBT: 0.007516153461097387
0/1 probability = 0.30000000000000004, 0.7 respectively
Maze size: 4 --- Avg Counts: 25.9 ; --- Avg PBT: 0.11999457853145952
Maze size: 5 --- Avg Counts: 181.8 ; --- Avg PBT: 0.02461944684007665
0/1 probability = 0.4, 0.6 respectively
Maze size: 4 --- Avg Counts: 18.8 ; --- Avg PBT: 0.14955495102452407
Maze size: 5 --- Avg Counts: 116.5 ; --- Avg PBT: 0.04098501045274518
0/1 probability = 0.5, 0.5 respectively
Maze size: 4 --- Avg Counts: 18.8 ; --- Avg PBT: 0.16790881250734505
Maze size: 5 --- Avg Counts: 121.6 ; --- Avg PBT: 0.03643350276728692
0/1 probability = 0.6, 0.4 respectively
Maze size: 4 --- Avg Counts: 18.2 ; --- Avg PBT: 0.1669735905653507
Maze size: 5 --- Avg Counts: 156.7 ; --- Avg PBT: 0.028016056354964722
0/1 probability = 0.7000000000000001, 0.29999999999999993 respectively
Maze size: 4 --- Avg Counts: 10.7 ; --- Avg PBT: 0.24729278879264865
Maze size: 5 --- Avg Counts: 98.9 ; --- Avg PBT: 0.039415975114075574
0/1 probability = 0.8, 0.19999999999999996 respectively
Maze size: 4 --- Avg Counts: 4.7 ; --- Avg PBT: 0.4307918109309506
Maze size: 5 --- Avg Counts: 19.2 ; --- Avg PBT: 0.1600311479209052
0/1 probability = 0.9, 0.09999999999999998 respectively
Maze size: 4 --- Avg Counts: 1.7 ; --- Avg PBT: 0.7598492063492068
Maze size: 5 --- Avg Counts: 3.2 ; --- Avg PBT: 0.5391605185449219
'''


'''
Generating experiment 100 times :
0/1 probability = 0.1, 0.9 respectively
Maze size: 4 --- Avg Counts: 570.0 ; --- Avg PBT: 0.006951907580516455
Maze size: 5 --- Avg Counts: 24732.1 ; --- Avg PBT: 0.0004549290725885819
0/1 probability = 0.2, 0.8 respectively
Maze size: 4 --- Avg Counts: 51.7 ; --- Avg PBT: 0.10661922475915944
Maze size: 5 --- Avg Counts: 936.4 ; --- Avg PBT: 0.0034642640030618788
0/1 probability = 0.30000000000000004, 0.7 respectively
Maze size: 4 --- Avg Counts: 27.4 ; --- Avg PBT: 0.12980650144685757
Maze size: 5 --- Avg Counts: 169.1 ; --- Avg PBT: 0.03637948282778433
0/1 probability = 0.4, 0.6 respectively
Maze size: 4 --- Avg Counts: 17.2 ; --- Avg PBT: 0.19442432149887484
Maze size: 5 --- Avg Counts: 117.7 ; --- Avg PBT: 0.03402498388941631
0/1 probability = 0.5, 0.5 respectively
Maze size: 4 --- Avg Counts: 19.6 ; --- Avg PBT: 0.1499870555543538
Maze size: 5 --- Avg Counts: 121.5 ; --- Avg PBT: 0.033910189875318485
0/1 probability = 0.6, 0.4 respectively
Maze size: 4 --- Avg Counts: 16.6 ; --- Avg PBT: 0.20298659731088006
Maze size: 5 --- Avg Counts: 137.0 ; --- Avg PBT: 0.03145774730424481
0/1 probability = 0.7000000000000001, 0.29999999999999993 respectively
Maze size: 4 --- Avg Counts: 13.2 ; --- Avg PBT: 0.229169428670584
Maze size: 5 --- Avg Counts: 104.9 ; --- Avg PBT: 0.0330556504888169
0/1 probability = 0.8, 0.19999999999999996 respectively
Maze size: 4 --- Avg Counts: 4.3 ; --- Avg PBT: 0.4745641262613882
Maze size: 5 --- Avg Counts: 21.7 ; --- Avg PBT: 0.14361132059368775
0/1 probability = 0.9, 0.09999999999999998 respectively
Maze size: 4 --- Avg Counts: 1.9 ; --- Avg PBT: 0.7225952380952384
Maze size: 5 --- Avg Counts: 3.4 ; --- Avg PBT: 0.5291075036075039

Generating experiment 1000 times :
0/1 probability = 0.1, 0.9 respectively
Maze size: 4 --- Avg Counts: 515.6 ; --- Avg PBT: 0.014284159170446151
Maze size: 5 --- Avg Counts: 27289.7 ; --- Avg PBT: 0.0002666512689522922
0/1 probability = 0.2, 0.8 respectively
Maze size: 4 --- Avg Counts: 57.9 ; --- Avg PBT: 0.06991924023931852
Maze size: 5 --- Avg Counts: 745.1 ; --- Avg PBT: 0.009668141104912563
0/1 probability = 0.30000000000000004, 0.7 respectively
Maze size: 4 --- Avg Counts: 27.3 ; --- Avg PBT: 0.12834465976054812
Maze size: 5 --- Avg Counts: 185.8 ; --- Avg PBT: 0.030350187181032322
0/1 probability = 0.4, 0.6 respectively
Maze size: 4 --- Avg Counts: 19.0 ; --- Avg PBT: 0.150601408584462
Maze size: 5 --- Avg Counts: 116.4 ; --- Avg PBT: 0.051205865819891935
0/1 probability = 0.5, 0.5 respectively
Maze size: 4 --- Avg Counts: 18.2 ; --- Avg PBT: 0.16310859263395952
Maze size: 5 --- Avg Counts: 117.5 ; --- Avg PBT: 0.039030196838525294
0/1 probability = 0.6, 0.4 respectively
Maze size: 4 --- Avg Counts: 17.1 ; --- Avg PBT: 0.16671762943376056
Maze size: 5 --- Avg Counts: 153.3 ; --- Avg PBT: 0.036330724648801715
0/1 probability = 0.7000000000000001, 0.29999999999999993 respectively
Maze size: 4 --- Avg Counts: 11.2 ; --- Avg PBT: 0.24898580446763474
Maze size: 5 --- Avg Counts: 98.2 ; --- Avg PBT: 0.042735512533438706
0/1 probability = 0.8, 0.19999999999999996 respectively
Maze size: 4 --- Avg Counts: 4.4 ; --- Avg PBT: 0.4408135177592022
Maze size: 5 --- Avg Counts: 20.8 ; --- Avg PBT: 0.1490959463093805
0/1 probability = 0.9, 0.09999999999999998 respectively
Maze size: 4 --- Avg Counts: 1.7 ; --- Avg PBT: 0.7688377344877351
Maze size: 5 --- Avg Counts: 3.2 ; --- Avg PBT: 0.5376084900178931

Generating experiment 10000 times :
0/1 probability = 0.1, 0.9 respectively
Maze size: 4 --- Avg Counts: 494.8 ; --- Avg PBT: 0.012618475087179332

'''
