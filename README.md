# maze-gan

Using GANs to generate Mazes

#### Using Random Noise to Generate(Check) Mazes
* Mazes were generated in each iteration until atleast  **one correct** maze was detected. 
* **Counts** (averaged over iterations) refers to the minimum number of times mazes were generated in each iteration before detecting it as one
* **Probability** (averaged over iterations) refers to the chance of detecting **one** correct maze over all the mazes generated till that point

| Number of iterations 	| 4x4 counts 	| 4x4 probability 	| 5x5 counts 	| 5x5 probability 	| 6x6 counts 	| 6x6 probability 	| 7x7 counts 	| 7x7 probability 	| 8x8 counts 	| 8x8 probability 	| 9x9 counts 	| 9x9 probability 	|
|----------------------	|------------	|-----------------	|------------	|-----------------	|------------	|-----------------	|------------	|-----------------	|------------	|-----------------	|------------	|-----------------	|
| 100                  	| 6.52       	| 0.371           	| 14.98      	| 0.200           	| 48.75      	| 0.104           	| 169.090    	| 0.0209          	| 733.47     	| 0.00981         	| 2776.06    	| 0.0121          	|
| 1000                 	| 6.065      	| 0.364           	| 16.514     	| 0.181           	| 53.546     	| 0.0734          	| 186.969    	| 0.0244          	| 671.398    	| 0.0152          	| 3109.4     	| 0.00165         	|
| 10000                	| 5.9484     	| 0.361           	| 16.7136    	| 0.1799          	| 53.3037    	| 0.751           	| 179.4268   	| 0.0286          	| 694.7302   	| 0.0089          	| 2952.4277  	| 0.00227         	|
| 100000               	| 5.969      	| 0.359           	| 16.834     	| 0.178           	| 52.648     	| 0.077           	| 181.999    	| 0.0288          	| 694.7499   	| 0.0093          	| --         	| --              	|
