import numpy as np
import os
import time
import agent_RL
import agent_SE

def demo(algo):
    filename = "qv_" + algo + ".npy"
    qv = np.load(filename)

    _, gameplay = agent_RL.run(algo, qv, False)
    print_gameplay(gameplay)


def print_gameplay(gameplay):
    for s in gameplay:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(s)
        time.sleep(0.2)


def count_down(string_format):
    for i in list(reversed(range(3))):
        print(string_format.format(i+1), end="\r")
        time.sleep(1)


def main():
    print("This is a demo of the algorithms")

    #first we run one epoch of the trained MC algorithm
    # count_down("Starting the Monte Carlo algorithm gameplay in {0}")
    # demo("MC")

    # #then we run one epoch of the trained QL algorithm
    # count_down("Starting the Q-Learning algorithm gameplay in {0}")
    # demo("QL")

    #finally we run an epoch of the SE algorithm
    count_down("Starting the search algorithm gameplay in {0}")
    agent_SE.run(using_terminal=True)

main()