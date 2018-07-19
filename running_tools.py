import numpy as np
import tools
import TIM

k = 20
data_set = "CA-HepTh.txt"
diffusion_rounds = 1000

def first_print():
    print("data set: " + data_set)
    print("k: " + str(k))
    print("second type graph loading")
    print("diffusion rounds"+str(diffusion_rounds))


def load_graph():
    # g = tools.load_graph(data_set)
    g = tools.load_graph2(data_set)
    return g


def calculate_theta():
    g = load_graph()
    kpt_star, r_p = TIM.kpt_estimation(g, k)
    print("kpt* : " + str(kpt_star))
    epsilon_p = 5.0 * np.power(1 * 0.01 / (k + 1), 1.0 / 3.0)
    kpt_plus = TIM.refine_kpt(g, k, kpt_star, epsilon_p, r_p)
    print("kpt+ : " + str(kpt_plus))
    Lambda = TIM.get_landa(g.number_of_nodes(), k)
    theta = Lambda / kpt_plus
    theta = int(theta)
    print("lambda : " + str(Lambda))
    print("theta : " + str(theta))

    return int(theta)


def run_seed_set1(theta):
    g = load_graph()
    seed_set1 = TIM.node_selection(g, k, theta)
    for node in seed_set1:
        tools.activate(g, node, 1)
    print("seed set 1:")
    print(seed_set1)
    tools.one_round_diffuse(g, seed_set1.copy(), [])

    activated1 = [x for x in g.graph["1"] if x not in seed_set1]
    print("number of activated nodes for 1st player: " + str(len(activated1)))

    return seed_set1,activated1


def calculate_alpha(theta, seed_set1, activated1):
    g = load_graph()
    for node in seed_set1:
        tools.activate(g, node, 1)

    activated1_set = set(activated1)
    counter_activated1 = 0
    len_g = len(g.graph["free"])
    for i in range(theta):
        v_index = np.random.randint(len_g)
        v = g.graph["free"][v_index]
        if v in activated1_set:
            counter_activated1 += 1

    a_theta = counter_activated1
    print("number of selecting first player activated nodes: " + str(counter_activated1))
    print("percent of selecting first player activated nodes: " + str(counter_activated1 / theta))
    return a_theta


def diff_analysis(factors, seed_set2):
    for i in range(len(factors)):
        print("factor: " + str(factors[i]))
        for j in range(len(factors)):
            diff = len(set(seed_set2[i]).difference(set(seed_set2[j])))
            print("f1: " + str(factors[i]) + " f2: " + str(factors[j]) + " diff:" + str(diff))
