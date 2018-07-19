import numpy as np
import tools
import TIM
import running_tools


running_tools.first_print()
print("-----------------------------------------")

# 1: theta
theta = running_tools.calculate_theta()
print("-----------------------------------------")

# 2: seed set 1
seed_set1, activated1 = running_tools.run_seed_set1(theta)
print("-----------------------------------------")

# 3: a_theta
alpha = running_tools.calculate_alpha(theta, seed_set1.copy(), activated1)
print("-----------------------------------------")

# 4: second node seed selection
G = running_tools.load_graph()
for node in seed_set1:
    tools.activate(G, node, 1)

factors = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
seed_sets2 = []

print("seed_set1:")
print(seed_set1)

for f in factors:
    g = G.copy()
    print("----------------")
    print("* factor: "+str(f))
    seed_set2 = TIM.node_selection_(g, running_tools.k, theta, activated1, int(alpha * f))
    seed_sets2.append(seed_set2.copy())
    for node in seed_set2:
        tools.activate(g, node, 2)
    print("seed set 2:")
    print(seed_set2)
    scores = np.zeros(running_tools.diffusion_rounds)
    for i in range(running_tools.diffusion_rounds):
        g_c = g.copy()
        tools.one_round_diffuse(g_c, seed_set1.copy(), seed_set2.copy())
        scores[i] = len(g_c.graph["2"]) - len(g_c.graph["1"])
    print("avg: "+str(np.average(scores))+" var: "+str(np.var(scores))+" std: "+str(np.std(scores)))

print("-----------------------------------------")

# 5: difference between seed sets
running_tools.diff_analysis(factors, seed_sets2)
