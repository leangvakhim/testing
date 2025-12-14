from ssa_pm import ssapm
from coverage import coverage
from benchmark import benchmark

print("1. Coverage testing")
print("2. Benchmark testing")
val = int(input("Enter opt: "))

pop_size = 20
max_iter = 100
lb = 0
ub = 50

dim = 2
f_best_prev = 0.0
num_sensor = 20
w = 50
h = 50
sensing_radius = 10
r_error = 5

params = {
    'tau_stagnate':5,
    'flag_stagnate': False,

    # levy-flight
    'beta_levy_flight': 1.5,
    'alpha_levy_flight': 0.01,
    'small_sigma_ve': 1,

    # chaotic-rebirth
    'chaotic_rebirth_mu': 4.0,

    # ATP
    'g_0': 100,
    'alpha_gsa': 20,
    't_0': 100,
    'alpha_sa': 0.95,
    'epsilon': 1e-50,
    'r_base': 5,
    'heat_lambda': 2,
    'st':0.8,

    # FBS
    's_min':2,
    's_max':10,
    'a_min':0.01,
    'a_max':0.1,
    'danger_p': 0.3,

    # Dynamic Role (producer <=> scrounger)
    'r_start': 0.8,
    'r_end': 0.2,
    'dynamic_role_lambda': 2,
}

if val == 1:
    print("Coverage testing")
    testing = ssapm(lb, ub, dim, pop_size, max_iter, params)
    x_val = testing.initialize()
    cov = coverage(w, h, num_sensor, sensing_radius, r_error, x_val)
    cov.plot_coverage()
    # print(x_val)
elif val == 2:
    print("Benchmark testing")
    lb = -30
    ub = 30
    pop_size = 30
    dim = 30
    max_iter = 500
    testing = ssapm(lb, ub, dim, pop_size, max_iter, params)
    list_fitness = testing.run()
    # print(f"List fitness: {list_fitness}")
    # x_val = testing.initialize()
    # score_fitness = testing.obj_func(x_val)
    # print(f"Score fitness: {score_fitness:.4e}")
else:
    print("Invalid option")