from ssa_pm import ssapm
from coverage import coverage
from benchmark import benchmark
import numpy as np
import opfunu
import opfunu.cec_based.cec2017 as cec2017
import opfunu.cec_based.cec2020 as cec2020
import opfunu.cec_based.cec2022 as cec2022

print("1. Coverage testing")
print("2. Benchmark testing (CEC 2017/2020/2022)")
val = int(input("Enter opt: "))

params = {
    'tau_stagnate': 100,
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
    'alpha_sa': 0.99,
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
    pop_size = 20
    max_iter = 500
    lb = 0
    ub = 50
    dim = 2
    num_sensor = 20
    w = 50
    h = 50
    sensing_radius = 10
    r_error = 5
    testing = ssapm(lb, ub, dim, pop_size, max_iter, params)
    # x_val = testing.initialize()
    best_fitness, best_pos, convergence_curve = testing.run()
    # print(f"Best fitness: {best_fitness:.4e}")
    # print(f"Best pos: {best_pos}")
    # cov = coverage(w, h, num_sensor, sensing_radius, r_error, best_pos)
    # cov.plot_coverage()
    # print(x_val)
elif val == 2:
    print("Benchmark testing")
    times = 30
    lb = -30
    ub = 30
    dim = 30
    pop_size = 100
    max_iter = 1000
    list_val = []
    func_name = "F5_function"

    # CEC 2017 (F1 - F30)
    # funcs = opfunu.cec_based.cec2017.CEC2017(ndim=dim)
    # func_range = range(1, 31)
    # func_id = [1] + list(range(3, 4))

    # CEC 2020 (F1 - F10)
    # func = opfunu.cec_based.cec2020.CEC2020(ndim=dim)
    # func_range = range(1, 11)

    # CEC 2022 (F1 - F12)
    # func = opfunu.cec_based.cec2022.CEC2022(ndim=dim)
    # func_id = list(range(1, 3))

    # print(f"{'Fun':<5} | {'Mean':<12} | {'Std':<12} | {'Best':<12} | {'Worst':<12}")
    # print("-" * 65)

    # for f_id in func_id:
    #     try:
    #         func_name = f"F{f_id}2020"
    #         func_class = getattr(cec2020, func_name)
    #         benchmark_obj = func_class(ndim=dim)
    #     except AttributeError:
    #         print(f"Function {func_name} not found")
    #         continue

    #     def obj_func(val):
    #         return benchmark_obj.evaluate(val)

    #     list_val = []

    #     for _ in range(times):
    #         optimizer = ssapm(lb, ub, dim, pop_size, max_iter, params, obj_func)
    #         best_fitness, best_pos, convergence_curve = optimizer.run()
    #         list_val.append(best_fitness)


    #     mean_val = np.mean(list_val)
    #     std_val = np.std(list_val)
    #     best_val = np.min(list_val)
    #     worst_val = np.max(list_val)

    #     print(f"F{f_id:<4} | {mean_val:.4e} | {std_val:4e} | {best_val:4e} | {worst_val:4e}")

    for _ in range(times):
        testing = ssapm(lb, ub, dim, pop_size, max_iter, params, func_name)
        best_fitness, best_pos, convergence_curve = testing.run()
        list_val.append(best_fitness)
    mean_val = np.mean(list_val)
    std_val = np.std(list_val)
    best_val = np.min(list_val)
    worst_val = np.max(list_val)
    print(f"Mean: {mean_val:.4e}")
    print(f"Std: {std_val:.4e}")
    print(f"Best: {best_val:.4e}")
    print(f"Worst: {worst_val:.4e}")
else:
    print("Invalid option")