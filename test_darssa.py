from dar_ssa import darssa
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
    # DAR
    's_min':5,
    's_max':30,
    'a_min':0.01,
    'a_max':0.4,
    'danger_p': 0.1,
    'gamma': 1.2,
    # 'gamma': 1.5,
    'omega': 0.7,

    # Dynamic Role (producer <=> scrounger)
    'r_start': 0.8,
    'r_end': 0.2,
    'dynamic_role_lambda': 2,
    'st':0.8,
    'epsilon': 1e-8,
}

if val == 1:
    print("Coverage testing")
    pop_size = 50
    max_iter = 100
    lb = 0
    ub = 50
    num_sensor = 20
    dim = num_sensor * 2
    params['w'] = 50
    params['h'] = 50
    params['sensing_radius'] = 7
    params['r_error'] = 0.5
    params['num_nodes'] = num_sensor
    func_name='coverage_optimization'
    # x = lb + np.random.rand(num_sensor, dim) * (ub - lb)
    testing = darssa(lb, ub, dim, pop_size, max_iter, params, func_name)
    # x_val = testing.initialize()
    best_fitness, best_pos, convergence_curve = testing.run()
    best_pos_reshaped = best_pos.reshape(num_sensor, 2)
    # print(f"Best fitness: {1 - best_fitness}")
    # print(f"Best pos: {best_pos}")
    # print(f"convergence curve: {convergence_curve}")

    # node_roles = []
    # for i in range(num_sensor):
    #     # Example: First 20% are Producers, rest are Scroungers
    #     if i < (0.2 * num_sensor):
    #         node_roles.append("P")
    #     else:
    #         node_roles.append("S")

    cov = coverage(params['w'], params['h'], num_sensor, params['sensing_radius'], params['r_error'], best_pos_reshaped)
    # cov.calculate_probabilistics_coverage()

    true_coverage = cov.calculate_probabilistics_coverage()
    print(f"True Final Coverage: {true_coverage * 100:.2f}%")
    # cov.plot_coverage(best_fitness)
    # cov.plot_coverage(1.0 - true_coverage, node_roles)
    # cov.plot_coverage(convergence_curve[-1], node_roles)
    # cov.plot_iterative_coverage(convergence_curve)
    cov.plot_results_combined(convergence_curve, best_fitness, name="DAR-SSA")
    # print(x_val)
elif val == 2:
    print("Benchmark testing")
    # Options:
    # f1: "sphere"       f2: "schwefel_2_21"   f3: "schwefel_2_22"
    # f4: "max"          f5: "rosenbrock"      f6: "step"
    # f7: "quartic"      f8: "schwefel_2_26"   f9: "rastrigin"
    # f10: "ackley"      f11: "griewank"
    times = 10
    # lb = -30
    # ub = 30
    # dim = 30
    pop_size = 50
    max_iter = 500
    list_val = []
    func_name = "rosenbrock"
    func, lb, ub, dim, target = benchmark.get_function(func_name)

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
        testing = darssa(lb, ub, dim, pop_size, max_iter, params, func_name)
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