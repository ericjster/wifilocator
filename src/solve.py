#!/usr/bin/env python3

import math
import get_sample_data
import numpy as np
import nlopt

favorites = [
    'SLSQP',
    'cmaes',
    'ImFil',
    'Nomad',
    # 'Bobyqa', # This version of bobyqa lacks quick termination
    nlopt.LN_NEWUOA,
    nlopt.LN_NELDERMEAD,
    nlopt.LN_BOBYQA,
]
cheap = [
    'Powell',
    'SLSQP',
    'ImFil',
    # 'SnobFit', # Has some warning messages and quits too early
    # 'Bobyqa', # This version of bobyqa lacks quick termination
    # Consider: Call pybobyqa directly, and it takes may other arguments.
    nlopt.LN_PRAXIS,
    nlopt.LN_NEWUOA,
    nlopt.LN_NELDERMEAD,
    nlopt.LN_BOBYQA,
    nlopt.GN_DIRECT,
]
expensive = [
    'COBYLA',
    'differential_evolution',
    'shgo',
    'Nomad',
    'Bobyqa', # python version is expensive
    'SnobFit',
    'SQSnobFit',
    'cmaes', # can be adjusted
    nlopt.LN_COBYLA,
    # nlopt.GN_MLSL, # lots of calls but fast
    # nlopt.GN_MLSL_LDS,
]
good_global_quality = [
    'shgo',
    'cmaes',
    'Nomad',
    'SLSQP',
    'differential_evolution',
    'shgo',
    nlopt.LN_NEWUOA,
]
poor_quality = [
    'SQSnobfit',
    'Powell', # Maybe need better stopping criteria
    'L-BFGS-B',
    'Nelder-Mead',
]
test_algo = [
    'differential_evolution',
]
ONLY_FAVORITES = False
ONLY_CHEAP = False
ONLY_EXPENSIVE = False
ONLY_TEST = False
EXCLUDE_EXPENSIVE = False
EXCLUDE_POOR_QUALITY = False

def should_exclude(algo):
    # print("Checking:", algo)
    if EXCLUDE_POOR_QUALITY and algo in poor_quality:
        return True
    if EXCLUDE_EXPENSIVE and algo in expensive:
        return True
    if ONLY_FAVORITES and algo not in favorites:
        return True
    if ONLY_CHEAP and algo not in cheap:
        return True
    if ONLY_EXPENSIVE and algo not in expensive:
        return True
    if ONLY_TEST and algo not in test_algo:
        return True
    return False

iter = 0

def x_to_str(x):
    return ("x: %5.1f %5.1f %5.1f c: %6.1f %5.1f %5.1f" % (
        x[0], x[1], x[2], x[3], x[4], x[5]))

def x_loss_to_str(x, loss):
    return ("x: %5.1f %5.1f %5.1f c: %6.1f %5.1f %5.1f, loss: %5.1f" % (
        x[0], x[1], x[2], x[3], x[4], x[5], loss))

def print_resx(res):
    print("xbest : %s" % (x_loss_to_str(res.x, res.fun)))

def print_algo_title(algo_name):
    print("----------")
    print("Algo: " + algo_name)

def solve():

    data = get_sample_data.get_sample_data()

    global iter
    iter = 0

    def loss(xcandidate):
        """Return the loss associated with a proposed x,y,z,c1,c2,c3.
        We estimate the distance based on RSS as: $d = c2 * e^{(RSS-c3)/c1}$
        
        Consider: Change to use numpy.
        Consider: Remember best solution.
        """
        cx = xcandidate[0]
        cy = xcandidate[1]
        cz = xcandidate[2]
        c1 = xcandidate[3]
        c2 = xcandidate[4]
        c3 = xcandidate[5]

        global iter
        iter += 1
        loss = 0.0

        # Some optizers cannot do bounds.
        # c1 must be negative because RSS is always negative.
        # c2 must always be positive
        if c1 < -10 and 0.1 < c2 and -25 < c3 and c3 < 25:
            # print()
            # print("x:", xcandidate)
            for row in data:
                rx = row[0]
                ry = row[1]
                rz = row[2]
                rss = row[3]
                dist_xyz = math.sqrt( (cx-rx)**2 + (cy-ry)**2 + (cz-rz)**2 )
                dist_rss = c2 * math.exp( (rss - c3) / c1 )
                loss += abs(dist_xyz - dist_rss)**2
            loss /= len(data)
        else:
            loss = 1.e10

        print("[%4d]: %s, loss: %5.1f" % ( iter, x_to_str(xcandidate), loss))
        return loss

    bounds = [(-200,200), (-200,200), (-200,200), (-200, -20), (0.2, 200), (-50, 50)]
    bounds = [(-100,100), (-100,100), (-100,100), (-100, -10), (0.1, 100), (-25, 25)]
    bounds = np.array(bounds, dtype=float)
    bounds_lower = bounds[:,0]
    bounds_upper = bounds[:,1]
    bounds_orig = np.copy(bounds)
    x0 = np.array([0.0, 0.0, 0.0, -21.0, 6.0, 2.5])
    x0 = np.array([0.0, 0.0, 0.0, -98.0, 31.0, -0.6])
    x0_orig = np.array([0.0, 0.0, 0.0, -98.0, 31.0, -0.6])

    def get_bounds():
        return np.copy(bounds_orig)

    def get_x0():
        # We were getting x0 set to the solution, giving later algos a different scenario.
        # Try: ./solve.py 2>&1 | grep "Algo\|best"
        return np.copy(x0_orig)

    if 1:
        from scipy.optimize import minimize
        for optalg in ['Powell', 'Nelder-Mead', 'L-BFGS-B', 'COBYLA', 'SLSQP']:
            if should_exclude(optalg):
                continue
            print_algo_title(optalg)
            bounds_tmp = get_bounds()
            if optalg in ('Nelder-Mead', 'COBYLA'):
                bounds_tmp = None
            iter = 0
            res = minimize(
                fun=loss,
                x0=get_x0(),
                method=optalg,
                bounds=bounds_tmp,
                tol=0.001,
                options={"maxiter":1000})
            print_resx(res)
            print("res.success:", res.success)
            # print("res.nit:", res.nit)

    if 1:
        from scipy.optimize import differential_evolution
        for alg in ['differential_evolution']:
            if should_exclude(alg):
                continue
            iter = 0
            print_algo_title("scipy.differential_evolution")
            # Generally I have had good luck with this, but:
            # problem 1: crazy values (but did converge after 8800 feval)
            # problem 2: cannot limit to number of feval
            res = differential_evolution(
                func=loss,
                bounds=get_bounds(),
                maxiter=1000,
                tol=0.01,
                atol=0.01)
            # print(res)
            print_resx(res)
            print("res.success:", res.success)

    if 1:
        from scipy.optimize import shgo
        for alg in ['shgo']:
            if should_exclude(alg):
                continue
            iter = 0
            print_algo_title("scipy.shgo")
            res = shgo(
                func=loss,
                bounds=get_bounds(),
                #n=100, # Number of sobel init points
                #iters=100,
                #minimizer_kwargs = { 'method': 'SLSQP', 'options': {'ftol:1.0'} },
                options = {
                    'maxfev': 1000,
                    'f_tol': 0.1,
                    'maxiter': 1000,
                    'maxev': 1000,
                },
                sampling_method='sobol', # 'simplicial', # or 'sobol'
            )
            # print(res)
            print_resx(res)
            print("res.success:", res.success)

    if 1:
        from cmaes import CMA

        for alg in ['cmaes']:
            if should_exclude(alg):
                continue
            optimizer = CMA(mean=get_x0(), sigma=5.0, bounds=get_bounds())
            print_algo_title("cmaes")
            iter = 0
            bestf = 1e100
            bestx = None
            for generation in range(100):
                solutions = []
                for _ in range(optimizer.population_size):
                    x = optimizer.ask()
                    value = loss(x)
                    solutions.append((x, value))
                    if bestf > value:
                        bestf = value
                        bestx = x
                    # print(f"#{generation} {value}")
                    # print("[sofar]: %s" % (x_to_str(x)))
                optimizer.tell(solutions)
            print("xbest : %s" % (x_loss_to_str(bestx, bestf)))

    if 1:
        from SQSnobFit import minimize

        for alg in ['SQSnobFit']:
            if should_exclude(alg):
                continue
            iter = 0
            print_algo_title("SQSnobFit")
            result, history = minimize(
                f=loss,
                x0=get_x0(),
                bounds=get_bounds(),
                budget=1000) # max iterations
            print("xbest : %s" % (x_loss_to_str(result.optpar, result.optval)))

    if 1:
        from skquant.opt import minimize

        for alg in ['ImFil', 'SnobFit', 'Bobyqa']:
            if should_exclude(alg):
                continue
            iter = 0
            print_algo_title(alg)
            result, history = minimize(
                func=loss,
                x0=get_x0(),
                bounds=get_bounds(),
                budget=1000, # max iterations
                method=alg,
                options={ 'maxfail': 200, },
            )
            print("xbest : %s" % (x_loss_to_str(result.optpar, result.optval)))

    if 1:
        from SQNomad import minimize
        for alg in ['Nomad']:
            if should_exclude(alg):
                continue
            iter = 0
            print_algo_title("Nomad")
            result, history = minimize(
                f=loss,
                x0=get_x0(),
                bounds=get_bounds(),
                budget=1000, # max iterations
                # Need a way to set abs stopping criteria
            )
            print("xbest : %s" % (x_loss_to_str(result.optpar, result.optval)))
        

    if 1:
        import nlopt

        def loss_nlopt(x, grad):
            if grad.size > 0:
                raise ValueError("grad function not supported")
            return loss(x)

        # Too poor: nlopt.LN_SBPLX, nlopt.GN_CRS2_LM
        for alg in [nlopt.LN_PRAXIS, nlopt.LN_COBYLA, nlopt.LN_NEWUOA, nlopt.LN_NELDERMEAD, nlopt.LN_BOBYQA,
            nlopt.GN_DIRECT, nlopt.GN_MLSL, nlopt.GN_MLSL_LDS,
            # nlopt.GD_STOGO, nlopt.GD_STOGO_RAND, # Not included
            # nlopt.GN_AGS, # Not included
            nlopt.GN_ISRES,
            nlopt.GN_CRS2_LM,
            nlopt.GN_ESCH,
            ]:
            if should_exclude(alg):
                continue
            iter = 0
            opt = nlopt.opt(alg, len(x0_orig))
            print_algo_title(opt.get_algorithm_name())
            opt.set_min_objective(loss_nlopt)
            opt.set_lower_bounds(bounds_lower)
            opt.set_upper_bounds(bounds_upper)
            opt.set_stopval(0.01) # this will be close enough
            opt.set_ftol_rel(0.001) # average loss
            opt.set_ftol_abs(0.001)
            opt.set_xtol_abs(0.001) # consider vector to treat xyz and c1c2c3 differently
            opt.set_initial_step((bounds_upper-bounds_lower)/20.0)
            opt.set_maxeval(1000)
            res = opt.optimize(get_x0())
            fbest = loss(res)
            print("xbest : %s" % (x_loss_to_str(res, fbest)))
            # print("last  : %s" % (x_loss_to_str(opt.last_optimize_result(), opt.last_optimum_value())))

    if 0:
        import nevergrad as ng

        n = len(get_x0())
        budget = 100
        for opt in [
            ng.optimizers.OnePlusOne(parametrization=n, budget=budget),
            ng.optimizers.TwoPointsDE(parametrization=n, budget=budget),
            ng.optimizers.CMA(parametrization=n, budget=budget),
            ng.optimizers.PSO(parametrization=n, budget=budget),
            ng.optimizers.NGO(parametrization=n, budget=budget),
            ng.optimizers.ScrHammersleySearchPlusMiddlePoint(parametrization=n, budget=budget),
            ng.optimizers.TBPSA(parametrization=n, budget=budget),
        ]:
            algname = type(opt).__name__
            if should_exclude(algname):
                continue

            iter = 0
            print_algo_title(algname)
            x0 = get_x0()
            # opt = ng.optimizers.registry[algo]
            opt.suggest(x0)

            # This is the way to set bounds.
            # Only used by some algos.
            # A penalty function would work better.
            opt.parametrization.register_cheap_constraint(lambda x: 
                np.all((bounds_lower <= x0) & (x0 <= bounds_upper)))

            res = opt.minimize(loss)
            # print("xbest : %s" % (x_to_str(res)))
            print("res:", res)


if __name__ == "__main__":
    # rows = get_sample_data.get_sample_data()
    # print("Found:")
    # for row in rows:
    #     print("  ", row)

    solve()
