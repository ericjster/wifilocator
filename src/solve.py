#!/usr/bin/env python3

import math
import get_sample_data
import numpy as np


iter = 0

def x_to_str(x):
    return ("x: %5.1f %5.1f %5.1f c: %5.1f %5.1f %5.1f" % (
        x[0], x[1], x[2], x[3], x[4], x[5]))

def x_loss_to_str(x, loss):
    return ("x: %5.1f %5.1f %5.1f c: %5.1f %5.1f %5.1f, loss: %5.1f" % (
        x[0], x[1], x[2], x[3], x[4], x[5], loss))

def print_resx(res):
    print("[res ]: %s" % (x_loss_to_str(res.x, res.fun)))

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
        
        TODO: Change to use numpy.
        TODO: Remember best solution.
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
                loss += abs(dist_xyz - dist_rss)
            loss /= len(data)
        else:
            loss = 1.e10

        print("[%4d]: %s, loss: %5.1f" % ( iter, x_to_str(xcandidate), loss))
        return loss

    bounds = [(-100,100), (-100,100), (-100,100), (-100, -10), (0.1, 100), (-25, 25)]
    bounds = np.array(bounds, dtype=float)
    bounds_lower = bounds[:,0]
    bounds_upper = bounds[:,1]
    x0 = np.array([0.0, 0.0, 0.0, -21.0, 6.0, 2.5])

    if 0:
        from scipy.optimize import minimize
        # for optalg in ['Powell', 'Nelder-Mead', 'L-BFGS-B', 'COBYLA', 'SLSQP']:
        for optalg in ['Powell', 'COBYLA', 'SLSQP']:
            print_algo_title(optalg)
            bounds_tmp = bounds
            if optalg in ('Nelder-Mead', 'COBYLA'):
                bounds_tmp = None
            iter = 0
            res = minimize(
                fun=loss,
                x0=x0,
                method=optalg,
                bounds=bounds_tmp,
                tol=0.001,
                options={"maxiter":1000})
            print_resx(res)
            print("res.success:", res.success)
            # print("res.nit:", res.nit)

    if 0:
        from scipy.optimize import differential_evolution
        iter = 0
        print_algo_title("scipy.differential_evolution")
        res = differential_evolution(
            func=loss,
            bounds=bounds,
            maxiter=1000,
            tol=0.01,
            atol=0.01)
        # print(res)
        print_resx(res)
        print("res.success:", res.success)

    if 0:
        from cmaes import CMA
        optimizer = CMA(mean=x0, sigma=5.0)

        print_algo_title("cmaes")
        iter = 0
        for generation in range(50):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = loss(x)
                solutions.append((x, value))
                # print(f"#{generation} {value}")
                # print("[sofar]: %s" % (x_to_str(x)))

            optimizer.tell(solutions)
        # TODO: Remember best

    if 0:
        from SQSnobFit import minimize

        iter = 0
        print_algo_title("SQSnobFit")
        res = minimize(
            f=loss,
            x0=x0,
            bounds=bounds,
            budget=1000) # max iterations
        print(res[0])
        # TODO: Get fbest, xbest from res
        # print(dir(res[0]))
        # print("fbest:", res[0])
        # print("xbest:", res[1])

    if 0:
        from skquant.opt import minimize

        # for method in ['ImFil', 'SnobFit', 'Bobyqa']:
        for method in ['ImFil', 'SnobFit']:
            iter = 0
            print_algo_title(method)
            result, history = minimize(
                func=loss,
                x0=x0,
                bounds=bounds,
                budget=1000, # max iterations
                method=method)
            print("xbest : %s" % (x_loss_to_str(result.optpar, result.optval)))

    if 0:
        import nlopt

        def loss_nlopt(x, grad):
            if grad.size > 0:
                raise ValueError("grad function not supported")
            return loss(x)

        # Too poor: nlopt.LN_SBPLX, nlopt.GN_CRS2_LM
        for algo in [nlopt.LN_PRAXIS, nlopt.LN_COBYLA, nlopt.LN_NEWUOA, nlopt.LN_NELDERMEAD, nlopt.LN_BOBYQA]:
            iter = 0
            opt = nlopt.opt(algo, len(x0))
            print_algo_title(opt.get_algorithm_name())
            opt.set_min_objective(loss_nlopt)
            opt.set_lower_bounds(bounds_lower)
            opt.set_upper_bounds(bounds_upper)
            opt.set_stopval(0.1) # this will be close enough
            opt.set_ftol_rel(0.001) # average loss
            opt.set_ftol_abs(0.001)
            opt.set_xtol_abs(0.001) # consider vector to treat xyz and c1c2c3 differently
            opt.set_initial_step((bounds_upper-bounds_lower)/20.0)
            opt.set_maxeval(1000)
            res = opt.optimize(x0)
            print("xbest : %s" % (x_to_str(res)))
            # print("last  : %s" % (x_loss_to_str(opt.last_optimize_result(), opt.last_optimum_value())))

    if 0:
        import nevergrad as ng

        n = len(x0)
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
            iter = 0
            print_algo_title(type(opt).__name__)
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

