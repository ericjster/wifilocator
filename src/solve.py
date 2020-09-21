#!/usr/bin/env python3

import math
import get_sample_data
import numpy as np

from scipy.optimize import minimize, differential_evolution

iter = 0

def x_to_str(x):
    return ("x: %5.1f %5.1f %5.1f c: %5.1f %5.1f %5.1f" % (
        x[0], x[1], x[2], x[3], x[4], x[5]))

def print_resx(res):
    print("[res ]: %s" % (x_to_str(res.x)))

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

    if 0:
        for optalg in ['Powell', 'Nelder-Mead', 'L-BFGS-B', 'COBYLA', 'SLSQP']:
            print_algo_title(optalg)
            x0 = (0.0, 0.0, 0.0, -21.0, 6.0, 2.5)
            bounds = [(-100,100), (-100,100), (-100,100), (-100, -10), (0.1, 100), (-25, 25)]
            if optalg in ('Nelder-Mead', 'COBYLA'):
                bounds = None
            iter = 0
            res = minimize(
                fun=loss,
                x0=x0,
                method=optalg,
                # bounds=bounds,
                tol=0.001,
                options={"maxiter":1000})
            print_resx(res)
            print("res.success:", res.success)
            # print("res.nit:", res.nit)

    if 0:
        bounds = [(-100,100), (-100,100), (-100,100), (-100, -10), (0.1, 100), (-25, 25)]
        iter = 0
        print_algo_title("scipy.ifferential_evolution")
        res = differential_evolution(
            func=loss,
            bounds=bounds,
            maxiter=1000,
            tol=0.001,
            atol=0.001)
        print_resx(res)
        print("res.success:", res.success)

    if 0:
        from cmaes import CMA
        bounds = [(-100,100), (-100,100), (-100,100), (-100, -10), (0.1, 100), (-25, 25)]
        x0 = (0.0, 0.0, 0.0, -21.0, 6.0, 2.5)
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




if __name__ == "__main__":
    # rows = get_sample_data.get_sample_data()
    # print("Found:")
    # for row in rows:
    #     print("  ", row)

    solve()

