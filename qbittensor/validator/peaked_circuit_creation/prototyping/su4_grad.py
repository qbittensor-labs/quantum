import sys
import sympy as sy
kron = sy.kronecker_product

# https://pennylane.ai/qml/demos/tutorial_kak_decomposition#kokcu-fdhs:
# For any given U in SU(4), we can write:
#   U =
#       U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
#       R_XX([eta0]) R_YY([eta1]) R_ZZ([eta2])
#       U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])

# https://arxiv.org/abs/quant-ph/0308033
# For any given U in SU(4), we can write:
#   U =
#       U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
#       CNOT
#       Rx(0, eta0) Rz(1, eta1)
#       CNOT
#       U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])
#       CNOT
#       Rz(1, eta2)

x = sy.Matrix([[0, 1], [1, 0]])
y = sy.Matrix([[0, -sy.I], [sy.I, 0]])
z = sy.Matrix([[1, 0], [0, -1]])
cnot = sy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

# def rot(ax: sy.Matrix, angle) -> sy.Matrix:
#     return sy.exp(-sy.I * angle / 2 * ax)

def u3(alpha, beta, gamma) -> sy.Matrix:
    return sy.Matrix([
        [
            sy.cos(alpha),
            -sy.exp(sy.I * gamma) * sy.sin(alpha),
        ],
        [
            sy.exp(sy.I * beta) * sy.sin(alpha),
            sy.exp(sy.I * (beta + gamma)) * sy.cos(alpha),
        ],
    ])

def rx(angle) -> sy.Matrix:
    return sy.Matrix([
        [ sy.cos(angle / 2), -sy.I * sy.sin(angle / 2), ],
        [ -sy.I * sy.sin(angle / 2), sy.cos(angle / 2), ],
    ])

def rz(angle) -> sy.Matrix:
    return sy.Matrix([
        [ sy.exp(-sy.I * angle / 2), 0 ],
        [ 0, sy.exp( sy.I * angle / 2) ],
    ])

def rxx(angle) -> sy.Matrix:
    return sy.Matrix([
        [ sy.cos(angle / 2), 0, 0, -sy.I * sy.sin(angle / 2) ],
        [ 0, sy.cos(angle / 2), -sy.I * sy.sin(angle / 2), 0 ],
        [ 0, -sy.I * sy.sin(angle / 2), sy.cos(angle / 2), 0 ],
        [ -sy.I * sy.sin(angle / 2), 0, 0, sy.cos(angle / 2) ],
    ])

def ryy(angle) -> sy.Matrix:
    return sy.Matrix([
        [ sy.cos(angle / 2), 0, 0,  sy.I * sy.sin(angle / 2) ],
        [ 0, sy.cos(angle / 2), -sy.I * sy.sin(angle / 2), 0 ],
        [ 0, -sy.I * sy.sin(angle / 2), sy.cos(angle / 2), 0 ],
        [  sy.I * sy.sin(angle / 2), 0, 0, sy.cos(angle / 2) ],
    ])

def rzz(angle) -> sy.Matrix:
    return sy.Matrix([
        [ sy.exp(-sy.I * angle / 2), 0, 0, 0 ],
        [ 0, sy.exp( sy.I * angle / 2), 0, 0 ],
        [ 0, 0, sy.exp( sy.I * angle / 2), 0 ],
        [ 0, 0, 0, sy.exp(-sy.I * angle / 2) ],
    ])

def main():
    alpha = sy.symbols("alpha0 alpha1 alpha2", real=True)
    beta = sy.symbols("beta0 beta1 beta2", real=True)
    eta = sy.symbols("eta0 eta1 eta2", real=True)
    gamma = sy.symbols("gamma0 gamma1 gamma2", real=True)
    delta = sy.symbols("delta0 delta1 delta2", real=True)

    ## "ising-like"
    # u_impl = (
    #     kron(u3(*alpha), u3(*beta))
    #     * rxx(eta[0]) * ryy(eta[1]) * rzz(eta[2])
    #     * kron(u3(*gamma), u3(*delta))
    # )

    ## "cnot-based"
    u_impl = (
        kron(u3(*alpha), u3(*beta))
        * cnot
        * kron(rx(eta[0]), rz(eta[1]))
        * cnot
        * kron(u3(*gamma), u3(*delta))
        * cnot
        * kron(sy.eye(2), rz(eta[2]))
    )

    u_given = sy.Matrix(
        sy.symbols(" ".join(f"U{i}{j}" for i in range(4) for j in range(4)))
    ).reshape(4, 4)

    with open("su4_grad.txt", "w") as outfile:
        print("compute fidelity ... ", end="", flush=True)
        trace = sum((u_impl.H * u_given).diagonal())
        fid = trace.conjugate() * trace / 16
        print("done", flush=True)
        print("fidelity", file=outfile)
        print(fid, file=outfile)

        print("compute gradients:", flush=True)
        print("gradients", file=outfile)
        params = [*alpha, *beta, *eta, *gamma, *delta]
        for param in params:
            print(param, "... ", end="", flush=True)
            grad = fid.diff(param)
            print("done", flush=True)
            print(param, file=outfile)
            print(grad, file=outfile)

if __name__ == "__main__":
    main()

