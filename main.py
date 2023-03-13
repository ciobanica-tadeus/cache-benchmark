import time
import numpy as np


# ijk and jik misses/iter = 1.25
# kij and ikj misses/iter = 0.5
# jki and kji misses/iter = 2.0


# Process_time is better than time.time() and perf_counter()
def naive_multiplication_ijk_perf_counter(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.perf_counter()
    for i in range(len(a)):
        for j in range(len(a)):
            for k in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    # End measure
    exec_time = time.perf_counter() - start_time
    print("Naive multiply ijk ==> " + str(exec_time) + " s")
    return c


def naive_multiplication_ijk(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for i in range(len(a)):
        for j in range(len(a)):
            for k in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    # End measure
    exec_time = time.process_time() - start_time
    print("Naive multiply ijk ==> " + str(exec_time) + " s")
    return c


def naive_multiplication_ikj(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for i in range(len(a)):
        for k in range(len(a)):
            for j in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    # End measure
    exec_time = time.process_time() - start_time
    print("Naive multiply ikj ==> " + str(exec_time) + " s")
    return c


def naive_multiplication_jik(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for j in range(len(a)):
        for i in range(len(a)):
            for k in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    # End measure
    exec_time = time.process_time() - start_time
    print("Naive multiply jik ==> " + str(exec_time) + " s")
    return c


def better_naive_multiplication_ijk(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for i in range(len(a)):
        for j in range(len(a)):
            result = 0.0
            for k in range(len(a)):
                result += a[i][k] * b[k][j]
            c[i][j] = result
    # End measure
    exec_time = time.process_time() - start_time
    print("Better multiply jik ==> " + str(exec_time) + " s")
    return c


def naive_multiplication_jki(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for j in range(len(a)):
        for k in range(len(a)):
            for i in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    # End measure
    exec_time = time.process_time() - start_time
    print("Naive multiply jki ==> " + str(exec_time) + " s")
    return c


def better_naive_multiplication_jki(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for j in range(len(a)):
        for k in range(len(a)):
            temp = b[k][j]
            for i in range(len(a)):
                c[i][j] += a[i][k] * temp
    # End measure
    exec_time = time.process_time() - start_time
    print("Better multiply jki ==> " + str(exec_time) + " s")
    return c


def naive_multiplication_kij(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for k in range(len(a)):
        for i in range(len(a)):
            for j in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    # End measure
    exec_time = time.process_time() - start_time
    print("Naive multiply kij ==> " + str(exec_time) + " s")
    return c


def better_naive_multiplication_kij(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for k in range(len(a)):
        for i in range(len(a)):
            temp = a[i][k]
            result = 0.0
            for j in range(len(a)):
                c[i][j] += temp * b[k][j]
    # End measure
    exec_time = time.process_time() - start_time
    print("Better multiply kij ==> " + str(exec_time) + " s")
    return c


def naive_multiplication_kji(a, b):
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for k in range(len(a)):
        for j in range(len(a)):
            for i in range(len(a)):
                c[i][j] += a[i][k] * b[k][j]
    # End measure
    exec_time = time.process_time() - start_time
    print("Naive multiply kji ==> " + str(exec_time) + " s")
    return c


def numpy_naive_multiplication(a, b):
    start_time = time.process_time()
    # Start measure
    c = a.dot(b)
    # End measure
    exec_time = time.process_time() - start_time
    print("Size of matrix = " + str(len(a)) + " ==> " + str(exec_time) + " s")
    return c


def strassen_algorithm(x, y):
    if x.size == 1 or y.size == 1:
        return x.dot(y)

    n = len(x)

    if n % 2 == 1:
        x = np.pad(x, (0, 1), mode='constant')
        y = np.pad(y, (0, 1), mode='constant')

    m = int(np.ceil(n / 2))
    # Split matrix x
    a11 = x[: m, : m]
    a12 = x[: m, m:]
    a21 = x[m:, : m]
    a22 = x[m:, m:]
    # Split matrix y
    b11 = y[: m, : m]
    b12 = y[: m, m:]
    b21 = y[m:, : m]
    b22 = y[m:, m:]

    # Calculate the seven products of matrices
    p1 = strassen_algorithm(a11, b12 - b22)
    p2 = strassen_algorithm(a11 + a12, b22)
    p3 = strassen_algorithm(a21 + a22, b11)
    p4 = strassen_algorithm(a22, b21 - b11)
    p5 = strassen_algorithm(a11 + a22, b11 + b22)
    p6 = strassen_algorithm(a12 - a22, b21 + b22)
    p7 = strassen_algorithm(a12 - a21, b11 + b12)

    # Initialize matrix to store results
    c = np.zeros((2 * m, 2 * m), dtype=np.int32)
    # Calculate each quadrant inside matrix
    c[: m, : m] = p5 + p4 - p2 + p6  # c11
    c[: m, m:] = p1 + p2  # c12
    c[m:, : m] = p3 + p4  # c21
    c[m:, m:] = p1 + p5 - p3 - p7  # c22

    return c[: n, : n]


def blocks_multiplication(a, b, block_size):
    start_time = time.process_time()
    # Start measure
    N = len(a)
    c = np.zeros((N, N), dtype=np.int32)

    for jj in range(0, N, block_size):
        for kk in range(0, N, block_size):
            for i in range(0, N):
                for j in range(jj, min(jj + block_size, N)):
                    result = 0.0
                    for k in range(kk, min(kk + block_size, N)):
                        result += a[i][k] * b[k][j]
                    c[i][j] = c[i][j] + result
    # End measure
    exec_time = time.process_time() - start_time
    print("Blocks " + str(block_size) + " multiply ==> " + str(exec_time) + " s")
    return c


def multiplication_transpose_ikj(a, b):
    b = np.transpose(b)
    c = np.zeros((len(a), len(a)), dtype=np.int32)
    # Start measure
    start_time = time.process_time()
    for i in range(len(a)):
        for k in range(len(a)):
            for j in range(len(a)):
                c[i][j] += a[i][k] * b[j][k]
    # End measure
    exec_time = time.process_time() - start_time
    print("Naive multiply ijk ==> " + str(exec_time) + " s")
    return c


def measurement_between_ijk():
    print("Differences between naive multiplication")
    for i in range(50, 1001, 50):
        print("#Size = " + str(i) + ":")
        a = random_matrix(i)
        b = random_matrix(i)
        naive_multiplication_ijk(a, b)
        naive_multiplication_ikj(a, b)
        naive_multiplication_jik(a, b)
        naive_multiplication_jki(a, b)
        naive_multiplication_kij(a, b)
        naive_multiplication_kji(a, b)
    return 0


def measurement_better_naive():
    print("Differences between better naive multiplication ")
    for i in range(50, 1001, 50):
        print("#Size = " + str(i) + ":")
        a = random_matrix(i)
        b = random_matrix(i)
        better_naive_multiplication_jik(a, b)
        better_naive_multiplication_jki(a, b)
        better_naive_multiplication_kij(a, b)


def measurement_strassen():
    print("Strassen multiplication ")
    for i in range(50, 1001, 50):
        print("#Size = " + str(i) + ":")
        a = random_matrix(i)
        b = random_matrix(i)
        start_time = time.process_time()
        strassen_algorithm(a, b)
        exec_time = time.process_time() - start_time
        print("Strassen multiply ==> " + str(exec_time) + " s")


def measurement_blocks_multiply():
    print("Blocks multiply multiplication ")
    for i in range(50, 1001, 50):
        print("#Size = " + str(i) + ":")
        a = random_matrix(i)
        b = random_matrix(i)
        blocks_multiplication(a, b, 16)
        blocks_multiplication(a, b, 32)
        blocks_multiplication(a, b, 64)
        blocks_multiplication(a, b, 128)


def measurement_transpose():
    print("Transpose multiply multiplication ")
    for i in range(800, 1001, 50):
        print("#Size = " + str(i) + ":")
        a = random_matrix(i)
        b = random_matrix(i)
        multiplication_transpose_ikj(a, b)

        # c2 = naive_multiplication_ikj(a, b)
        # print(c1)
        # print(c2)


def random_matrix(size):
    matrix = np.random.randint(50, size=(size, size))
    return matrix


def main():
    measurement_transpose()
    # measurement_between_ijk()
    # measurement_better_naive()
    # measurement_blocks_multiply()
    # measurement_strassen()


if __name__ == "__main__":
    main()
