from numpy import array, zeros, diag, diagflat, dot, linalg, copy, matmul

def sor(A, b, x, omega):
	while True:
		xp = copy(x)
		for i in range(3):
			d = 0
			for j in range(3):
				if j != i:
					d += A[i][j] * x[j]
			x[i] = (1 - omega) * x[i] + (omega / A[i][i]) * (b[i] - d)
		if (linalg.norm(x-xp)<= 0.0001):
			return x
