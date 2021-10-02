from numpy import array, zeros, diag, diagflat, dot, linalg, copy, matmul

def jacobi(A,b,x):
	D = diag(A)
	R = A - diagflat(D)
	while True:
		xp = copy(x)
		x = (b - dot(R,x)) / D
		if (linalg.norm(x-xp)<= 0.0001):
			return x
