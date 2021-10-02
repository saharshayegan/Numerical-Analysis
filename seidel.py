from numpy import array, zeros, diag, diagflat, dot, linalg, copy, matmul

def seidel(A,b,x):
	n = len(A)
	while True:
		xp = copy(x)
		for j in range(0, n):
			d = b[j]
			for i in range(0, n):
				if(j != i):
					d-=A[j][i] * x[i]
			x[j] = d / A[j][j]
		if (linalg.norm(x-xp)<= 0.0001):
		   return x
