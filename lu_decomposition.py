from numpy import array, zeros

def lu_decomposition(A2,n):
	U = zeros([n, n])
	L = zeros([n, n])
	for j in range(n):
		L[j][j] = 1.0
		for i in range(j+1):
			s1 = sum(U[k][j] * L[i][k] for k in range(i))
			U[i][j] = A2[i][j] - s1
		for i in range(j, n):
			s2 = sum(U[k][j] * L[i][k] for k in range(j))
			L[i][j] = (A2[i][j] - s2) / U[j][j]
	return (L,U)
