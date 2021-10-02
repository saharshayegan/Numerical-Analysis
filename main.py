from numpy import array, zeros, diag, diagflat, dot, linalg, copy, matmul
import time
from scipy.linalg import lu
from sor import sor
from seidel import seidel
from jacobi import jacobi
from lu_decomposition import lu_decomposition

def generate_A(n):
	main_diagonal = []
	for i in range(n):
		main_diagonal.append(4)
	upper_diagonal = []
	for i in range(n-1):
		upper_diagonal.append(-1)
	lower_diagonal = []
	for i in range(n-1):
		lower_diagonal.append(-1)
	A = diag(main_diagonal) + diag(upper_diagonal, 1) + diag(lower_diagonal, -1)
	return A

def generate_b(n):
	b = []
	for i in range(n):
		b.append(1)
	return b


def pivotize(m):
	n = len(m)
	ID = [[float(i == j) for i in range(n)] for j in range(n)]
	for j in range(n):
		row = max(range(j, n), key=lambda i: abs(m[i][j]))
		if j != row:
			ID[j], ID[row] = ID[row], ID[j]
	return ID

def lu__(A):
	p, l, u = lu(A)
	print("actual L:\n", l)
	print("actual U:\n", u)

# def __main__():
for n in [ 4, 16, 64, 256, 1024]:
	print("\n********************* n = ", n, " *********************")
	A = generate_A(n)
	b = generate_b(n)
	# print("A:\n", A)
	# print("b:\n", b)
	x = zeros(len(A[0]))

	start=time.time()
	jacobi_solution = jacobi(A,b,x)
	end=time.time()
	jacobi_time = end-start


	start=time.time()
	seidel_solution = seidel(A,b,x)
	end=time.time()
	seidel_time = end-start


	start-time.time()
	sor_solution = sor(A,b,x,1.5)
	end=time.time()
	sor_time = end-start

	print("Times:\njacobi: ", jacobi_time,"\nseidel: ", seidel_time,"\n   sor: ",sor_time)

	print("actual solution:\n", linalg.solve(A,b))
	print("jacobi Solution:\n", jacobi_solution)
	print("seidel solution:\n", seidel_solution)
	print("sor solution:\n", sor_solution)

	print("\n")

	start=time.time()
	(L, U) = lu_decomposition(A,n)
	end=time.time()
	lu1_time = end-start

	print("LU decomposition results without partial pivoting")
	print("L:\n", L)
	print("U:\n", U)

	print("\n")

	start=time.time()
	P = pivotize(A)
	A2 = matmul(P, A)
	(L, U) = lu_decomposition(A2,n)
	end=time.time()
	lu2_time = end-start

	print("LU decomposition results with partial pivoting")
	print("L:\n", L)
	print("U:\n", U)

	print("\n")

	print("LU decomposition actual results")
	lu__(A)

	print("\n")

	print("Time with pivoting:    ", lu2_time,"\n"
		  "Time without pivoting: ", lu1_time)
