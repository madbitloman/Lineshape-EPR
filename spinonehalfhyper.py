from __future__ import division
# from ggplot import *
# from numbapro import cuda
# from numba import *
import numpy as np 
import itertools
from numpy.random import rand
from pylab import *
import time
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.color'] = 'r'
time_start=time.clock() 

"""Parametrs Set"""
# trate=0.05
H=20
# H=80
aval=6
m0val,m0sval=2,2
# kom = trate
# W=np.array([[-trate,trate],[trate,-trate]])

# AdjMat=loadtxt("Adj.txt", delimiter=",")
# W=AdjMat*trate
# W=np.array([[-4*kom,kom,kom,kom,kom,0.0],
# 		   [kom,-4*kom,kom,kom,0.0,kom], 
# 		   [kom,kom,-4*kom,0.0,kom,kom],  
# 		   [kom,kom,0.0,-4*kom,kom,kom],   
# 		   [kom,0.0,kom,kom,-4*kom,kom],
# 		   [0.0,kom,kom,kom,kom,-4*kom]])

	

def angles(a):

	if a==6:
		alpha, beta, gama=np.array([0,0,pi/2,3*pi/2,pi,0]), np.array([0,0.5*pi,0.5*pi,0.5*pi,pi*0.5,pi]), np.array([0,0,0.5*pi,1.5*pi,pi,0])
	elif a==48:
		angles=loadtxt("Euler.txt", delimiter=",")
	 	alpha,beta,gama=angles[:,0], angles[:,1],angles[:,2]
	return alpha, beta, gama		


def H20zem(g20,g22):

	H20zeeman=np.zeros(len(alpha), dtype=complex)
	for i in range(len(alpha)):
		H20zeeman[i]=(0.5*g20*(3*np.cos(beta[i])**2-1)+g22*np.sqrt(1.5)*np.cos(2*gama[i])*np.sin(beta[i])**2)
	return H20zeeman

def H21zem(g20,g22):
	i1=complex(0,1)
	H21zeeman=np.empty(len(alpha), dtype=complex)
	for i in range(len(alpha)):
		H21zeeman[i]=(-1)*np.exp(i1*alpha[i])*(g20*np.exp(-i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1-np.cos(beta[i])))+\
			np.exp(i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1+np.cos(beta[i])))*g22)
	return H21zeeman

def A20hyp():
	A20hyper=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A20hyper[i]=(0.5*a20*(3*np.cos(beta[i])**2-1)+a22*sqrt(1.5)*np.cos(2*gama[i])*np.sin(beta[i])**2)
	return A20hyper	

def A2p2hyp():
	i1=complex(0,1)
	A2p2hyper=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A2p2hyper[i]=np.exp(i1*2*alpha[i])*((0.5*(1+np.cos(beta[i])**2)*np.cos(2*gama[i])+i1*np.cos(beta[i])*np.sin(2*gama[i]))*a22+sqrt(3/8)*np.sin(beta[i])**2*a20)
	return A2p2hyper	

def A21hyp():
	i1=complex(0,1)
	A21hyper=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A21hyper[i]=(-1)*np.exp(i1*alpha[i])*(a20*np.exp(-i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1-np.cos(beta[i])))+\
			np.exp(i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1+np.cos(beta[i])))*a22)
	return A21hyper






a,b = aval, aval
m0,m1,m00,m11 = m0val, m0val, m0val, m0val
m0s,m1s,m00s, m11s = m0sval, m0sval, m0sval, m0sval
max=a*m0*m1*m0s*m1s

Hmm=np.array([[0,0.5],[0.5,0]])
Hm=np.array([[1,0],[0,1]])
# Hmm=np.array([[0.5,0.5],[0.5,-0.5]])

Hminv=np.linalg.inv(Hm)
Hmminv=np.linalg.inv(Hmm)
# Hmone=np.array([[0.4082,0.701,0.5774],[0.8165,0.0,-0.5774],[0.4082,-0.701,0.5774]])
Hmone=np.array([[1,0,0],[0,1,0],[0,0,1]])
# Hmone=np.array([[0,1,0],[1,0,1],[0,1,0]])*(1/np.sqrt(2))
# Hminvone=np.array([[0,1,0],[1,0,1],[0,1,0]])*(1/np.sqrt(2))
Hminvone=np.linalg.inv(Hmone)

def MatSpinHalf(W,p,H00zeeman,H20zeeman,H2p1zeeman,H2m1zeeman,A00hyper,A20hyper,A2p2hyper,A2m2hyper, A2p1hyper, A2m1hyper):
	"""This method assign Spin 1/2 to 1/2 Coupling Matrix"""
	P=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	WW=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	A=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	B=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	# C=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	# CC=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")

	for i,k,l,ks,ls,j,m,n,ms,ns in itertools.product(xrange(0,a,1),xrange(0,m0,1),xrange(0,m1,1),xrange(0,m0s,1),xrange(0,m1s,1),xrange(0,b,1),xrange(0,m00,1),xrange(0,m11,1),xrange(0,m00s,1),xrange(0,m11s,1)):
		if i==j and k==m and l==n and ks==ms and ls==ns:	
			P[i,k,l,ks,ls,j,m,n,ms,ns]=p
			""" First Delta """
		if i==j and l==n and ls==ns and k==0 and m==0 and ks==0 and ms==0:	
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*0.25
		if i==j and l==n and ls==ns and k==0 and m==0 and ks==1 and ms==1:	
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(0.5)+(A00hyper+A20hyper[i])*(-0.25)
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==0 and ms==0:	
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(-0.25)
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==1 and ms==1:	
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(0.25)
			""" Second Delta """
		if i==j and k==m and ks==ms and l==0 and n==0 and ls==0 and ns==0:	
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*0.25
		if i==j and k==m and ks==ms and l==0 and n==0 and ls==1 and ns==1:	
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(0.5)+(A00hyper+A20hyper[i])*(-0.25)
		if i==j and k==m and ks==ms and l==1 and n==1 and ls==0 and ns==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(-0.25)
		if i==j and k==m and ks==ms and l==1 and n==1 and ls==1 and ns==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(0.25)

		# 	"""Rising and Lowering Operators are here"""
		# 	""" First Delta """
		if i==j and l==n and ls==ns and k==0 and m==0 and ks==0 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]+0.5*A2p1hyper[i]		# S+ and S+Iz
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==0 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]+(-0.5)*A2p1hyper[i]	# S+ and S+Iz
		if i==j and l==n and ls==ns and k==0 and m==0 and ks==1 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]+0.5*A2m1hyper[i]		# S- and S-Iz
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==1 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]+(-0.5)*A2m1hyper[i]	# S- and S-Iz	
			
		if i==j and l==n and ls==ns and k==0 and m==1 and ks==0 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*A2p1hyper[i]						# SzI+
		if i==j and l==n and ls==ns and k==0 and m==1 and ks==1 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= -0.5*A2p1hyper[i]					# SzI+
		if i==j and l==n and ls==ns and k==1 and m==0 and ks==0 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= 0.5*A2m1hyper[i]					# SzI-
		if i==j and l==n and ls==ns and k==1 and m==0 and ks==1 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= -0.5*A2m1hyper[i]					# SzI-	
			
		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==0 and ms==1:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=A2p2hyper[i]							# S+I+
		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==1 and ms==0:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=A2m2hyper[i] 						# S-I-	

		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==0 and ms==1:	
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=A00hyper*0.5 - A20hyper[i]*0.25		# S+I-
		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==1 and ms==0:	
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=A00hyper*0.5 - A20hyper[i]*0.25		# S-I+
		
		# # # 	""" Second Delta """
		if i==j and k==m and ks==ms and n==0 and l==0 and ns==0 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]+0.5*A2p1hyper[i]		# S+ and S+Iz
		if i==j and k==m and ks==ms and n==1 and l==1 and ns==0 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]+(-0.5)*A2p1hyper[i]	# S+ and S+Iz
		if i==j and k==m and ks==ms and n==0 and l==0 and ns==1 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]+0.5*A2m1hyper[i]		# S- and S-Iz
		if i==j and k==m and ks==ms and n==1 and l==1 and ns==1 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]+(-0.5)*A2m1hyper[i]	# S- and S-Iz	
			
		if i==j and k==m and ks==ms and n==0 and l==1 and ns==0 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= 0.5*A2p1hyper[i]					# SzI+
		if i==j and k==m and ks==ms and n==0 and l==1 and ns==1 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= -0.5*A2p1hyper[i]					# SzI+
		if i==j and k==m and ks==ms and n==1 and l==0 and ns==0 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= 0.5*A2m1hyper[i]					# SzI-
		if i==j and k==m and ks==ms and n==1 and l==0 and ns==1 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= -0.5*A2m1hyper[i]					# SzI-	
			
		

		# if i==j and k==m and ks==ms and n==0 and l==1 and ns==0 and ls==1:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=A2p2hyper[i]							# S+I+
		# if i==j and k==m and ks==ms and n==1 and l==0 and ns==1 and ls==0:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=A2m2hyper[i] 						# S-I-	

		# if i==j and k==m and ks==ms and n==1 and l==0 and ns==0 and ls==1:	
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=A00hyper*0.5 - A20hyper[i]*0.25		# S+I-
		# if i==j and k==m and ks==ms and n==0 and l==1 and ns==1 and ls==0:	
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=A00hyper*0.5 - A20hyper[i]*0.25		# S-I+
		
			"""Frequencies"""
		if k==m and l==n and ks==ms and ls==ns:
			WW[i,k,l,ks,ls,j,m,n,ms,ns]=-W[i,j]	
	
	C=-complex(0,1)*(A-B)+P+WW

	"""We reshape Array Here"""
	AA=np.reshape(C,(max,max),order='F') # F is for Fortran-like indexing(other option is C-like)
	AAinv=np.linalg.inv(AA)
	AA2ndresh=np.reshape(AAinv,(a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),order='F')
	return AA2ndresh

def MatSpinOne(p, H00zeeman,H20zeeman,H2p1zeeman,H2m1zeeman,A00hyper,A20hyper,A2p2hyper,A2m2hyper, A2p1hyper, A2m1hyper):
	"""This method assighn Spin 1/2 to 1 Coupling Matrix"""
	P=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	WW=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	A=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	B=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	D=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	DD=np.zeros((a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),dtype="complex")
	for i,k,l,ks,ls,j,m,n,ms,ns in itertools.product(xrange(0,a,1),xrange(0,m0,1),xrange(0,m1,1),xrange(0,m0s,1),xrange(0,m1s,1),xrange(0,b,1),xrange(0,m00,1),xrange(0,m11,1),xrange(0,m00s,1),xrange(0,m11s,1)):
		# k=m0
		# l=m1
		# ks=m0s
		# ls=m1s
		# m=m00
		# n=m11
		# ms=m00s
		# ns=m11
		if i==j and k==m and l==n and ks==ms and ls==ns:	
			P[i,k,l,ks,ls,j,m,n,ms,ns]=p
			""" First Delta """
		if i==j and l==n and ls==ns and k==0 and m==0 and ks==0 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*0.5
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==0 and ms==0: 					
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5
		if i==j and l==n and ls==ns and k==2 and m==2 and ks==0 and ms==0:  
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*(-0.5)
		if i==j and l==n and ls==ns and k==0 and m==0 and ks==1 and ms==1:  
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(-0.5)
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==1 and ms==1: 		
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)
		if i==j and l==n and ls==ns and k==2 and m==2 and ks==1 and ms==1: 
			A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(0.5)
			

		if i==j and l==n and ls==ns and k==0 and m==0 and ks==0 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i] + A2p1hyper[i]			# S+ and S+Iz
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==0 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]							# S+ and S+Iz
		if i==j and l==n and ls==ns and k==2 and m==2 and ks==0 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]-A2p1hyper[i]				# S+ and S+Iz
		
		if i==j and l==n and ls==ns and k==0 and m==0 and ks==1 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i] + A2m1hyper[i]			# S- and S-Iz
		if i==j and l==n and ls==ns and k==1 and m==1 and ks==1 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]							# S- and S-Iz
		if i==j and l==n and ls==ns and k==2 and m==2 and ks==1 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]-A2m1hyper[i]				# S- and S-Iz	

		if i==j and l==n and ls==ns and k==0 and m==1 and ks==0 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2p1hyper[i]			  		# SzI+
		if i==j and l==n and ls==ns and k==1 and m==2 and ks==0 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2p1hyper[i]					# SzI+
		if i==j and l==n and ls==ns and k==0 and m==1 and ks==1 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2p1hyper[i]				# SzI+
		if i==j and l==n and ls==ns and k==1 and m==2 and ks==1 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2p1hyper[i]				# SzI+

		if i==j and l==n and ls==ns and k==1 and m==0 and ks==0 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2m1hyper[i]			  		# SzI-
		if i==j and l==n and ls==ns and k==2 and m==1 and ks==0 and ms==0:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2m1hyper[i]					# SzI-
		if i==j and l==n and ls==ns and k==1 and m==0 and ks==1 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2m1hyper[i]				# SzI-
		if i==j and l==n and ls==ns and k==2 and m==1 and ks==1 and ms==1:
			A[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2m1hyper[i]				# SzI-	
	
				
		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==0 and ms==1:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2p2hyper[i]						#S+I+
		# if i==j and l==n and ls==ns and k==1 and m==2 and ks==0 and ms==1:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2p2hyper[i]						#S+I+

		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==1 and ms==0:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2m2hyper[i] 					#S-I-
		# if i==j and l==n and ls==ns and k==2 and m==1 and ks==1 and ms==0:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2m2hyper[i]						#S-I-	

		# if i==j and l==n and ls==ns and k==1 and m==2 and ks==1 and ms==0:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i])	#S+I-
		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==1 and ms==0:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) 	#S+I-

		# if i==j and l==n and ls==ns and k==2 and m==1 and ks==0 and ms==1:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i])	#S-I+
		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==0 and ms==1:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) 	#S-I+					
			


			""" Second Delta """
		if i==j and k==m and ks==ms and l==0 and n==0 and ls==0 and ns==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*0.5
		if i==j and k==m and ks==ms and l==1 and n==1 and ls==0 and ns==0: 		
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5
		if i==j and k==m and ks==ms and l==2 and n==2 and ls==0 and ns==0:  
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5-(A00hyper+A20hyper[i])*-0.5
		if i==j and k==m and ks==ms and l==0 and n==0 and ls==1 and ns==1:  
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(-0.5)
		if i==j and k==m and ks==ms and l==1 and n==1 and ls==1 and ns==1: 		
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)
		if i==j and k==m and ks==ms and l==2 and n==2 and ls==1 and ns==1: 
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(0.5)

		
		if i==j and k==m and ks==ms and n==0 and l==0 and ns==0 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]+A2p1hyper[i]				# S+ and S+Iz
		if i==j and k==m and ks==ms and n==1 and l==1 and ns==0 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]							# S+ and S+Iz
		if i==j and k==m and ks==ms and n==2 and l==2 and ns==0 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2p1zeeman[i]-A2p1hyper[i]				# S+ and S+Iz	

		if i==j and k==m and ks==ms and n==0 and l==0 and ns==1 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]+A2m1hyper[i]				# S- and S-Iz
		if i==j and k==m and ks==ms and n==1 and l==1 and ns==1 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]							# S- and S-Iz
		if i==j and k==m and ks==ms and n==2 and l==2 and ns==1 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]= H2m1zeeman[i]+-A2m1hyper[i]				# S- and S-Iz

		if i==j and k==m and ks==ms and n==0 and l==1 and ns==0 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2p1hyper[i]			  		# SzI+
		if i==j and k==m and ks==ms and n==1 and l==2 and ns==0 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2p1hyper[i]					# SzI+
		if i==j and k==m and ks==ms and n==0 and l==1 and ns==1 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2p1hyper[i]				# SzI+
		if i==j and k==m and ks==ms and n==1 and l==2 and ns==1 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2p1hyper[i]				# SzI+

		if i==j and k==m and ks==ms and n==1 and l==0 and ns==0 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2m1hyper[i]			  		# SzI-
		if i==j and k==m and ks==ms and n==2 and l==1 and ns==0 and ls==0:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2m1hyper[i]					# SzI-
		if i==j and k==m and ks==ms and n==1 and l==0 and ns==1 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2m1hyper[i]				# SzI-
		if i==j and k==m and ks==ms and n==2 and l==1 and ns==1 and ls==1:
			B[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*sqrt(2)*A2m1hyper[i]				# SzI-					
		
		
		# if i==j and k==m and ks==ms and n==0 and l==1 and ns==0 and ls==1:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2p2hyper[i]						#S+I+
		# if i==j and k==m and ks==ms and n==1 and l==2 and ns==0 and ls==1:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2p2hyper[i]						#S+I+

		# if i==j and k==m and ks==ms and n==1 and l==0 and ns==1 and ls==0:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2m2hyper[i] 					#S-I-	
		# if i==j and k==m and ks==ms and n==2 and l==1 and ns==1 and ls==0:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*A2m2hyper[i] 					#S-I-	

		# if i==j and k==m and ks==ms and n==1 and l==0 and ns==0 and ls==1:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) 	#S+I-	
		# if i==j and k==m and ks==ms and n==2 and l==1 and ns==0 and ls==1:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]= sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i])	#S+I-

		# if i==j and k==m and ks==ms and n==0 and l==1 and ns==1 and ls==0:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) 	#S-I+	
		# if i==j and k==m and ks==ms and n==1 and l==2 and ns==1 and ls==0:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) 	#S-I+				
		
		
		if k==m and l==n and ks==ms and ls==ns:
			WW[i,k,l,ks,ls,j,m,n,ms,ns]=-W[i,j]	
	
	C=P+WW-complex(0,1)*(A-B)
	AA=np.reshape(C,(max,max),order='F')
	AAinv=np.linalg.inv(AA)
	AA2ndresh=np.reshape(AAinv,(a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),order='F')
	return AA2ndresh


def SumMa(A):
	"""
	This method summs up the Matrix

	"""
	su=0
	if m0val==2:
		for k,l,ks,ls,m,n,ms,ns,i,j in itertools.product(xrange(0,m0,1),xrange(0,m1,1),xrange(0,m0s,1),\
			xrange(0,m1s,1),xrange(0,m00,1),xrange(0,m11,1),xrange(0,m00s,1),xrange(0,m11s,1),xrange(0,a,1),xrange(0,b,1)):
			# if m!=n and k!=l:

				su+=Hm[ls,ks]*Hmm[l,k]*A[i,k,l,ks,ls,j,m,n,ms,ns]*Hminv[ms,ns]*Hmminv[m,n]*(1/(np.sqrt(2)*aval)) #matrix spin-1/2
				# su+=Hm[l,k]*Hmm[ls,ks]*A[i,k,l,ks,ls,j,m,n,ms,ns]*Hminv[m,ns]*Hmminv[m,n]*(1/(np.sqrt(2)*aval)) #matrix spin-1/2

				# su+=Hmm[ls,ks]*A[i,k,l,ks,ls,j,m,n,ms,ns]*Hmminv[ms,ns]*(1/(np.sqrt(2)*aval)) #matrix spin-1/2 isolated spin
		

	elif m0val==3:
		for k,l,ks,ls,m,n,ms,ns,i,j in itertools.product(xrange(0,m0,1),xrange(0,m1,1),xrange(0,m0s,1),\
			xrange(0,m1s,1),xrange(0,m00,1),xrange(0,m11,1),xrange(0,m00s,1),xrange(0,m11s,1),xrange(0,a,1),xrange(0,b,1)):
			# if m!=n and k!=l and ms!=ns and ks!=ls:	
				
				su+=Hmm[ls,ks]*Hmone[l,k]*A[i,k,l,ks,ls,j,m,n,ms,ns]*Hmminv[ms,ns]*Hminvone[m,n]*(1/(np.sqrt(3)*aval))	#matrix spin-1
				# su+=Hmm[l,k]*Hmone[ls,ks]*A[i,k,l,ks,ls,j,m,n,ms,ns]*Hmminv[m,n]*Hminvone[ms,ns]*(1/(np.sqrt(3)*aval))	#matrix spin-1
	return su

def Main():

	global alpha, beta,gama, H, W, a00, a22, a20, trates,W
	
	k=0.566/0.4 #scaling factor fro A-tensor values       
	ge=2.0002   # electron g-value 
	
	""" g-tensor and A-tensor principal values"""
	gxx,gyy,gzz=2.0089, 2.0061, 2.00032
	axx,ayy,azz=5.5*k, 4.5*k, 34*k

	# gxx,gyy,gzz=2.0061, 2.0061, 2.0032
	# axx,ayy,azz=5.5*k, 4.0*k, 29.8*k

	"""""" 
	
	a00,a22,a20= -ge*(axx+ayy+azz)/sqrt(3), 0.5*ge*(axx-ayy), 	ge*(azz-0.5*(axx+ayy))*sqrt(2/3)
	g00,g22,g20= -(gxx+gyy+gzz)/sqrt(3), 	0.5*(gxx-gyy),		(gzz-0.5*(gxx+gyy))*sqrt(2/3)
	
	alpha, beta, gama=np.array([0,0,pi/2,3*pi/2,pi,0]), np.array([0,0.5*pi,0.5*pi,0.5*pi,pi*0.5,pi]), np.array([0,0,0.5*pi,1.5*pi,pi,0])
	# angles=loadtxt("Euler.txt", delimiter=",")
	# alpha,beta,gama=angles[:,0], angles[:,1],angles[:,2]
	
	H00zeeman=-g00*H/np.sqrt(3)
	H20zeeman= H*np.sqrt(2/3)*H20zem(g20,g22)
	# H2p1zeeman=-0.5*H*H21zem(g20,g22)
	# H2m1zeeman= 0.5*H*np.conjugate(H2p1zeeman)
	
	A00hyper=-a00/sqrt(3)
	A20hyper=np.sqrt(2/3)*A20hyp()
	# A2p2hyper=0.5*A2p2hyp()
	# A2m2hyper=0.5*np.conjugate(A2p2hyper)
	A2p1hyper=-0.5*A21hyp()
	A2m1hyper=0.5*np.conjugate(A2p1hyper)

	# A00hyper=0
	# A20hyper=np.zeros(aval)
	A2p2hyper=np.zeros(aval)
	A2m2hyper=np.zeros(aval)
	# A2p1hyper=np.zeros(aval)
	# A2m1hyper=np.zeros(aval)
	H2p1zeeman=np.zeros(aval)
	H2m1zeeman=np.zeros(aval)

	""" Computation starts here"""
	# x=np.arange(79900,80400,1)
	# x=np.arange(0,15000,1)
	
	# x=np.arange(79940,80390,0.1) #40000
	# trates=[0.001,0.005,0.008,0.01,0.02,0.03,0.05,100]

	# trates=[0.1,1,3,10,100]
	# trates=[0.1,1,3,5,10,100]
	# trates=[100,1000,1000000]
	trates=[100]
	# x=np.arange(50,270,0.1)
	# x=np.arange(79900,80400,1)
	x = np.arange(0,100,0.1)
	y=np.zeros(len(x))
	AdjMat=loadtxt("Adj.txt", delimiter=",")
	for z in range(0, len(trates)):
		trate=trates[z]
		kom=trate
		f=trate*1
		m=trate
		s=trate*1
	
		# W=AdjMat*trate
		W=np.array([[-(f+s+m),m,0,s,0,f],
		   [m,-(f+s+m),s,0,f, 0], 
		   [0,s,-(f+s+m),f,0,m],  
		   [s,0,f,-(f+m+s),m,0],   
		   [0,f,0,m,-(f+m+s),s],
		   [f,0,m,0,s,-(f+m+s)]])
		for i in range(0,len(x)):
			p=(x[i]-0.001*complex(0,1))*complex(0,1)
			A=MatSpinHalf(W,p, H00zeeman,H20zeeman,H2p1zeeman,H2m1zeeman,A00hyper,A20hyper,A2p2hyper,A2m2hyper, A2p1hyper, A2m1hyper)		
			
			y[i]=np.real(SumMa(A))			
		ydiv=np.diff(y)
		xdiv=np.arange(0,len(ydiv))
		# x=np.arange(15.5,16.5,0.001)
		figure()
		# plt.plot(xdiv, ydiv)
		plt.plot(x,y)
		# plot(x,y, 'b')
	show()
	
Main()	
time_elapsed = (time.clock() - time_start)
print (time_elapsed)
  