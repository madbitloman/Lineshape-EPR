##############################################################################################################
# DESCRIPTION IS HERE TO DO
##############################################################################################################
from __future__ import division
import numpy as np 
import numexpr as ne
import itertools
from numpy.random import rand
from pylab import *
import time
import matplotlib 
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
# import matplotlib.pyplot as plt
import collections
##############################################################################################################

# remove time.clock if needed
# it is only to check the efficiency 
time_start=time.clock() 

# # class ValuesInitialization:
# # Simple class to initialize all the "global" variables
# H=80
# aval=6
# m0val,m0sval=2,2
# ##############################################################################################################
# a,b = aval, aval
# m0,m1,m00,m11 = m0val, m0val, m0val, m0val
# m0s,m1s,m00s, m11s = m0sval, m0sval, m0sval, m0sval
# max = a*m0*m1*m0s*m1s
# Hmm = np.array([[0,0.5],[0.5,0]])
# Hm = np.array([[1,0],[0,1]])
# # Hmm = np.array([[0.5,0.5],[0.5,-0.5]])
# Hminv = np.linalg.inv(Hm)
# Hmminv = np.linalg.inv(Hmm)
# Hmone = np.array([[1,0,0],[0,1,0],[0,0,1]])
# # Hmone = np.array([[0,1,0],[1,0,1],[0,1,0]])*(1/np.sqrt(2))
# # Hminvone = np.array([[0,1,0],[1,0,1],[0,1,0]])*(1/np.sqrt(2))
# Hminvone = np.linalg.inv(Hmone)



def Main():
	k = 0.566/0.4 #scaling factor fro A-tensor values       
	ge = 2.0002   # electron g-value 
	gxx,gyy,gzz=2.00032, 2.00032, 2.00032
	axx,ayy,azz=5.5*k, 5.5*k, 5.5*k
	# gxx,gyy,gzz=2.0089, 2.0061, 2.00032
	# axx,ayy,azz=5.5*k, 4.5*k, 34*k
	# gxx,gyy,gzz=2.0061, 2.0061, 2.0032
	# axx,ayy,azz=5.5*k, 4.0*k, 29.8*k
	a00,a22,a20 = -ge*(axx+ayy+azz)/sqrt(3), 0.5*ge*(axx-ayy), 	ge*(azz-0.5*(axx+ayy))*sqrt(2/3)
	g00,g22,g20 = -(gxx+gyy+gzz)/sqrt(3), 	0.5*(gxx-gyy),		(gzz-0.5*(gxx+gyy))*sqrt(2/3)

	angles_total_out = anglesLoader(aval)
	alpha,beta,gama = angles_total_out.alpha,angles_total_out.beta,angles_total_out.gama
	
	
	H00zeeman = -g00*H/np.sqrt(3)
	H20zeeman = H*np.sqrt(2/3)*H20zem(g20,g22,alpha,beta,gama)
	# H2p1zeeman=-0.5*H*H21zem(g20,g22,alpha,beta,gama)
	# H2m1zeeman= 0.5*H*np.conjugate(H2p1zeeman)
	
	A00hyper = -a00/sqrt(3)
	A20hyper = np.sqrt(2/3)*A20hyp(a20,a22,alpha,beta,gama)
	# A2p2hyper=0.5*A2p2hyp(a20,a22,alpha,beta,gama)
	# A2m2hyper=0.5*np.conjugate(A2p2hyper)
	A2p1hyper = -0.5*A21hyp(a20,a22,alpha,beta,gama)
	A2m1hyper = 0.5*np.conjugate(A2p1hyper)
	
	# A00hyper=0
	# A20hyper=np.zeros(aval)
	A2p2hyper = np.zeros(aval)
	A2m2hyper = np.zeros(aval)
	# A2p1hyper=np.zeros(aval)
	# A2m1hyper=np.zeros(aval)
	H2p1zeeman = np.zeros(aval)
	H2m1zeeman = np.zeros(aval)

	""" Computation starts here"""
	# x=np.arange(79900,80400,1)
	
	# x=np.arange(79940,80390,0.1) #40000
	# trates=[0.001,0.005,0.008,0.01,0.02,0.03,0.05,100]

	# trates=[0.1,1,3,5,10,100]
	trates=[0.1]

	freq_range=np.arange(50,270,10)
	# x = np.arange(79900,80400,0.1)
	intensity = np.zeros(len(freq_range))
	for trate in trates:
		f,m,s = 1*trate,1*trate,1*trate

		W = matrixLoader(aval,trate,f,s,m)
		
		i=0
		for freq in freq_range:
			p=(freq-0.001*complex(0,1))*complex(0,1)
			if m0val == 2:		
				A = MatSpinHalfInit(W,p, H00zeeman,H20zeeman,H2p1zeeman,H2m1zeeman,A00hyper,A20hyper,A2p2hyper,A2m2hyper, A2p1hyper, A2m1hyper)		
			elif m0val == 3:
				A = MatSpinOneInit(p, H00zeeman,H20zeeman,H2p1zeeman,H2m1zeeman,A00hyper,A20hyper,A2p2hyper,A2m2hyper, A2p1hyper, A2m1hyper)	
			intensity[i] = np.real(SumMatrix(A))	
			i=i+1
						
		
		# ydiv = np.diff(y)
		# xdiv = np.arange(0,len(ydiv))
		plotResults(freq_range,intensity)

		 

def plotResults(x,y):
	fig=plt.figure()
	plt.plot(x, y)		
	show()

##############################################################################################################		
# W-matrix and angles loaders
##############################################################################################################
def matrixLoader(aval,trate,f,s,m):
	W = np.zeros((aval,aval), dtype=complex)
	if aval == 6:
		W=np.array([[-(f+s+m),m,0,s,0,f],
			   [m,-(f+s+m),s,0,f, 0], 
			   [0,s,-(f+s+m),f,0,m],  
			   [s,0,f,-(f+m+s),m,0],   
			   [0,f,0,m,-(f+m+s),s],
			   [f,0,m,0,s,-(f+m+s)]])
	elif aval == 48:
		AdjMat = loadtxt("Adj.txt", delimiter=",")
		W = AdjMat*trate	
	elif aval == 2:
		# Hidden Gem
		W=np.array([[-trate,trate],[trate,-trate]])	
	return W	

def anglesLoader(aval):
	angles = collections.namedtuple('Angles',['alpha','beta','gama'])
	if aval == 6:
		a, b, g = np.array([0,0,pi/2,3*pi/2,pi,0]), np.array([0,0.5*pi,0.5*pi,0.5*pi,pi*0.5,pi]), np.array([0,0,0.5*pi,1.5*pi,pi,0])
	elif aval == 48:
		angles_load_from_file = loadtxt("Euler.txt", delimiter=",")
		a,b,g = angles_load_from_file[:,0], angles_load_from_file[:,1], angles_load_from_file[:,2]	
	angles_out = angles(a,b,g)	
	return 	angles_out
##############################################################################################################		


##############################################################################################################		
# Methods for parts
##############################################################################################################
def H20zem(g20,g22,alpha,beta,gama):
	H20zeeman = np.zeros(len(alpha), dtype=complex)
	for i in range(len(alpha)):
		H20zeeman[i] = (0.5 * g20 * (3 * np.cos(beta[i])**2-1)+g22*np.sqrt(1.5)*np.cos(2*gama[i])*np.sin(beta[i])**2)
	return H20zeeman

def H21zem(g20,g22,alpha,beta,gama):
	i1 = complex(0,1)
	H21zeeman = np.empty(len(alpha), dtype = complex)
	for i in range(len(alpha)):
		H21zeeman[i] = (-1)*np.exp(i1*alpha[i])*(g20*np.exp(-i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1-np.cos(beta[i])))+\
			np.exp(i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1+np.cos(beta[i])))*g22)
	return H21zeeman

def A20hyp(a20,a22,alpha,beta,gama):
	A20hyper = np.zeros(len(alpha), dtype = complex)
	for i in range(0,len(alpha)):
		A20hyper[i] = (0.5*a20*(3*np.cos(beta[i])**2-1)+a22*sqrt(1.5)*np.cos(2*gama[i])*np.sin(beta[i])**2)
	return A20hyper	

def A2p2hyp(a20,a22,alpha,beta,gama):
	i1 = complex(0,1)
	A2p2hyper = np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A2p2hyper[i] = np.exp(i1*2*alpha[i])*((0.5*(1+np.cos(beta[i])**2)*np.cos(2*gama[i])+i1*np.cos(beta[i])*np.sin(2*gama[i]))*a22+sqrt(3/8)*np.sin(beta[i])**2*a20)
	return A2p2hyper	

def A21hyp(a20,a22,alpha,beta,gama):
	i1=complex(0,1)
	A21hyper=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A21hyper[i]=(-1)*np.exp(i1*alpha[i])*(a20*np.exp(-i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1-np.cos(beta[i])))+\
			np.exp(i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1+np.cos(beta[i])))*a22)
	return A21hyper
##############################################################################################################		


##############################################################################################################		
# Matrix initialization is here
##############################################################################################################
def MatSpinHalfInit(W,p,H00zeeman,H20zeeman,H2p1zeeman,H2m1zeeman,A00hyper,A20hyper,A2p2hyper,A2m2hyper, A2p1hyper, A2m1hyper):
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
	"""We reshape Array Here"""
	output_matrix = P + WW - complex(0,1) * (A-B)
	output_matrix_to_2d_reshape = np.reshape(output_matrix,(max,max),order='F')
	output_matrix_inverse = np.linalg.inv(output_matrix_to_2d_reshape)
	output_matrix_to_nd_reshape = np.reshape(output_matrix_inverse,(a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),order='F')
	return output_matrix_to_nd_reshape

def MatSpinOneInit(W,p, H00zeeman,H20zeeman,H2p1zeeman,H2m1zeeman,A00hyper,A20hyper,A2p2hyper,A2m2hyper, A2p1hyper, A2m1hyper):
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
	
	output_matrix = P + WW - complex(0,1) * (A-B)
	output_matrix_to_2d_reshape = np.reshape(output_matrix,(max,max),order='F')
	output_matrix_inverse = np.linalg.inv(output_matrix_to_2d_reshape)
	output_matrix_to_nd_reshape = np.reshape(output_matrix_inverse,(a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),order='F')
	return output_matrix_to_nd_reshape
##############################################################################################################		


def SumMatrix(input_matrix):
	"""
	This method summs up the Matrix
	"""
	matrix_sum=0
	if m0val == 2:
		for k,l,ks,ls,m,n,ms,ns,i,j in itertools.product(xrange(0,m0,1),xrange(0,m1,1),xrange(0,m0s,1),\
			xrange(0,m1s,1),xrange(0,m00,1),xrange(0,m11,1),xrange(0,m00s,1),xrange(0,m11s,1),xrange(0,a,1),xrange(0,b,1)):
			# if m!=n and k!=l:

				matrix_sum += Hm[ls,ks]*Hmm[l,k]*input_matrix[i,k,l,ks,ls,j,m,n,ms,ns]*Hminv[ms,ns]*Hmminv[m,n]*(1/(np.sqrt(2)*aval)) #matrix spin-1/2
				# matrix_sum+=Hm[ls,ks]*Hmm[l,k]*input_matrix[i,k,l,ks,ls,j,m,n,ms,ns]*Hminv[ms,ns]*Hmminv[m,n] #matrix spin-1/2
				# matrix_sum+=Hm[l,k]*Hmm[ls,ks]*input_matrix[i,k,l,ks,ls,j,m,n,ms,ns]*Hminv[m,ns]*Hmminv[m,n]*(1/(np.sqrt(2)*aval)) #matrix spin-1/2
				# su+=Hmm[ls,ks]*input_matrix[i,k,l,ks,ls,j,m,n,ms,ns]*Hmminv[ms,ns]*(1/(np.sqrt(2)*aval)) #matrix spin-1/2 isolated spin
	elif m0val == 3:
		for k,l,ks,ls,m,n,ms,ns,i,j in itertools.product(xrange(0,m0,1),xrange(0,m1,1),xrange(0,m0s,1),\
			xrange(0,m1s,1),xrange(0,m00,1),xrange(0,m11,1),xrange(0,m00s,1),xrange(0,m11s,1),xrange(0,a,1),xrange(0,b,1)):
			# if m!=n and k!=l and ms!=ns and ks!=ls:	
				matrix_sum += Hmm[ls,ks]*Hmone[l,k]*input_matrix[i,k,l,ks,ls,j,m,n,ms,ns]*Hmminv[ms,ns]*Hminvone[m,n]*(1/(np.sqrt(3)*aval))	#matrix spin-1
				# matrix_sum+=Hmm[l,k]*Hmone[ls,ks]*input_matrix[i,k,l,ks,ls,j,m,n,ms,ns]*Hmminv[m,n]*Hminvone[ms,ns]*(1/(np.sqrt(3)*aval))	#matrix spin-1
	return matrix_sum

Main()
time_elapsed = (time.clock() - time_start)
print (time_elapsed)
  
