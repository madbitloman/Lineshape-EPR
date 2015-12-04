from __future__ import division
import numpy as np 
from fractions import Fraction 
import itertools
from numpy.random import rand
from pylab import *
import time
import sys
time_start=time.clock()


trate=0.01
# Hinput=input('Enter value of Fixed Magnetic Field:')
# a,b = map(int,sys.stdin.readline().split())
H=40
kom = trate
# W=np.array([[-trate,trate],[trate,-trate]])
# W=np.ones((48,48))
# angles=loadtxt("Euler.txt", delimiter=",")
# AdjMat=loadtxt("Adj.txt", delimiter=",")
# alpha,beta,gama=angles[:,0], angles[:,1],angles[:,2]
# W=AdjMat*trate

W=np.array([[-4*kom,kom,kom,kom,kom,0.0],
		   [kom,-4*kom,kom,kom,0.0,kom], 
		   [kom,kom,-4*kom,0.0,kom,kom],  
		   [kom,kom,0.0,-4*kom,kom,kom],   
		   [kom,0.0,kom,kom,-4*kom,kom],
		   [0.0,kom,kom,kom,kom,-4*kom]])
# gxx,gyy,gzz=raw_input("Now it is time for g-tensor! Enter TAB separated values of gx,gy and gz: ").split()
# print 'Warning! It is Calculating!'
# gxx=float(gxx)
# gyy=float(gyy)
# gzz=float(gzz)	
gxx=2.0089
gyy=2.0061
gzz=2.00032
kk=0.566/0.4          
Axx=5.5*kk
Ayy=4.5*kk
Azz=34*kk

ge=2.002 
a00=-ge/sqrt(3)*(Axx+Ayy+Azz)
a22=0.5*ge*(Axx-Ayy)
a20=sqrt(0.666)*ge*(Azz-0.5*(Axx+Ayy));

g00=-(gxx+gyy+gzz)/np.sqrt(3)
g22=0.5*(gxx-gyy)
g20=(gzz-0.5*(gyy+gxx))*sqrt(0.666)
alpha=np.array([0,0,pi/2,3*pi/2,pi,0])
beta=np.array([0,0.5*pi,0.5*pi,0.5*pi,pi*0.5,pi])
gama=np.array([0,0,0.5*pi,1.5*pi,pi,0])



def H20zem():
	H20zeeman=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		H20zeeman[i]=H*sqrt(0.666)*(0.5*g20*(3*np.cos(beta[i])**2-1)+g22*sqrt(1.5)*np.cos(2*gama[i])*np.sin(beta[i])**2)
	return H20zeeman	

def H21zem():
	i1=complex(0,1)
	H21zeeman=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		H21zeeman[i]=-H*0.5*(-1)*np.exp(i1*alpha[i])*(np.exp(-i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1-np.cos(beta[i])))+\
			np.exp(i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1+np.cos(beta[i]))))*g22
	return H21zeeman

def A20hyp():
	A20hyper=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A20hyper[i]=sqrt(0.666)*(0.5*a20*(3*np.cos(beta[i])**2-1)+a22*sqrt(1.5)*np.cos(2*gama[i])*np.sin(beta[i])**2)
	return A20hyper	

def A2p2hyp():
	i1=complex(0,1)
	A2p2hyper=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A2p2hyper[i]=0.5*np.exp(i1*2*alpha[i])*((0.5*(1+np.cos(beta[i])**2)*np.cos(2*gama[i])+i1*np.cos(beta[i])*np.sin(2*gama[i]))*a22+Fraction(3,8)**0.5*np.sin(beta[i])**2*a20)
	return A2p2hyper	

def A21hyp():
	i1=complex(0,1)
	A21hyper=np.zeros(len(alpha), dtype=complex)
	for i in range(0,len(alpha)):
		A21hyper[i]=(-1)*np.exp(i1*alpha[i])*(np.exp(-i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1-np.cos(beta[i])))+\
			np.exp(i1*2*gama[i])*(-0.5*np.sin(beta[i])*(1+np.cos(beta[i]))))*a22
	return A21hyper


H00zeeman=-g00*H/sqrt(3)
A00hyper=-a00/sqrt(3)
H20zeeman=H20zem()
A20hyper=A20hyp()
A2p1hyper=A21hyp()
H2p1zeeman=H21zem()
A2m1hyper=-np.conjugate(A2p1hyper)
H2m1zeeman=-np.conjugate(H2p1zeeman)
# A2p2hyper=A20hyp()
A2p2hyper=A2p2hyp()
A2m2hyper=0.5*np.conjugate(A2p2hyper)


a=6
b=6
m0=3
m1=3
m00=3
m11=3
m0s=2
m1s=2
m00s=2
m11s=2
max=a*m0*m1*m0s*m1s
Hms=np.array([[1,0,0],[0,1,0],[0,0,1]])
Hinvs=np.linalg.inv(Hms)

Hmm=np.array([[0,0.5],[0.5,0]])
Hinv=np.linalg.inv(Hmm)

def MatAs(p):
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
		# ns=m11s
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
			B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5-(A00hyper+A20hyper[i])*0.5
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


			""""""
		# if i==j and l==n and ls==ns and k==0 and m==0 and ks==0 and ms==0:
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*0.5
		# if i==j and l==n and ls==ns and k==1 and m==1 and ks==0 and ms==0: 		
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5
		# if i==j and l==n and ls==ns and k==2 and m==2 and ks==0 and ms==0:  
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*(-0.5)
		# if i==j and l==n and ls==ns and k==0 and m==0 and ks==1 and ms==1:  
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(-0.5)
		# if i==j and l==n and ls==ns and k==1 and m==1 and ks==1 and ms==1: 		
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)
		# if i==j and l==n and ls==ns and k==2 and m==2 and ks==1 and ms==1: 
		# 	A[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(0.5)

		# 	"""Other Operators"""
		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==0 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= 0.5*sqrt(2)*A2p2hyper[i]	#S+I+

		# if i==j and l==n and ls==ns and k==1 and m==2 and ks==0 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2p2hyper[i]	#S+I+

		# if i==j and l==n and ls==ns and k==0 and m==0 and ks==0 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*H21zeeman[i]-0.5*A21hyper[i]	#S+Iz	
		# if i==j and l==n and ls==ns and k==1 and m==1 and ks==0 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*H21zeeman[i]#S+Iz	
		# if i==j and l==n and ls==ns and k==2 and m==2 and ks==0 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*H21zeeman[i]+0.5*A21hyper[i]#S+Iz	

		# if i==j and l==n and ls==ns and k==0 and m==0 and ks==1 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*H211zeeman[i]+0.5*A211hyper[i]	#S-Iz	
		# if i==j and l==n and ls==ns and k==1 and m==1 and ks==1 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*H211zeeman[i]#S-Iz	
		# if i==j and l==n and ls==ns and k==2 and m==2 and ks==1 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*H211zeeman[i]-0.5*A211hyper[i]#S-Iz

		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==0 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A21hyper[i]	#SzI+	
		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==1 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A21hyper[i]
		# if i==j and l==n and ls==ns and k==1 and m==2 and ks==0 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A21hyper[i]	#SzI+	
		# if i==j and l==n and ls==ns and k==1 and m==2 and ks==1 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A21hyper[i]	#SzI+		

		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==0 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A211hyper[i]	#SzI-	
		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==1 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A211hyper[i]
		# if i==j and l==n and ls==ns and k==2 and m==1 and ks==0 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A211hyper[i]	#SzI-	
		# if i==j and l==n and ls==ns and k==2 and m==1 and ks==1 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A211hyper[i]	#SzI-			
				

		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==1 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*np.conjugate(A2p2hyper[i]) #S-I-
		# if i==j and l==n and ls==ns and k==2 and m==1 and ks==1 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*np.conjugate(A2p2hyper[i]) #S-I-	

		# if i==j and l==n and ls==ns and k==1 and m==2 and ks==1 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i])#S+I-
		# if i==j and l==n and ls==ns and k==0 and m==1 and ks==1 and ms==0:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) #S+I-

		# if i==j and l==n and ls==ns and k==2 and m==1 and ks==0 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i])#S-I+
		# if i==j and l==n and ls==ns and k==1 and m==0 and ks==0 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) #S-I+					
			
		# k=m0
		# l=m1
		# ks=m0s
		# ls=m1s
		# m=m00
		# n=m11
		# ms=m00s
		# ns=m11s

			""" Second Delta """
		# if i==j and k==m and ks==ms and l==0 and n==0 and ls==0 and ns==0:
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5+(A00hyper+A20hyper[i])*0.5
		# if i==j and k==m and ks==ms and l==1 and n==1 and ls==0 and ns==0: 		
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5
		# if i==j and k==m and ks==ms and l==2 and n==2 and ls==0 and ns==0:  
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*0.5-(A00hyper+A20hyper[i])*0.5
		# if i==j and k==m and ks==ms and l==0 and n==0 and ls==1 and ns==1:  
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(-0.5)
		# if i==j and k==m and ks==ms and l==1 and n==1 and ls==1 and ns==1: 		
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)
		# if i==j and k==m and ks==ms and l==2 and n==2 and ls==1 and ns==1: 
		# 	B[i,k,l,ks,ls,j,m,n,ms,ns]=(H00zeeman+H20zeeman[i])*(-0.5)+(A00hyper+A20hyper[i])*(0.5)

		# # """Other operators"""
		# if i==j and k==m and ks==ms and ls==1 and ns==0 and l==1 and n==0:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2p2hyper[i]	#S+I+
		# if i==j and k==m and ks==ms and ls==1 and ns==0 and l==2 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*A2p2hyper[i]	#S+I+

		# if i==j and k==m and ks==ms and ls==0 and ns==1 and l==0 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*np.conjugate(A2p2hyper[i]) #S-I-	
		# if i==j and k==m and ks==ms and ls==0 and ns==1 and l==1 and n==2:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*sqrt(2)*np.conjugate(A2p2hyper[i]) #S-I-	

		# if i==j and k==m and ks==ms and ls==1 and ns==0 and l==0 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) #S+I-	
		# if i==j and k==m and ks==ms and ls==1 and ns==0 and l==1 and n==2:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]= sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i])#S+I-

		# if i==j and k==m and ks==ms and ls==0 and ns==1 and l==1 and n==0:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) #S-I+	
		# if i==j and k==m and ks==ms and ls==0 and ns==1 and l==2 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=sqrt(2)*(0.5*A00hyper-0.25*A20hyper[i]) #S-I+	


		# # if i==j and k==m and ks==ms and ls==1 and ns==0 and l==0 and n==0:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*H21zeeman[i]-0.5*A21hyper[i]	#S+Iz	
		# if i==j and k==m and ks==ms and ls==1 and ns==0 and l==1 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*H21zeeman[i]#S+Iz	
		# if i==j and k==m and ks==ms and ls==1 and ns==0 and l==2 and n==2:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*H21zeeman[i]+0.5*A21hyper[i]#S+Iz	

		# if i==j and k==m and ks==ms and ls==0 and ns==1 and l==0 and n==0:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*H211zeeman[i]+0.5*A211hyper[i]	#S-Iz	
		# if i==j and k==m and ks==ms and ls==0 and ns==1 and l==1 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]=0.5*H211zeeman[i]#S-Iz	
		# if i==j and k==m and ks==ms and ls==0 and ns==1 and l==2 and n==2:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]=-0.5*H211zeeman[i]-0.5*A211hyper[i]#S-Iz

		# if i==j and k==m and ks==ms and ls==0 and ns==0 and l==1 and n==0:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A211hyper[i]	#SzI+	
		# if i==j and k==m and ks==ms and ls==1 and ns==1 and l==1 and n==0:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A211hyper[i]		
		# if i==j and k==m and ks==ms and ls==0 and ns==0 and l==2 and n==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A211hyper[i]	#SzI+	
		# if i==j and l==n and ls==ns and k==1 and m==1 and ks==2 and ms==1:
		# 	D[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A211hyper[i]	#SzI+	

		# if i==j and k==m and ks==ms and ls==0 and ns==0 and l==0 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A211hyper[i]	#SzI+	
		# if i==j and k==m and ks==ms and ls==1 and ns==1 and l==0 and n==1:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A211hyper[i]		
		# if i==j and k==m and ks==ms and ls==0 and ns==0 and l==1 and n==2:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]= 0.25*A211hyper[i]	#SzI+	
		# if i==j and l==n and ls==ns and k==1 and m==1 and ks==1 and ms==2:
		# 	DD[i,k,l,ks,ls,j,m,n,ms,ns]= -0.25*A211hyper[i]	#SzI+					
		
		
		if k==m and l==n and ks==ms and ls==ns:
			WW[i,k,l,ks,ls,j,m,n,ms,ns]=-W[i,j]	
	
	C=-complex(0,1)*(A+D-B-DD)+P+WW


	AA=np.reshape(C,(max,max),order='F')
	AAinv=np.linalg.inv(AA)
	AA2ndresh=np.reshape(AAinv,(a,m0,m1,m0s,m1s,b,m00,m11,m00s,m11s),order='F')
	return AA2ndresh

def SumMa(A):
	"""
	This method summs up the Matrix
	"""
	su=0
	for i,k,l,ks,ls,j,m,n,ms,ns in itertools.product(xrange(0,a,1), xrange(0,m0,1),xrange(0,m1,1),xrange(0,m0s,1),\
		xrange(0,m1s,1),xrange(0,b,1),xrange(0,m00,1),xrange(0,m11,1),xrange(0,m00s,1),xrange(0,m11s,1)):
		if ms!=ns and ks!=ls:
			su+=Hmm[ls,ks]*Hms[l,k]*A[i,k,l,ks,ls,j,m,n,ms,ns]*Hinv[ms,ns]*Hinvs[m,n]*0.5*(1/6)
	return su

def Main():
	# ss=np.arange(79700,80600,0.1)
	ss=np.arange(0,150,0.1)
	ff=np.zeros(len(ss))
	for ol in range(0,len(ss)):
		p=(ss[ol]-0.01*complex(0,1))*complex(0,1)
		A=MatAs(p)
		# z=SumMa(A)
		ff[ol]=np.real(SumMa(A))

	figure()
	# xkcd()
	plot(ss,ff, 'blue')
	# fill_between(ss,ff,0.0,facecolor='blue',alpha=0.5)
	# axis(ymin=0)
	# plt.xlabel('Frequency')
	# plt.ylabel('Intensity')
	show()
	
Main()	

time_elapsed = (time.clock() - time_start)
print 'Success!'
print 'Calculation time',time_elapsed