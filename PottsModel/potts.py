#!/usr/bin/python

import numpy as np

class potts:
	def __init__(self, L,q):	#Create Potts model
	
		self.L=L		#lattice length/width
		self.size=L**2	#model size
		self.q=q		#number of states
		self.h=np.zeros(self.q)		#bias
		self.J=1	#couplings
		self.randomize_state()
	
		self.set_temp()
		
	def set_temp(self,T=1):
		self.T=1.0
		self.beta=1.0/self.T
	def set_beta(self,B):
		self.beta=B
		if B==0:
			self.B=None
		else:
			self.T=1/B

	def randomize_state(self):
		self.s = np.random.randint(0,self.q,(self.L,self.L))

	def MetropolisStep(self):	    #Execute step of Metropolis algorithm
		i = np.random.randint(self.L)		#row
		j = np.random.randint(self.L)		#column
		snew = np.random.randint(self.q)	#target state
		
		eDiff = self.h[self.s[i,j]]-self.h[snew]
		eDiff+= self.J*(int(self.s[i,j]==self.s[(i+1)%self.L,j])-int(snew==self.s[(i+1)%self.L,j]))	#up
		eDiff+= self.J*(int(self.s[i,j]==self.s[(i-1)%self.L,j])-int(snew==self.s[(i-1)%self.L,j]))	#down
		eDiff+= self.J*(int(self.s[i,j]==self.s[i,(j+1)%self.L])-int(snew==self.s[i,(j+1)%self.L]))	#right
		eDiff+= self.J*(int(self.s[i,j]==self.s[i,(j-1)%self.L])-int(snew==self.s[i,(j-1)%self.L]))	#left
		if eDiff <= 0 or np.random.rand() < np.exp(-self.beta*eDiff):    # Metropolis!
			self.s[i,j] = snew
			
	def Energy(self):
		E=0
		for i in range(self.L):
			for j in range(self.L):
				E-= self.h[self.s[i,j]]
				E-= self.J*int(self.s[i,j]==self.s[(i+1)%self.L,j])	#up
				E-= self.J*int(self.s[i,j]==self.s[(i-1)%self.L,j])	#down
				E-= self.J*int(self.s[i,j]==self.s[i,(j+1)%self.L])	#right
				E-= self.J*int(self.s[i,j]==self.s[i,(j-1)%self.L])	#left
		return(E)
