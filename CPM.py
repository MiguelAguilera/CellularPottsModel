#!/usr/bin/python

import numpy as np
from skimage import measure

class CPM:
	def __init__(self, L,cells=[1],V0=None,th=1):	#Create Potts model
	
		self.L=L		#lattice length/width
		self.size=L**2	#model size
		self.cells=cells
		self.c=1+np.sum(self.cells)
		self.q=1+len(self.cells)		#number of states
		self.J=np.zeros((self.q,self.q))	#couplings
		self.th = th	# volume constraint
		self.type = [0]
		i=0
		for c in self.cells:
			i+=1
			for rep in range(c):
				self.type+=[i]
				
		if V0 is None:
			self.V0=int(np.round(L*L/(self.c-1)*0.5))
		else:
			self.V0=V0
			
		self.randomize_couplings()
		self.initialize_state()
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
	
	
	def randomize_couplings(self):
		for i in range(self.q):
			for j in range(i,self.q):
				self.J[i,j]=np.random.rand()
				if not i==j:
					self.J[j,i]=self.J[i,j]
		self.J[0,0]=0
		
			
	def initialize_state(self):
#		self.s = np.random.randint(0,self.c,(self.L,self.L))
		self.s=np.zeros((self.L,self.L),int)	#add starting cells in random positions
		for ind in range(1,self.c):
			i,j = np.random.randint(0,self.L,2)		#row,column
			self.s[i-1:i+2,j]=ind
			self.s[i,j-1:j+2]=ind
		self.VE = self.volume_energy(self.s)


	def volume_energy(self,s):
		unique, counts = np.unique(s, return_counts=True)
		V=counts[1:]
		return np.sum((V-self.V0)**2)
		
#		cells= measure.label(s,connectivity=1)
#		props=measure.regionprops(cells)
#		print('areas',[prop.filled_area for prop in props])
#		return np.sum([(prop.filled_area-self.V0)**2 for prop in props])	#areas of connected blobs


	def get_neighbors(self,i,j):	# Get the array of neighbors of a cell
		nn = []
		nn+=[self.s[(i+1)%self.L,j]]	#up
		nn+=[self.s[(i-1)%self.L,j]]	#down
		nn+=[self.s[i,(j-1)%self.L]]	#left
		nn+=[self.s[i,(j+1)%self.L]]	#right

		return np.array(nn)
			
			
	def ModifiedMetropolisStep(self):	    # Execute step of Metropolis algorithm
		
		#Select target node
		i,j = np.random.randint(0,self.L,2)		# row,column of cell
		nn=self.get_neighbors(i,j)				# array of cell neighbors 
		while np.sum(np.abs(self.s[i,j]-nn))==0:
			i,j = np.random.randint(0,self.L,2)	
			nn=self.get_neighbors(i,j)
		notequal=np.where(nn!=self.s[i,j])[0]	# neighbors different from cell
		sijnew = nn[notequal[np.random.randint(len(notequal))]]		# target state

		eDiff = 0
		
		#Compute adhesion energy difference
		for sn in nn:
			eDiff+= self.J[self.type[sijnew],self.type[sn]]*int(sijnew!=sn) - self.J[self.type[self.s[i,j]],self.type[sn]]*int(self.s[i,j]!=sn) 
		
		#Compute volume energy difference
		snew=self.s.copy()
		snew[i,j] = sijnew
		VEnew = self.volume_energy(snew)
		eDiff += VEnew - self.VE
		
		if eDiff <= 0 or np.random.rand() < np.exp(-self.beta*eDiff):    # Metropolis
			self.s[i,j] = sijnew
			self.VE = VEnew
				

	def Energy(self):
		E=self.VE
		for i in range(self.L):
			for j in range(self.L):
				E-= self.h[self.s[i,j]]
				E-= self.J*int(self.s[i,j]==self.s[(i+1)%self.L,j])	#up
				E-= self.J*int(self.s[i,j]==self.s[(i-1)%self.L,j])	#down
				E-= self.J*int(self.s[i,j]==self.s[i,(j+1)%self.L])	#right
				E-= self.J*int(self.s[i,j]==self.s[i,(j-1)%self.L])	#left
		return(E)
