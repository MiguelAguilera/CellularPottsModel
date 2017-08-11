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
			i,j = np.random.randint(1,self.L-1,2)	# row,column of cell, borders are forbidden
			self.s[i,j]=ind
		self.VE = self.volume_energy(self.s)


	def volume_energy(self,s):
		unique, counts = np.unique(s, return_counts=True)
		V=counts[1:]
		return self.th/(2*self.V0)*np.sum((V-self.V0)**2)
		
		#old code computing areas of connected blobs
#		cells= measure.label(s,connectivity=1)
#		props=measure.regionprops(cells)
#		print('areas',[prop.filled_area for prop in props])
#		return np.sum([(prop.filled_area-self.V0)**2 for prop in props])


	def get_neighbors(self,i,j):	# Get the array of Von Newmann neighbors of a cell
		nn = []
		nn+=[self.s[(i+1)%self.L,j]]	#up
		nn+=[self.s[(i-1)%self.L,j]]	#down
		nn+=[self.s[i,(j-1)%self.L]]	#left
		nn+=[self.s[i,(j+1)%self.L]]	#right

		return np.array(nn)
			
			
	def get_moore_neighbors(self,i,j):	# Get the array of Moore neighbors of a cell
		nn = []
		nn+=[self.s[(i+1)%self.L,(j-1)%self.L]]	#up-left
		nn+=[self.s[(i+1)%self.L,j]]			#up
		nn+=[self.s[(i+1)%self.L,(j+1)%self.L]]	#up-right
		nn+=[self.s[i,(j+1)%self.L]]			#right
		nn+=[self.s[(i-1)%self.L,(j+1)%self.L]]	#down-right
		nn+=[self.s[(i-1)%self.L,j]]			#down
		nn+=[self.s[(i-1)%self.L,(j-1)%self.L]]	#down-left
		nn+=[self.s[i,(j-1)%self.L]]			#left
		return np.array(nn)	
		
		
	def is_locally_connected(self,cell_domain): # Detect local connectivity
		# cell_domain should be 1 if cell is present and 0 otherwise
		transitions=0
		is_connected=False
		if np.sum(cell_domain)>0:
			for i in range(8):
				if cell_domain[i]<cell_domain[(i+1)%8]:	# +1 if transition from 0 to 1
					transitions+=1
			if transitions<=1:
				is_connected=True
		return is_connected



	def MetropolisStep(self,mode='CA'):	    # Execute step of Metropolis algorithm
		
		#Select candidate and target nodes	
		if mode=='MMA':
			i,j = np.random.randint(1,self.L-1,2)	# row,column of cell, borders are forbidden
			nn=self.get_neighbors(i,j)				# array of cell neighbors 
			sijnew = nn[np.random.randint(len(nn))]		# target state
			cond = sijnew!=self.s[i,j]
			
		if mode=='CA':
			i,j = np.random.randint(1,self.L-1,2)	# row,column of cell
			nn=self.get_neighbors(i,j)				# array of cell neighbors 
			nn_unique=np.unique(nn)
			sijnew = nn_unique[np.random.randint(len(nn_unique))]		# target state
			domain=self.get_moore_neighbors(i,j)
			lc_candidate=self.is_locally_connected(domain==self.s[i,j])
			lc_target=self.is_locally_connected(domain==sijnew)
			cond = lc_candidate and lc_target and sijnew!=self.s[i,j]

			
		#Evaluate acceptance of change
		if cond:
			eDiff = 0
			
			coupling_neighbors=self.get_neighbors(i,j)
			#Compute adhesion energy difference
			for sn in coupling_neighbors:
				eDiff+= self.J[self.type[sijnew],self.type[sn]]*int(sijnew!=sn) - self.J[self.type[self.s[i,j]],self.type[sn]]*int(self.s[i,j]!=sn) 
			#Compute volume energy difference
			snew=self.s.copy()
			snew[i,j] = sijnew
			VEnew = self.volume_energy(snew)
			eDiff += VEnew - self.VE
			if eDiff <= 0 or np.log(np.random.rand()) < -self.beta*eDiff:    # Metropolis
				self.s[i,j] = sijnew
				self.VE = VEnew
				

	def Energy(self):
		E=self.volume_energy(self.s)
		for i in range(self.L):
			for j in range(self.L):
				if self.s[i,j]>0:
					coupling_neighbors=self.get_neighbors(i,j)
					#Compute adhesion energy difference
					for sn in coupling_neighbors:
						if sn<self.s[i,j]:	#we compute each link just once
							E+= self.J[self.type[self.s[i,j]],self.type[sn]]*int(self.s[i,j]!=sn)
		return(E)
