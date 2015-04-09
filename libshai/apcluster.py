eps = 1e-15

class APCluster:
  def start(self,S):
    assert S.shape==(S.shape[0],S.shape[0]), "distance matrix is square"
    # Initialize messages
    self.A = zeros_like(S)
    self.R = zeros_like(S)
    # Remove degeneracies
    self.S = S + eps*100*rand(*S.shape)
    # Set damping factor
    self.lam = 0.5
  
  def step(self):
    N = self.S.shape[0]
    S = self.S
    A = self.A
    # Compute responsibilities
    AS=A+S 
    I=argmax(AS,axis=1)
    Y=AS[I,0]
    for k in xrange(len(I)):
      AS[k,I[k]]=-inf
    I2=argmax(AS,axis=1)
    Y2=AS[I2,0]
    R=S-Y[newaxis,:]
    for k in xrange(len(I)):
      R[k,I[k]]=S[k,I[k]]-Y2[k]
    # Dampen responsibilities
    R=(1-self.lam)*R+self.lam*self.R

    # Compute availabilities
    Aold=A
    Rp=R.clip(min=0,max=inf) 
    for k in xrange(Rp.shape[0]):
      Rp[k,k]=R[k,k]
    A=sum(Rp,axis=0)[newaxis,:]-Rp
    dA=diag(A)
    A=A.clip(min=-inf,max=0)
    for k in xrange(N):
      A[k,k]=dA[k]
    # Dampen availabilities
    A=(1-self.lam)*A+self.lam*Aold
    
    self.A = A
    self.R = R
    self.S = S
    
  def stop(self):
    # Pseudomarginals
    E=self.R+self.A 
    # Indices of exemplars
    I=find(diag(E)>0) 
    K=len(I)
    # Assignments
    c=argmax(self.S[:,I],axis=1) 
    c[I]=arange(K) 
    idx=I[c]
    return I, idx

  def __call__(self, S, count=100):
    """Cluster based on a (dense) affinity matrix"""
    self.start(S)
    for k in xrange(count):
      self.step()
    return self.stop()

