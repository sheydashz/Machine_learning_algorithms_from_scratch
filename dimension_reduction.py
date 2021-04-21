from tools import pcaData,getFourSeven
import time
import matplotlib.pyplot as plt
from sklearn import manifold ## for tsne distribution
import numpy as np
from scipy.linalg import eigh
######### PCA CLASS ##############
class PCA():
    def __init__(self,*numbers):

        #self.digit=N*xy
        self.digit,self.delt=numbers
        self.N4=len(self.digit[0])
        self.N7=len(self.digit[1])
        self.N8=len(self.digit[2])
        self.tot_num=self.N4+self.N7+self.N8
        self.features=784
        self.N=len(self.digit)

        self.avg=self.get_average()
        self.data=self.digit-self.avg
        self.cov=self.covariance()
        self.vals,self.vecs=self.eignVal()

    #calculate avg: 784*1
    def get_average(self):
        avg = np.mean(self.digit, axis=0)
        return avg

    #calculate covariance for normalized data set
    def covariance(self):
        cov = np.dot(np.transpose(self.data), self.data)
        return np.divide(cov,self.N-1)


    #calculate eigenvalues and vectors of covariance matrix
    #and select larget self.dim
    #vercotrs: xy,dim
    def eignVal(self):
        cov=self.covariance()
        values, vectors = eigh(cov)
        return values,vectors.T

    def top_eignvect(self):
        return self.vecs[-self.dim:].T


    #calculate each data points' projection
    #vectors--> xy,dim
    def rebuild(self):
        vectors=self.top_eignvect()
        result = np.dot(self.data,vectors)
        return result,vectors

    #result:17958*2
    #vec:2*784
    def error(self,dim):
        self.dim=dim
        result,vec=self.rebuild()
        x_bar=np.dot(result,vec.T)
        x_bar=x_bar+self.avg
        err=np.divide(np.linalg.norm(self.digit-x_bar,ord=2),np.linalg.norm(self.digit,ord=2))
        if self.dim==2:
            return err,result
        else:
            return err

################### PCA ##########################
pca_data=pcaData()
pca_dim=[2,20,50,100,150,200]
time_start = time.time()
delt=0
# apply PCA algorithm to each digit
obj=PCA(pca_data,delt)
err=[0,0,0,0,0,0]
for i in range(len(pca_dim)):
    if pca_dim[i]==2:
        err[i],pca_new_data=obj.error(pca_dim[i])
    else:
        err[i] = obj.error(pca_dim[i])

#### PLOT ERROR RATE
x=['2','10','50','100','150','200']
plt.plot(x,err)
plt.title('pca error rate')
plt.xlabel('dimension')
plt.grid(True)
plt.ylabel('error')
plt.show(block=False)

### PLOT PCA
plt.figure(1)
plt.scatter(pca_new_data[:5842,0],pca_new_data[:5842,1],s=4,edgecolors=None,color='blue',label='pca 4')
plt.scatter(pca_new_data[5842:5842+6265,0],pca_new_data[5842:5842+6265,1],s=4,edgecolors=None,color='pink',marker='x',label='pca 7')
plt.scatter(pca_new_data[5842+6265:,0],pca_new_data[5842+6265:,1],s=4,marker='x',color='purple',label='pca 8')
plt.title('PCA',fontsize=30)
plt.legend(fontsize=20)
plt.axis('off')
plt.show(block=False)
plt.show()
############ LDA Class ########################
import numpy as np
from scipy.linalg import eigh

class LDA():
    def __init__(self,data,dim):
        self.data=data
        self.dim=dim
        self.N4=len(self.data[0])
        self.N7=len(self.data[1])
        self.N8=len(self.data[2])
        self.total_num=self.N4+self.N7+self.N8
        self.features=784
        self.m=self.mean_cal()
    #calculate the mean of each class
    # m: 3*xy
    def mean_cal(self):
        m=[0,0,0]
        for i in range(len(self.data)):
            m[i] =np.divide([sum(column) for column in zip(*self.data[i])],len(self.data[i]))
        return m

    def total_mean(self):
        tot_mean = np.zeros((1,784))
        for i in range(3):
            tot_mean= tot_mean+[sum(column) for column in zip(*self.data[i])]
        #returns a 1*xy vector
        return np.divide(tot_mean,self.total_num)
    def covariance(self,i):
        mtrx=[num-self.m[i] for num in self.data[i]]
        # cov = np.cov(np.transpose(mtrx))
        cov=(np.matmul(np.transpose(mtrx), mtrx))
        return cov
    #within class scatter matrix
    def within_s(self):
        s_w=np.zeros((784,784))
        for i in range(len(self.data)):
            s_w=s_w+self.covariance(i)
        return s_w
    def covariance2(self,i,avg):
        mtrx=self.m[i]-avg
        cov =len(self.data[i])*(np.matmul(mtrx.T,mtrx))
        return cov
    #between class scatter matrix
    def between_s(self):
        s_b=np.zeros((784,784))
        avg=self.total_mean()
        for i in range(len(self.data)):
           s_b=s_b+self.covariance2(i,avg)
        return s_b
    def eignVec(self):
        s_w=self.within_s()
        s_b=self.between_s()
        s_w_i=np.linalg.pinv(s_w)
        X=np.matmul(s_w_i,s_b)
        values, vectors = eigh(X)
        return vectors.T[-2:].T
    #vectors: xy,dim
    #self.data[i]: N,xy
    def rebuild(self):
        vectors=self.eignVec()
        tot_num = len(self.data[0]) + len(self.data[1]) + len(self.data[2])
        result=np.zeros((tot_num,self.dim))
        result[:self.N4] = np.matmul(vectors.T,self.data[0].T).T
        index=self.N4+self.N7
        result[self.N4:index]= np.matmul(vectors.T,self.data[1].T).T
        result[index:index+self.N8] = np.matmul(vectors.T, self.data[2].T).T
        return result
# ##################### LDA CALL ########################
four,seven,eight=getFourSeven()
data=[four['data'],seven['data'],eight['data']]
dim=2
obj = LDA(data,dim)
LDA_new_data=obj.rebuild()
## PLOT LDA
plt.figure(3)
plt.scatter(LDA_new_data[:5842,0],LDA_new_data[:5842,1],s=3,color='grey',label='lda 4')
plt.scatter(LDA_new_data[5842:5842+6265,0],LDA_new_data[5842:5842+6265,1],s=3,label='lda 7')
plt.scatter(LDA_new_data[5842+6265:,0],LDA_new_data[5842+6265:,1],s=3,color='cyan',label='lda 8')
plt.title('LDAA',fontsize=30)
plt.legend(fontsize=20)
plt.axis('off')
plt.show(block=False)
plt.show()

# ########################### tsne   ##################################333
tsne = manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_data)
## PLOT TSNE
plt.scatter(tsne_results[:5842,0],tsne_results[:5842,1],s=5,edgecolors=None,color='green',marker='*',label='tsne 4')
plt.scatter(tsne_results[5842:5842+6265,0],tsne_results[5842:5842+6265,1],s=5,edgecolors=None,color='purple',marker='*',label='tsne 7')
plt.scatter(tsne_results[5842+6265:,0],tsne_results[5842+6265:,1],s=5,color='pink',marker='*',label='tsne 8')

plt.title('TSNE',fontsize=30)
plt.legend(fontsize=20)
plt.axis('off')
plt.show()
