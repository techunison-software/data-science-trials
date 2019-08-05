# Normal Distribution
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)
samples=np.random.lognormal(mean=1.,sigma=.4,size=10000)
shape,loc,scale=scipy.stats.lognorm.fit(samples,floc=0)
num_bins=50
clr="#EFEFEF"
counts,edges,patches=plt.hist(samples,bins=num_bins,color=clr)
centers=0.5*(edges[:-1]+edges[1:])
cdf=scipy.stats.lognorm.cdf(edges,shape,loc=loc,scale=scale)
prob=np.diff(cdf)
plt.plot(centers,samples.size*prob,'k-',linewidth=2)
plt.show()


# Binomial Distribution
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import binom
data=binom.rvs(n=17,p=0.7,loc=0,size=1010)
ax=seaborn.distplot(data,
                kde=True,
                color='pink',
                hist_kws={"linewidth": 22,'alpha':0.77})
ax.set(xlabel='Binomial',ylabel='Frequency')
plt.show()


# Poisson Distribution
import numpy as np
import matplotlib.pyplot as plt
s=np.random.poisson(5, 10000)
plt.hist(s,16,normed=True,color='Green')
plt.show()


# Bernoulli Distribution
import numpy as np
import matplotlib.pyplot as plt
s=np.random.binomial(10,0.5,1000)
plt.hist(s,16,normed=True,color='Brown')
plt.show()


# Exponential Distribution

import numpy as N
import matplotlib.pyplot as P

n = 1000    
scale_radius = 2
central_surface_density = 100 #I would like this to be the controlling variable, even if it's specification had knock on effects on n.
radius_array = N.random.exponential(scale_radius,(n,1))     
P.figure()    
nbins = 100
number_density, radii = N.histogram(radius_array, bins=nbins,normed=False)
P.plot(radii[0:-1], number_density)
P.xlabel('$R$')
P.ylabel(r'$\Sigma$')
P.ylim(0, central_surface_density)
P.legend()      
P.show()
