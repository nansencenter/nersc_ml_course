import matplotlib.pyplot as plt
import numpy as np

dpi = 200

#Plot activation functions
x = np.arange(-4,4,1e-4)
lw = 4.0

Ly = [x,np.tanh(x),np.maximum(0,x),np.where(x > 0, x, x * 0.01)]
Lname = ['linear','tanh','relu','l-relu']

for y,name in zip(Ly,Lname):
	plt.plot(x,y,'r',linewidth=lw)
	plt.plot(x,np.zeros_like(x),':',color='black')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('activ-'+name+'.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
	plt.show()

