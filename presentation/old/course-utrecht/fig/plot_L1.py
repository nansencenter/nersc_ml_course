import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer

from matplotlib import rc, rcParams
rc('text', usetex=True)
rcParams.update({'font.size':16})
seed = 10
np.random.seed(seed)
N = 80
dpi = 200

#Default colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

#Genere data
X = np.sort(1*np.random.rand(N,1)+1, axis=0)
y = (2*X).ravel()
y += 0.2*np.random.randn(N)
y[::5] += 2*(0.5-np.random.rand(y[::5].size))

#Train linear model
model = make_pipeline(PolynomialFeatures(1),linear_model.LinearRegression(fit_intercept=False))
model.fit(X,y)
y_pred = model.predict(X)

regr = model.steps[1][1]
#Plot Data
plt.scatter(X,y,s=20, edgecolor="black",
            c="lightgray", label="data")
plt.xlabel('data($x$)')
plt.ylabel('target($y$)')
plt.legend()
plt.savefig('data-lin.png', dpi=dpi,bbox_inches = 'tight', pad_inches = 0)


s = '-' if regr.coef_[0]<0 else '+'
plt.plot(X,y_pred,#color="cornflowerblue",
         label="$y={:3.3f}x {} {:3.3f}$".format(regr.coef_[1],s,np.abs(regr.coef_[0])), linewidth=2)
plt.legend()

plt.savefig('interp-lin.png', dpi=dpi,bbox_inches = 'tight', pad_inches = 0)
plt.show()

deg_param = model.steps[0][0] + '__' + 'degree'

N = 35
X = np.sort(1*np.random.rand(N,1)+1, axis=0)
y = ((3*X-4)**2).ravel()
y += 0.15*np.random.randn(N)
y[::5] += 1.5*(0.5-np.random.rand(y[::5].size))

Ldeg = [1,2,30]
y_pred = dict()
plt.scatter(X,y,s=20, edgecolor="black",
            c="lightgray", label="data")
for deg in Ldeg:
	model.set_params(**{deg_param:deg})
	model.fit(X,y)
	y_pred[deg] = model.predict(X)

#separate plots
for deg,col in zip(Ldeg,colors):
	plt.scatter(X, y, s=20, edgecolor="black", c="lightgray", label="data")
	plt.plot(X, y_pred[deg], label='degree={:d}'.format(deg), linewidth=2,color=col)
	plt.title('degree={:d}'.format(deg))
	plt.savefig('interp-pol-'+str(deg)+'.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
	plt.show()


#Train,val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
plt.scatter(X_train, y_train, s=20, edgecolor="black", c="lightgray", label="training")
plt.scatter(X_val, y_val, s=20, edgecolor="red", c="lightcoral", label="validation")
plt.legend()
plt.savefig('datasplit.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
#plt.show()

for deg in Ldeg:
	model.set_params(**{deg_param:deg})
	score = cross_val_score(model,X,y,cv=20,verbose=0,scoring=make_scorer(r2_score))
	model.fit(X_train,y_train)
	y_pred[deg] = model.predict(X)
	plt.plot(X, y_pred[deg],
		label='degree={:d}'.format(deg), linewidth=2)
	print('deg',deg,',score: val=',model.score(X_val,y_val),
		' , train=',model.score(X_train,y_train),
		' , cv=',np.mean(score))


plt.legend()
plt.savefig('modelchoice.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.show()