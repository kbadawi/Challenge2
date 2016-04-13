import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

#Je charge mes valeurs X/Y
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_Y = diabetes.target

# Je crée et entraine ma ligne de regression
regr = linear_model.LinearRegression()
regr.fit(diabetes_X, diabetes_Y)

# J'affiche mon résultat
plt.scatter(diabetes_X, diabetes_Y,  color='black')
plt.plot(diabetes_X, regr.predict(diabetes_X), color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
