import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))

    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)


df = pd.read_csv("Data/track_features/tf_mini.csv")
df.columns = ['track_id', 'duration', 'release_year', 'us_popularity_estimate',
              'acousticness', 'beat_strength', 'bounciness', 'danceability',
              'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
              'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness',
              'tempo', 'time_signature', 'valence', 'acoustic_vector_0',
              'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
              'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6',
              'acoustic_vector_7']
# df= df.sample(frac=0.1)
df = df.drop(['track_id'], axis=1)
df["mode"], uniques = pd.factorize(df["mode"] )
df.info()

X = df.drop(['us_popularity_estimate'], axis=1).astype('float64')
y = df[['us_popularity_estimate']]
# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# # Create a pair grid instance
# grid = sns.PairGrid(data=df.sample(1000), vars=df.columns, height=4)
#
# # Map the plots to the locations
# grid = grid.map_upper(plt.scatter, color='darkred')
# grid = grid.map_upper(corr)
# grid = grid.map_lower(sns.kdeplot, cmap='Reds')
# grid = grid.map_diag(plt.hist, bins=10, edgecolor='k', color='darkred')
# plt.show()
#
# # visualize distribution of popularity
# fig, ax = plt.subplots()
# ax.hist(df['popularity'])
# ax.set_xlabel('popularity')
# ax.set_ylabel('Probability density')
# plt.show()

# the dataset is not balance, if we want to do classification, we must balance the data first


from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

# Ridge Regression
alphas = 10**np.linspace(10,-2,100)*0.5
# alphas=[0.5,1,5]
print(alphas)

ridge = Ridge(normalize=True)
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_.flatten())

np.shape(coefs)

figure_location="Thesis/figure_py/"
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.savefig(figure_location+"LASSO.pdf",format="pdf")
plt.show()

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)
print(ridgecv.alpha_)

ridgecv_best = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridgecv_best.fit(X_train, y_train)
mean_squared_error(y_test, ridgecv_best.predict(X_test))

# LASSO
alphas = 10**np.linspace(2,-8,100)

lasso = Lasso(max_iter=10000, normalize=True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas * 2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train.values.ravel())

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
mean_squared_error(y_test, lasso.predict(X_test))

# ref
lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{"alpha": alphas}]
n_folds = 10

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, "b--")
plt.semilogx(alphas, scores - std_error, "b--")

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel("CV score +/- std error")
plt.xlabel("alpha")
plt.axhline(np.max(scores), linestyle="--", color=".5")
plt.xlim([alphas[0], alphas[-1]])
plt.show()

# from sklearn.model_selection import KFold
# lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
# k_fold = KFold(3)
#
#
#
# for k, (train, test) in enumerate(k_fold.split(X, y)):
#     lasso_cv.fit(X.loc[train], y.loc[train].values.ravel())
#     print(
#         "[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(
#             k, lasso_cv.alpha_, lasso_cv.score(X.loc[train], y.loc[train].values.ravel())
#         )
#     )





