"""C-dev: a classical learner fitted on the real dev rows (ridge for regression, multinomial logistic for classes),
on the same feature table; regularisation chosen by internal CV within dev. Used only where the model card already
has a dev-fitted classical baseline."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_predict(kind, Xd, yd, Xe):
    ok = np.isfinite(yd)
    Xd, yd = np.asarray(Xd, float)[ok], yd[ok]
    Xd = np.where(np.isfinite(Xd), Xd, np.nan)
    Xe = np.where(np.isfinite(Xe), np.asarray(Xe, float), np.nan)
    if kind == 'reg':
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        m.fit(Xd, yd)
        return m.predict(Xe)
    cls = np.unique(yd)
    if len(cls) < 2:
        return np.full(len(Xe), cls[0] if len(cls) else 0.0)
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xd, yd.astype(int))
    if kind == 'clf':
        return m.predict_proba(Xe)[:, list(m.classes_).index(1)]
    return m.predict(Xe).astype(float)


def predict_all(targets, Xd, Yd, Xe):
    return {t['name']: fit_predict(t['kind'], Xd, Yd[t['name']], Xe) for t in targets}


def cv(a, X, Y, folds=5, seed=0):
    from sklearn.model_selection import KFold
    import harness as h
    oof = {t['name']: np.full(len(X), np.nan) for t in a.TARGETS}
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        for t in a.TARGETS:
            oof[t['name']][te] = fit_predict(t['kind'], X[tr], Y[t['name']][tr], X[te])
    sc = h.target_scores(a, Y, oof)
    return {'scores': sc, 'headline': h.headline(a, sc)}
