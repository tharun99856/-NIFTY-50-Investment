import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_regime_model(index_returns, n_regimes=3, n_iter=200, random_state=42):
    returns = index_returns.dropna().values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(returns)
    states = model.predict(returns)

    state_means = model.means_.flatten()
    state_labels = {}
    sorted_indices = np.argsort(state_means)

    if n_regimes == 3:
        state_labels[sorted_indices[0]] = "Bear"
        state_labels[sorted_indices[1]] = "Sideways"
        state_labels[sorted_indices[2]] = "Bull"
    elif n_regimes == 2:
        state_labels[sorted_indices[0]] = "Bear"
        state_labels[sorted_indices[1]] = "Bull"

    labeled_states = pd.Series(
        [state_labels[s] for s in states],
        index=index_returns.dropna().index,
        name="Regime",
    )

    return model, labeled_states


def compute_index_returns(df):
    pivot = df.pivot_table(index="Date", columns="Symbol", values="Close")
    daily_returns = pivot.pct_change()
    index_return = daily_returns.mean(axis=1)
    index_return.name = "NIFTY50_Return"
    return index_return


def regime_summary(regimes, index_returns):
    aligned = pd.concat([index_returns, regimes], axis=1).dropna()
    aligned.columns = ["Return", "Regime"]

    summary = aligned.groupby("Regime")["Return"].agg(
        Count="count",
        Mean_Daily_Return="mean",
        Std_Daily_Return="std",
        Min_Return="min",
        Max_Return="max",
    )
    summary["Annual_Return"] = summary["Mean_Daily_Return"] * 252
    summary["Annual_Vol"] = summary["Std_Daily_Return"] * np.sqrt(252)
    summary["Days_Pct"] = summary["Count"] / summary["Count"].sum() * 100

    return summary.round(4)


def regime_transition_matrix(regimes):
    states = regimes.values
    unique = regimes.unique()
    matrix = pd.DataFrame(0.0, index=unique, columns=unique)
    for i in range(len(states) - 1):
        matrix.loc[states[i], states[i + 1]] += 1
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix.round(3)


def get_regime_periods(regimes):
    periods = []
    current = regimes.iloc[0]
    start = regimes.index[0]

    for date, regime in regimes.items():
        if regime != current:
            periods.append({"Regime": current, "Start": start, "End": date, "Days": (date - start).days})
            current = regime
            start = date

    periods.append({"Regime": current, "Start": start, "End": regimes.index[-1],
                     "Days": (regimes.index[-1] - start).days})

    return pd.DataFrame(periods)
