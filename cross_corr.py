from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.signal import correlate
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def is_significant(cross_correlation):
    num_obs = len(cross_correlation)
    middle_index = len(cross_correlation)//2
    cross_correlation = pd.Series(cross_correlation)
    cross_correlation.index = range(len(cross_correlation))
    max_index = cross_correlation[
        cross_correlation == cross_correlation.max()
    ].index[0]
    lag = abs(middle_index - max_index)
    return cross_correlation.max() > (2/np.sqrt(num_obs - lag))

def cross_correlation_plot(feature_one, feature_two):
    feature_one = feature_one - feature_one.mean()
    feature_two = feature_two - feature_two.mean()
    cross_correlation = correlate(feature_one, feature_two)
    cross_correlation /= (len(feature_one) * feature_one.std() * feature_two.std())
    plt.xcorr(feature_one, feature_two, maxlags=5)
    absolute_cross_correlation = abs(cross_correlation)
    print("Max cross correlation", cross_correlation.max())
    print("Average cross correlation", cross_correlation[:20].mean())
    if is_significant(cross_correlation):
        statistically_significant = True
        print("and is statistically significant")
    else:
        statistically_significant = False
        print("and is not statistically significant")
    print()
    plt.show()
    cross_correlation = pd.Series(cross_correlation)
    cross_correlation.index = range(len(cross_correlation))
    return cross_correlation, statistically_significant

def compare_timeseries(feature_one, feature_two, tag, max_cross_correlations):
    print(tag)
    cross_correlation, statistically_significant = cross_correlation_plot(
        feature_one, feature_two
    )
    if statistically_significant:
        max_cross_correlations.append(cross_correlation.max())
    return max_cross_correlations

def smooth_feature(feature):
    feature_smoother = ExponentialSmoothing(
        feature,
        trend="add"
    ).fit(use_boxcox=True)
    smoothed_feature = feature_smoother.predict(start=0, end=len(feature)-1)
    smoothed_feature.fillna(0, inplace=True)
    return smoothed_feature

def check_smoothed_feature(smoothed_feature):
    zero_count = (smoothed_feature == 0).astype(int).sum(axis=0)
    return (zero_count == 0) and np.isfinite(smoothed_feature).all()


def timeseries_analysis(series_one,
                        series_two,
                        cross_correlated,
                        cointegrated, count,
                        max_cross_correlations,
                        cointegrated_series, tag):
    
    if breaks_cusumolsresid(series_one)[1] > 0.05:
        pass
    if breaks_cusumolsresid(series_two)[1] > 0.05:
        pass
    if adfuller(series_one)[1] < 0.05 and adfuller(series_two)[1] < 0.05:
        max_cross_correlations = compare_timeseries(
            series_one, series_two, tag, 
            max_cross_correlations
        )
        cross_correlated += 1

    if adfuller(series_one)[1] > 0.05 and adfuller(series_two)[1] < 0.05:
        try:
            smoothed_feature = smooth_feature(series_one)
            if np.isfinite(smoothed_feature).all() and (smoothed_feature.iloc[0] != smoothed_feature).all():
                max_cross_correlations = compare_timeseries(
                    smoothed_feature, series_two, tag, 
                    max_cross_correlations
                )
                cross_correlated += 1
            
        except ValueError:
            zero_percent = (series_one == 0).astype(int).sum(axis=0)/len(series_one)
            if zero_percent < 0.05:
                feature = series_one.replace(to_replace=0, method='ffill')
                smoothed_feature = smooth_feature(feature)
                if check_smoothed_feature(smoothed_feature):
                    max_cross_correlations = compare_timeseries(
                        smoothed_feature, series_two, tag, 
                        max_cross_correlations
                    )
                    cross_correlated += 1
    if adfuller(series_one)[1] < 0.05 and adfuller(series_two)[1] > 0.05:
        try:
            smoothed_feature = smooth_feature(series_two)
            if np.isfinite(smoothed_feature).all() and (smoothed_feature.iloc[0] != smoothed_feature).all():
                max_cross_correlations = compare_timeseries(
                    series_one, smoothed_feature, tag, 
                    max_cross_correlations
                )
                cross_correlated += 1
        except ValueError:
            zero_percent = (series_two == 0).astype(int).sum(axis=0)/len(series_two)
            if zero_percent < 0.05:
                feature = series_two.replace(to_replace=0, method='ffill')
                smoothed_feature = smooth_feature(feature)
                if check_smoothed_feature(smoothed_feature):
                    max_cross_correlations = compare_timeseries(
                        series_one, smoothed_feature, tag, 
                        max_cross_correlations
                    )
                    cross_correlated += 1
    if adfuller(series_one)[1] > 0.05 and adfuller(series_two)[1] > 0.05:
        print(tag)
        coint_pvalue = coint(series_one, series_two)[1]
        cointegrated += 1
        print("no cointegration with p-value", coint_pvalue)
        if coint_pvalue < 0.05:
            cointegrated_series += 1
    count += 1
    return (
        cross_correlated,
        cointegrated, count,
        max_cross_correlations,
        cointegrated_series
    )

def timeseries_print(
        cross_correlated,
        cointegrated, count,
        max_cross_correlations,
        cointegrated_series
):
    print()
    print("possibly cross correlated", cross_correlated/count)
    print("possibly cointegrated", cointegrated/count)
    print("percent statistically significant", len(max_cross_correlations)/count)
    print("total statistically significant", len(max_cross_correlations))
    print("overall total", count)
    print("average maximum cross correlation", np.mean(max_cross_correlations))
    print("standard deviation maximum cross_correlation", np.std(max_cross_correlations))
    print("percent cointegrated series", cointegrated_series/cointegrated)
    print("total cointegrated series", cointegrated_series)
    print("total related series", cointegrated_series + len(max_cross_correlations))
    print("percent total related series", (cointegrated_series + len(max_cross_correlations))/count)
    print("missed", 1 - (cross_correlated + cointegrated)/count)
