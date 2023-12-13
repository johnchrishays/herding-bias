import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import csv
import time
import itertools
from numpy import random
from operator import itemgetter
import importlib
import multiprocessing as mp
import json

import experiment_helper
import exp_wrapper 

import finite_sim as sim
import finite_sim_wrapper as sim_wrapper

""" Runs simulations for herding and competition bias investigations using the same parameter values. """
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-r", "--n_runs", help="number of simulations to run")
argParser.add_argument("-l", "--n_listings", help="number of listings")
argParser.add_argument("--herding", action='store_true',  help="whether to run hearding simulations")
argParser.add_argument("--herding_no_recency", action='store_true', help="number of listings")
argParser.add_argument("--crowding", action='store_true', help="number of listings")
argParser.add_argument("--crtime", action='store_true', help="purchase rates over time, customer-side")
argParser.add_argument("--lrtime", action='store_true', help="purchase rates over time, listing-side")

args = argParser.parse_args()

print("BEGIN")

# Parameters
alpha = 1
epsilon = 1
tau = 1
lams = [ .1, 1, 10]

tsr_ac_al_values = [.5]
cr_a_C = 0.5
lr_a_L = 0.5

tsr_est_types = ['tsr_est_naive', 'tsri_1.0','tsri_2.0'] + ['mrd_direct', 'mrd_spillover_seller', 'mrd_spillover_buyer', 'mrd_avg']

customer_types = ['c1']
listing_types = ['l']
exp_conditions = ['control', 'treat']
rhos_pre_treat = {'l':1} #adds up to 1
customer_proportions = {'c1':1} #adds up to 1

# used for multiplicative utilities
customer_type_base = {'c1':.315 }
listing_type_base = {'l':1}
exp_condition_base = {'control':1, 'treat':1.25}

vs = {}
#multiplicative -- default
for c in customer_types:
    vs[c] = {}
    vs[c]['treat'] = {}   
    vs[c]['control'] = {}
    for l in listing_types:
        for e in exp_conditions:
            vs[c][e][l] = round(customer_type_base[c]
                                *exp_condition_base[e]
                                *listing_type_base[l],4)

for c in customer_types:
    print("customer", c)
    print("control", vs[c]['control'])
    print("treatment", vs[c]['treat'], "\n")
print('Rhos:', rhos_pre_treat)
print("Customer Proportions:", customer_proportions)

T_0 = 5
T_1 = 25

# normalizes time horizon by min(lam, tau)
T_start = {lam: T_0/min(lam,tau) for lam in lams}
T_end = {lam: T_1/min(lam,tau) for lam in lams}

varying_time_horizons = True

n_runs = int(args.n_runs)
n_listings = int(args.n_listings)
print('Number of runs:',n_runs)
print('Number of listings:',n_listings)

choice_set_type = 'alpha' #customers sample items into consideration set with prob alpha
k = None

params = sim_wrapper.calc_all_params(listing_types, 
                                     rhos_pre_treat, 
                                     customer_types, 
                                     customer_proportions, 
                                     vs, 
                                     alpha, 
                                     epsilon, 
                                     tau, 
                                     lams,
                                     tsr_ac_al_values, 
                                     cr_a_C, 
                                     lr_a_L)

def run_sim(herding, recency, fname_suffix):
    events = sim_wrapper.run_all_sims(n_runs, 
                                      n_listings, 
                                      T_start, 
                                      T_end, 
                                      choice_set_type, 
                                      k,
                                      alpha, 
                                      epsilon, 
                                      tau, 
                                      lams,
                                      **params, 
                                      herding=herding,
                                      recency=recency)
    est_stats_herding = sim_wrapper.calc_all_ests_stats("sample_", 
                                                        T_start, 
                                                        T_end, 
                                                        n_listings, 
                                                        tau=tau, 
                                                        tsr_est_types=tsr_est_types, 
                                                        events=events, 
                                                        varying_time_horizons=varying_time_horizons, 
                                                        **params, 
                                                        fname_suffix=fname_suffix)

def get_concat(events):
    concat_df = pd.concat(events).sort_index()
    concat_df["not_outside"] = concat_df["choice_type"] != "outside_option"
    return concat_df

def get_mean_t(df, t_0, t_1, l_type):
    if l_type == "l_treat" or l_type == "l_control":
        return (df[(df.index >= t_0) & (df.index <= t_1)]["choice_type"] == l_type).mean()
    else:
        return df[(df.index >= t_0) & (df.index <= t_1)].mean()["not_outside"]

def get_mean(df, l_type = None):
    return np.array([get_mean_t(df, t_0, t_0+1, l_type) for t_0 in range(T_1)])

def get_sds(dfs_t, dfs_c):
    for dfs in [dfs_t, dfs_c]:
        for df in dfs:
            df["not_outside"] = df["choice_type"] != "outside_option" 
    if exp_type == "lr_params":
        means = [np.array(get_mean(df_t, l_type="l_treat"))-np.array(get_mean(df_c, l_type="l_control")) for df_t,df_c  in zip(dfs_t, dfs_c)]
    else:
        means = [np.array(get_mean(df_t))-np.array(get_mean(df_c)) for df_t,df_c  in zip(dfs_t, dfs_c)]
    return np.std(np.vstack(means), axis=0)/math.sqrt(len(dfs_t))

def get_sds_bias(dfs_t, dfs_c, dfs_exp_t, dfs_exp_c):
    for dfs in [dfs_t, dfs_c, dfs_exp_t, dfs_exp_c]:
        for df in dfs:
            df["not_outside"] = df["choice_type"] != "outside_option" 
    if exp_type == "lr_params":
        means = [(np.array(get_mean(df_exp_t, l_type="l_treat")) - np.array(get_mean(df_exp_c, l_type="l_control"))) - (np.array(get_mean(df_t))-np.array(get_mean(df_c))) for df_t,df_c,df_exp_t,df_exp_c  in zip(dfs_t, dfs_c, dfs_exp_t, dfs_exp_c)]
    else:
        means = [(np.array(get_mean(df_exp_t))-np.array(get_mean(df_exp_c))) - (np.array(get_mean(df_t))-np.array(get_mean(df_c))) for df_t,df_c,df_exp_t,df_exp_c  in zip(dfs_t, dfs_c, dfs_exp_t, dfs_exp_c)]
    return np.std(np.vstack(means), axis=0)/math.sqrt(len(dfs_t))

def purchase_rate_plot(ax, avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, prefix = "CR"):
    ax.plot(avgs_t, label="Global treatment")
    ax.plot(avgs_c, label="Global control")
    ax.plot(avgs_exp_treat, label=prefix + " treatment")
    ax.plot(avgs_exp_control, label=prefix + " control")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Purchase rate")
    
def effect_size_plot(ax, avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr):
    effects = avgs_t - avgs_c
    ax.plot(effects, label="Global average treatment effect")
    ax.fill_between(range(T_1), effects + sds_gte[0], effects - sds_gte[1], alpha=0.2, label="S.E. of GATE")
    effects = avgs_exp_treat - avgs_exp_control
    ax.plot(effects, label="Estimate using difference-in-means")
    ax.fill_between(range(T_1), effects + sds_cr[0], effects - sds_cr[1], alpha=0.2, label="S.E. of estimator")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Effect size")
    
def bias_plot(ax, avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_bias):
    biases = (avgs_exp_treat - avgs_exp_control) - (avgs_t - avgs_c)
    ax.plot(biases, label="Bias of difference-in-means estimator")
    ax.fill_between(range(T_1), biases + sds_bias[0], biases - sds_bias[1], alpha=0.2, label="S.E. of estimator")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Bias")

def run_sims_over_time(N, herding, recency, exp_type):
    events = [get_events(herding, recency, exp_type) for _ in range(N)]
    customer_events_t, customer_events_c, customer_events_exp_treat, customer_events_exp_control = zip(*events)

    avgs_t = get_mean(get_concat(customer_events_t))
    avgs_c = get_mean(get_concat(customer_events_c))
    if exp_type == "lr_params":
        avgs_exp_treat = get_mean(get_concat(customer_events_exp_treat), l_type = 'l_treat')
        avgs_exp_control = get_mean(get_concat(customer_events_exp_control), l_type = 'l_control')
    else:
        avgs_exp_treat = get_mean(get_concat(customer_events_exp_treat))
        avgs_exp_control = get_mean(get_concat(customer_events_exp_control))
    sds_gte = get_sds(customer_events_t, customer_events_c)
    sds_cr = get_sds(customer_events_exp_treat, customer_events_exp_control)
    sds_bias = get_sds_bias(customer_events_t, customer_events_c, customer_events_exp_treat, customer_events_exp_control)

    return avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr, sds_bias

def plot_sims_over_time(avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr, sds_bias, fname=None, prefix="LR"):
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(18,4))
    purchase_rate_plot(axes[0], avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, prefix=prefix)
    effect_size_plot(axes[1], avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr)
    bias_plot(axes[2], avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_bias)
    if fname:
        plt.savefig(fname, dpi=1000)
    return fig, axes

def get_events(herding, recency, exp_type='cr_params'):

    s_full = {l: int(n_listings*rhos_pre_treat[l]) for l in rhos_pre_treat}
    lam = lams[1]

    # normalizes time horizon by min(lam, tau)
    T_start = T_0/min(lam,tau)
    T_end = T_1/min(lam,tau)

    lam_gammas = {c: lam for c in customer_proportions}
    vs = {}
    #multiplicative -- default
    for e in exp_conditions:
        vs[e] = {}
        for c in customer_types:
            vs[e][c] = {}
            for l in listing_types:
                vs[e][c][l] = round(customer_type_base[c]
                                    *exp_condition_base[e]
                                    *listing_type_base[l],4)
                
    def run_sim_same_params(herding, recency, vs):
        return sim.run_mc_listing_ids(choice_set_type="alpha", 
                   n=n_listings, 
                   k=None, 
                   s_0=copy.copy(s_full), 
                   s_full=s_full, 
                   T=T_end, 
                   thetas=rhos_pre_treat, 
                   gammas=customer_proportions, 
                   vs=vs, 
                   tau=tau, 
                   lam_gammas=lam_gammas, 
                   alpha=alpha, 
                   epsilon=epsilon, 
                   run_number=1, 
                   herding=herding, 
                   recency=recency)["events"]

    mc_treat = run_sim_same_params(herding, recency, vs['treat'])

    mc_control = run_sim_same_params(herding, recency, vs['control'])
    
    exp_params = params[exp_type][lam]
    s_full_exp = {l: int(n_listings*exp_params['rhos_exp'][l]) for l in exp_params['rhos_exp']}

    mc_exp = sim.run_mc_listing_ids(choice_set_type="alpha", 
               n=n_listings, 
               k=None, 
               s_0=copy.copy(s_full_exp), 
               s_full=s_full_exp, 
               T=T_end, 
               thetas=exp_params["rhos_exp"], 
               gammas=exp_params["gammas_exp"], 
               vs=exp_params['v_gammas_exp'], 
               tau=tau, 
               lam_gammas=exp_params['lam_gammas_exp'], 
               alpha=alpha, 
               epsilon=epsilon, 
               run_number=1, 
               herding=herding, 
               recency=recency)["events"]
    
    def get_customer_choice_events(df, exp=None):
        if exp == "c1_control" or exp == "c1_treat":
            new_df = df.loc[pd.notnull(df['choice_type']) & (df['customer_type'] == exp)]
        else:
            new_df = df.loc[pd.notnull(df['choice_type'])]
        return new_df[["choice_type", "time"]].set_index(["time"])

    customer_events_t = get_customer_choice_events(mc_treat)
    customer_events_c = get_customer_choice_events(mc_control) 
    if exp_type == "cr_params":
        customer_events_exp_control = get_customer_choice_events(mc_exp, exp="c1_control") 
        customer_events_exp_treat = get_customer_choice_events(mc_exp, exp="c1_treat")
    if exp_type == "lr_params":
        customer_events_exp_control = get_customer_choice_events(mc_exp, exp="l_control") 
        customer_events_exp_treat = get_customer_choice_events(mc_exp, exp="l_treat")
    return (customer_events_t, customer_events_c, customer_events_exp_treat, customer_events_exp_control)

if args.herding:
    print("HERDING")
    run_sim(herding=True, recency=True, fname_suffix="_herding.csv")

if args.herding_no_recency:
    print("HERDING NO RECENCY")
    run_sim(herding=True, recency=False, fname_suffix="_herding_no_recency.csv")

if args.crowding:
    print("CROWDING")
    run_sim(herding=False, recency=True, fname_suffix="_competition.csv")

if args.crtime:
    print("CR over time")
    herding = True
    recency = False
    exp_type = "cr_params"

    T_0 = 5
    T_1 = 30

    avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr, sds_bias = run_sims_over_time(n_runs, herding, recency, exp_type)
    plot_sims_over_time(avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr, sds_bias, fname="cr_herdingovertime.png", prefix="CR")

if args.lrtime:
    print("LR over time")
    herding = True
    recency = True
    exp_type = "lr_params"

    T_0 = 5
    T_1 = 60
    avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr, sds_bias = run_sims_over_time(n_runs, herding, recency, exp_type)
    plot_sims_over_time(avgs_t, avgs_c, avgs_exp_treat, avgs_exp_control, sds_gte, sds_cr, sds_bias, fname="lr_herdingovertime.png", prefix="LR")

print("END")
