import numpy as np
import pandas as pd
from numpy import random
from operator import itemgetter
import copy
import time
import itertools
from multiprocessing import Pool
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as tck 
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

import exp_wrapper
import experiment_helper
import finite_sim


import os
import csv
import json

"""
Runs multiple simulations of finite model with N listings. 
Starts with all listings available at time t=0, and simulates
arrival of customers and resulting bookings over time. 

Parameters used across different functions:
Events - time series of state of system and events that happen.
Event_times - dictionary where keys are listings and values are the times
at which listing became booked or unavailable. 
Listing_dfs - df indexed by listing ID, gives number of bookings available at end of time horizon.
"""

def mult_runs_events(choice_type, k, exp_type, exp_param, tau, alpha, epsilon,
                        n_runs, n_listings, T_start, T_end, 
                        a_C=None, a_L=None,
                        tsr_est_types=None, cr_weight=None, num_threads=45, 
                        herding=False, recency=True):
    s_full = {l: int(n_listings * exp_param['rhos_exp'][l]) for l in exp_param['thetas_exp']}

    events = {} #
    
    if exp_type == 'cr':
        events['cr'] = []
    elif exp_type == 'lr':
        events['lr'] = []
    elif exp_type == 'clustered':
        events['clustered'] = []
    elif exp_type == 'tsr':
        for est in tsr_est_types:
            events[est] = []
    
    pool = Pool(num_threads)
    
    for events_one_run in pool.map(partial(finite_sim.run_mc_listing_ids, 
                                           choice_type, 
                                           n_listings, 
                                           k, 
                                           s_full,  
                                           s_full, 
                                           T_end, 
                                           exp_param['thetas_exp'], 
                                           exp_param['gammas_exp'], 
                                           exp_param['v_gammas_exp'], 
                                           tau, 
                                           exp_param['lam_gammas_exp'], 
                                           alpha, 
                                           epsilon, 
                                           herding=herding, 
                                           recency=recency), 
                                   range(n_runs)):
        if exp_type=="cr":
            events['cr'].append([events_one_run['events']])
        elif exp_type=='lr':
            events['lr'].append([events_one_run['events']])
        elif exp_type=='clustered':
            events['clustered'].append([events_one_run['events']])
        elif exp_type == 'tsr':
            for est in tsr_est_types:
                events[est].append([events_one_run['events']])
    pool.close()
    pool.join()

    return {'events':events}


def mult_runs_global_conditions(choice_type, k, global_cond, params, tau, alpha, epsilon,
                                n_runs, n_listings, T_start, T_end, num_threads=50, herding=True, recency=False):
    """
    Runs simulations for GC and GT conditions
    """
    events = []
    params['thetas_c'] = list(params['rhos_c'].keys())
    params['thetas_t'] = list(params['rhos_t'].keys())
    
    if global_cond == 'global_treat':
        s_full = {l: int(n_listings * params['rhos_t'][l]) for l in params['rhos_t']}
    elif global_cond == 'global_control':
        s_full = {l: int(n_listings * params['rhos_c'][l]) for l in params['rhos_c']}


    pool = Pool(num_threads)
    if global_cond == 'global_treat':
        for events_one_run in pool.map(partial(finite_sim.run_mc_listing_ids, 
                                               choice_type, 
                                               n_listings, 
                                               k, 
                                               s_full, 
                                               s_full, 
                                               T_end, 
                                               params['thetas_t'], 
                                               params['gammas_t'], 
                                               params['v_gammas_t'], 
                                               tau, 
                                               params['lam_gammas_t'], 
                                               alpha, 
                                               epsilon, 
                                               herding=herding, 
                                               recency=recency), 
                                       range(n_runs)):
            events.append([events_one_run])
    
    elif global_cond == 'global_control':
        for events_one_run in pool.map(partial(finite_sim.run_mc_listing_ids, 
                                               choice_type, 
                                               n_listings, 
                                               k, 
                                               s_full, 
                                               s_full, 
                                               T_end, 
                                               params['thetas_c'], 
                                               params['gammas_c'], 
                                               params['v_gammas_c'], 
                                               tau, 
                                               params['lam_gammas_c'], 
                                               alpha, 
                                               epsilon, 
                                               herding=herding, 
                                               recency=recency), 
                                       range(n_runs)):
            events.append([events_one_run])
    pool.close()
    pool.join()

    return events

def calc_all_global_cond_rates(T_start, T_end, n, events_mult_sims):
    """
    Calculates rate of booking in global control simulation.
    """
    booking_rates = []
    for events in events_mult_sims:
        events = events[0]['events']
        bookings = events[(events['is_customer']==1) & 
                          (events['time']>T_start) & 
                          (events['time']<T_end) & 
                          (events['choice_type']!='outside_option')]
        n_bookings = len(bookings)
        booking_rate = n_bookings / n / (T_end - T_start)
        booking_rates.append(booking_rate)

    return booking_rates



def calc_estimates_from_events(events, exp_type, exp_params, 
                               T_start, T_end, n_listings, 
                                a_C=None, a_L=None,
                                customer_alloc=None, listing_alloc=None,
                                tsr_est_types=None, 
                                cr_weights=None,
                                fixed_ac_al=False,
                                varying_time_horizons=False,
                                save=False, file_path=None, fname_suffix=".csv"):
    start_times = T_start
    end_times = T_end
    
    estimates = {}
    for lam in events.keys():
        if varying_time_horizons:
            T_start = start_times[lam]
            T_end = end_times[lam]
        
        estimates[lam] = {}
        
        if exp_type == 'cr':
            estimates[lam]['cr'] = []
            estimates[lam]['cr_adj'] = []
            for events_one_run in events[lam]['events']['cr']:
                events_one_run = events_one_run[0]
                est = finite_sim.calc_cr_estimate(T_start, 
                                                  T_end, 
                                                  n_listings, 
                                                  events_one_run, 
                                                  a_C, 
                                                  exp_params[lam]['gammas_c'], 
                                                  exp_params[lam]['gammas_t'])

                est_adj = finite_sim.calc_adj_cr_estimate(T_start, 
                                                          T_end, 
                                                          n_listings, 
                                                          events_one_run, 
                                                          a_C, 
                                                          exp_params[lam]['gammas_c'], 
                                                          exp_params[lam]['gammas_t'])
                
                estimates[lam]['cr'].append(est)
                estimates[lam]['cr_adj'].append(est)
            
        elif exp_type == 'lr':
            estimates[lam]['lr'] = []
            estimates[lam]['lr_adj'] = []
            for events_one_run in events[lam]['events']['lr']:
                events_one_run = events_one_run[0]
                est = finite_sim.calc_lr_estimate(T_start, 
                                                  T_end, 
                                                  n_listings, 
                                                  events_one_run, 
                                                  a_L, 
                                                  exp_params[lam]['thetas_c'], 
                                                  exp_params[lam]['thetas_t'])

                est_adj = finite_sim.calc_adj_lr_estimate(T_start, 
                                                          T_end, 
                                                          n_listings, 
                                                          events_one_run, 
                                                          a_L, 
                                                          exp_params[lam]['thetas_c'], 
                                                          exp_params[lam]['thetas_t'])
                
                estimates[lam]['lr'].append(est)
                estimates[lam]['lr_adj'].append(est_adj)
        elif exp_type == 'tsr':
            any_tsr_exp = list(events[lam]['events'].keys())[0]
            for est in tsr_est_types:
                estimates[lam][est] = []
            for events_one_run in events[lam]['events'][any_tsr_exp]: # all tsr_ests have same events
                events_one_run = events_one_run[0]
                ests = finite_sim.calc_tsr_estimate(T_start, 
                                                    T_end, 
                                                    n_listings, 
                                                    cr_weights[lam], 
                                                    events_one_run, 
                                                    a_C[lam], 
                                                    a_L[lam], 
                                                    exp_params[lam]['gammas_c'], 
                                                    exp_params[lam]['gammas_t'], 
                                                    exp_params[lam]['thetas_c'], 
                                                    exp_params[lam]['thetas_t'], 
                                                    tsr_est_types)
                for est in tsr_est_types:
                    estimates[lam][est].append(ests[est])
        
        elif exp_type == 'gc':
            estimates[lam]['gc'] = []
            estimates[lam]['gc'] = []
            for events_one_run in events[lam]['events']['cr']:
                events_one_run = events_one_run[0]
                est = finite_sim.calc_global_estimate(T_start, 
                                                      T_end, 
                                                      n_listings, 
                                                      events_one_run, 
                                                      exp_params[lam]['gammas_c'])
                estimates[lam]['gc'].append(est)
         
    if exp_type == 'cr':
        estimates_df = {}
        for est in ['cr', 'cr_adj']:
            estimates_df[est] = {lam:estimates[lam][est] 
                                        for lam in estimates.keys()}
            estimates_df[est] = pd.DataFrame(estimates_df[est])
            if save:
                estimates_df[est].to_csv(file_path+str(est)+"_estimates" + fname_suffix)

    elif exp_type == 'lr':
        estimates_df = {}
        for est in ['lr', 'lr_adj']:
            estimates_df[est] = {lam:estimates[lam][est] 
                                        for lam in estimates.keys()}
            estimates_df[est] = pd.DataFrame(estimates_df[est])
            if save:
                estimates_df[est].to_csv(file_path+str(est)+"_estimates" + fname_suffix)
            
    elif exp_type == 'tsr':
        estimates_df = {}
        rand_type='_opt'
        for tsr_est in tsr_est_types:
            estimates_df[tsr_est] = {lam:estimates[lam][tsr_est] 
                                        for lam in estimates.keys()}
            estimates_df[tsr_est] = pd.DataFrame(estimates_df[tsr_est])
            if save:
                estimates_df[tsr_est].to_csv(file_path+str(tsr_est)+rand_type+"_estimates" + fname_suffix)

    elif exp_type == 'gc':
        estimates_df = {lam:estimates[lam]['gc'] for lam in estimates.keys()}
        estimates_df = pd.DataFrame(estimates_df)
        if save:
            estimates_df.to_csv(file_path+"gc_estimates" + fname_suffix)

    return estimates_df
    


def calc_n_bookings_from_events(events, exp_type, exp_params, 
                               T_start, T_end, n_listings, 
                                a_C=None, a_L=None,
                                tsr_est_types=None, cr_weights=None,
                                fixed_ac_al=False,
                                save=False, file_path=None, fname_suffix=".csv"):
    """
    Calculates number of bookings made.
    """
    
    bookings_per_cell = {}
    bookings_dfs = {}
    
    for lam in events.keys():
        bookings_per_cell[lam] = {}
        if exp_type == 'cr':
            bookings_per_cell[lam]['cr'] = []
            for events_one_run in events[lam]['cr']:
                events_one_run = events_one_run[0]
                bookings = finite_sim.calc_bookings_per_cell(T_start, T_end, events_one_run,
                                                             gammas_c=exp_params[lam]['gammas_c'],
                                                            gammas_t=exp_params[lam]['gammas_t'],
                                                            thetas_c=[],
                                                            thetas_t=exp_params[lam]['thetas_exp']
                                                            )
                bookings_per_cell[lam]['cr'].append(bookings)
            bookings_dfs[lam] = pd.DataFrame(bookings_per_cell[lam]['cr'])
            if save:
                bookings_dfs[lam].to_csv(file_path+"cr_n_bookings"+str(lam).replace(".","")+fname_suffix)
            
        elif exp_type == 'lr':
            bookings_per_cell[lam]['lr'] = []
            for events_one_run in events[lam]['lr']:
                events_one_run = events_one_run[0]
                bookings = finite_sim.calc_bookings_per_cell(T_start, T_end, events_one_run,
                                                             gammas_c=[],
                                                            gammas_t=exp_params[lam]['gammas_exp'],
                                                            thetas_c=exp_params[lam]['thetas_c'],
                                                            thetas_t=exp_params[lam]['thetas_t'])
                bookings_per_cell[lam]['lr'].append(bookings)
            bookings_dfs[lam] = pd.DataFrame(bookings_per_cell[lam]['lr'])
            if save:
                bookings_dfs[lam].to_csv(file_path+"lr_n_bookings"+str(lam).replace(".","") + fname_suffix)
        elif exp_type == 'tsr':
            for est in tsr_est_types:
                bookings_per_cell[lam][est] = []
            for events_one_run in events[lam]['tsr_est_naive']: # all tsr_ests have same events
                events_one_run = events_one_run[0]
                bookings = finite_sim.calc_bookings_per_cell(T_start, T_end, events_one_run,
                                                             gammas_c=exp_params[lam]['gammas_c'],
                                                            gammas_t=exp_params[lam]['gammas_t'],
                                                            thetas_c=exp_params[lam]['thetas_c'],
                                                            thetas_t=exp_params[lam]['thetas_t'])
                for est in tsr_est_types:
                    bookings_per_cell[lam][est].append(bookings)
            bookings_dfs[lam] = {}
            for tsr_est in tsr_est_types:
                bookings_dfs[lam][tsr_est] = pd.DataFrame(bookings_per_cell[lam][tsr_est])
                if save:
                    bookings_dfs[lam][tsr_est].to_csv(file_path+tsr_est+"_n_bookings"+str(lam).replace(".","")+fname_suffix)
    return bookings_dfs


def calc_estimator_stats(estimator_type, estimates_df, gtes, tau, normalized=False,
                         is_tsr_fixed=False, tsr_fixed_ac_val=None,
                         ):
    """
    For a given experiment and estimator type, calculates the bias,
    standard error, and root mean squared error (RMSE) of the estimators.
    """
    norm_constants = {}
    for lam in gtes.keys():
        if normalized:
            norm_constants[lam] = 1/ min(lam, tau)
        else:
            norm_constants[lam] = 1 
    
    estimates_df = estimates_df * norm_constants
    gtes = {lam: gtes[lam]*norm_constants[lam] for lam in gtes.keys()}
    
    stats = pd.DataFrame()

    stats['est'] = estimates_df.mean()
    stats['gte'] = [gtes[lam] for lam in gtes.keys()]
    stats['bias'] = (estimates_df - gtes).mean()
    stats['bias_over_GTE'] = stats['bias'] / stats['gte']
    stats['abs_bias_over_GTE'] = np.abs(stats['bias']) / stats['gte']
    stats['std'] = estimates_df.std()
    stats['std_over_GTE'] = stats['std'] / stats['gte']
    stats['rmse'] = np.sqrt(((estimates_df - gtes)**2).mean())
    stats['rmse_over_GTE'] = np.sqrt(((estimates_df - gtes)**2).mean()) / stats['gte']
    
    stats = pd.concat([stats], keys=[estimator_type], names=['estimator_type', 'lambda'])
    
    return stats



def calc_all_params(listing_types, rhos_pre_treat, customer_types, customer_proportions, vs, 
                    alpha, epsilon, tau, lams,
                    tsr_ac_al_values, cr_a_C, lr_a_L,
                    a_Cs=None, a_Ls=None):
    """
    Given global control parameters, calculates experiment parameters for all experiment types.
    """
    tsr_fixed_params = {}
    tsr_opt_params = {}
    cr_params = {}
    lr_params = {}
    global_params = {}
    # re-order dict
    vs_global = {}
    vs_global['treat'] = {}
    vs_global['control'] = {}

    for c, c_dict in vs.items():
        vs_global['treat'][c] = {}
        vs_global['control'][c] = {}
        for e, e_dict in c_dict.items():
            for l, l_dict in e_dict.items():
                vs_global[e][c][l] = vs[c][e][l]

    
    # Interpolating from cr_a_C to cr_a_L
    a_Cs = {}
    a_Ls = {}
    cr_weights = {}
    for lam in lams:
        cr_weights[lam] =  np.exp(-1*lam/tau)
        a_Cs[lam] = cr_a_C * cr_weights[lam] + 1*(1-cr_weights[lam])
        a_Ls[lam] = 1 * cr_weights[lam] + lr_a_L * (1-cr_weights[lam])

    for val in tsr_ac_al_values:
        tsr_fixed_params[val] = {}
        a_C = val
        a_L = val
        for lam in lams:
            tsr_fixed_params[val][lam] = exp_wrapper.tsr_exp_induced_parameters(listing_types, rhos_pre_treat, 
                                                                                a_C, a_L, 
                                                                                customer_types, customer_proportions,
                                                                                alpha, vs, lam)
            
    for lam in lams:    
        cr_params[lam] = exp_wrapper.customer_side_exp_induced_parameters(listing_types, rhos_pre_treat, 
                                                                cr_a_C, customer_types, alpha, vs, 
                                                                lam, customer_proportions)
        lr_params[lam] = exp_wrapper.listing_side_exp_induced_parameters(listing_types, rhos_pre_treat, 
                                                                lr_a_L, customer_types, alpha, vs, 
                                                                lam, customer_proportions)
        tsr_opt_params[lam] = exp_wrapper.tsr_exp_induced_parameters(listing_types, rhos_pre_treat, 
                                                                    a_Cs[lam], a_Ls[lam], 
                                                                    customer_types, customer_proportions,
                                                                    alpha, vs, lam)
        global_params[lam] = {
            "rhos_c": rhos_pre_treat,
            "gammas_c": customer_types,
            "v_gammas_c": vs_global['control'],
            "lam_gammas_c": {c: lam for c in customer_types},
            "rhos_c": rhos_pre_treat,
            "gammas_t": customer_types,
            "v_gammas_t": vs_global['treat'],
            "lam_gammas_t": {c: lam for c in customer_types},
            "rhos_t": rhos_pre_treat
        }
    return {'a_Cs':a_Cs, 'a_Ls':a_Ls, 'cr_weights':cr_weights, 
            'cr_params':cr_params, 'lr_params':lr_params, 'tsr_fixed_params':tsr_fixed_params,
            'tsr_opt_params':tsr_opt_params, 'cr_a_C':cr_a_C, 'lr_a_L':lr_a_L, 
            'tsr_ac_al_values':tsr_ac_al_values, 'global_params': global_params}


def run_all_sims(n_runs, n_listings, T_start, T_end, choice_set_type, k, 
                 alpha, epsilon, tau, lams,
                 a_Cs, a_Ls, cr_weights, cr_params, lr_params, tsr_fixed_params, tsr_opt_params,
                 cr_a_C, lr_a_L, tsr_ac_al_values, global_params, herding=False, recency=True):
    """
    Runs multiple simulations for all experiment types.
    """
    tsr_est_types = ['tsr_est_naive', 'tsr_est_gw','tsr_est_hl', 'tsr_est_robust'] + ['mrd_direct', 'mrd_spillover_seller', 'mrd_spillover_buyer', 'mrd_avg']
    cr_events = {}
    lr_events = {}
    tsr_opt_events = {}
    tsr_fixed_events = {}
    global_control_events = {}
    global_treat_events = {}
    global_control_rates = {}
    global_treat_rates = {}
    gtes = {}
    
    for val in tsr_ac_al_values:
        tsr_fixed_events[val] = {}

    for lam in lams:
        print("lambda=",lam)
        t0 = time.time()
        
        # CR experiment
        cr_events[lam] = mult_runs_events(choice_set_type, k, 'cr', cr_params[lam], 
                                                    tau, alpha, epsilon,
                                                n_runs, n_listings, T_start[lam], T_end[lam], 
                                                a_C=cr_a_C, herding=herding, recency=recency)

        # LR experiment
        lr_events[lam] = mult_runs_events(choice_set_type, k, 'lr', lr_params[lam], 
                                                    tau, alpha, epsilon,
                                    n_runs, n_listings, T_start[lam], T_end[lam], 
                                    a_L=lr_a_L, herding=herding, recency=recency)

        # TSR experiment
        # a_C, a_L varying based on lambda/tau
        tsr_opt_events[lam] = mult_runs_events(choice_set_type, k, 'tsr', tsr_opt_params[lam], 
                                                        tau, alpha, epsilon,
                                                    n_runs, n_listings, T_start[lam], T_end[lam], 
                                                    a_C=a_Cs[lam], a_L=a_Ls[lam], 
                                                    tsr_est_types=tsr_est_types,
                                                cr_weight=cr_weights[lam], herding=herding, recency=recency)

        
        global_control_events[lam] = mult_runs_global_conditions(choice_set_type, 
                                                                  k, 
                                                                  "global_control", 
                                                                  global_params[lam], 
                                                                  tau, 
                                                                  alpha, 
                                                                  epsilon, 
                                                                  n_runs, 
                                                                  n_listings, 
                                                                  T_start[lam], 
                                                                  T_end[lam],
                                                                  herding=herding, 
                                                                  recency=recency)
        global_treat_events[lam] = mult_runs_global_conditions(choice_set_type, 
                                                                  k, 
                                                                  "global_treat", 
                                                                  global_params[lam], 
                                                                  tau, 
                                                                  alpha, 
                                                                  epsilon, 
                                                                  n_runs, 
                                                                  n_listings, 
                                                                  T_start[lam], 
                                                                  T_end[lam],
                                                                  herding=herding, 
                                                                  recency=recency)
        
        global_control_rates[lam] = calc_all_global_cond_rates(T_start[lam], 
                                                               T_end[lam], 
                                                               n_listings, 
                                                               global_control_events[lam])
        global_treat_rates[lam] = calc_all_global_cond_rates(T_start[lam], 
                                                             T_end[lam], 
                                                             n_listings, 
                                                             global_treat_events[lam])
        gtes[lam] = np.mean(global_treat_rates[lam]) - np.mean(global_control_rates[lam])
        t1 = time.time()
        print("Time elapsed: ", round(t1-t0,2))

    return {'cr_events': cr_events, 'lr_events':lr_events, 
            'tsr_opt_events':tsr_opt_events, 
            'tsr_fixed_events':tsr_fixed_events,
            'gtes':gtes}
    

def calc_all_ests_stats(file_path, T_start, T_end, n_listings,
                    cr_a_C, lr_a_L, a_Cs, a_Ls, 
                    tsr_ac_al_values, cr_weights,
                    cr_params, lr_params, tsr_opt_params, tsr_fixed_params,
                    events, 
                    tau, tsr_est_types, global_params,
                    gtes=None, normalized_by_lam=True,
                    varying_time_horizons=False, fname_suffix=".csv"):
    """
    For experiments (CR, LR, TSR), calculates the bias,
    standard error, and root mean squared error (RMSE) of the estimators.
    """
    cr_events = events['cr_events']
    lr_events = events['lr_events']
    tsr_opt_events = events['tsr_opt_events']
    if gtes:
        gtes=gtes
    else:
        gtes = events['gtes']
    
    cr_estimates = calc_estimates_from_events(cr_events, 'cr', cr_params, 
                                                T_start, T_end, n_listings, a_C=cr_a_C,
                                                varying_time_horizons=varying_time_horizons, 
                                                save=True, file_path=file_path, fname_suffix=fname_suffix)
    lr_estimates = calc_estimates_from_events(lr_events, 'lr', lr_params, 
                                                T_start, T_end, n_listings, a_L=lr_a_L,
                                                varying_time_horizons=varying_time_horizons, 
                                                save=True, file_path=file_path, fname_suffix=fname_suffix)
    tsr_opt_estimates = calc_estimates_from_events(tsr_opt_events, 'tsr', tsr_opt_params, 
                                                    T_start, T_end, n_listings, 
                                                    a_C=a_Cs, a_L=a_Ls,  
                                                    tsr_est_types=tsr_est_types,
                                                    cr_weights=cr_weights, fixed_ac_al=False,
                                                    varying_time_horizons=varying_time_horizons, 
                                                    save=True, file_path=file_path, fname_suffix=fname_suffix)

    cr_stats = {}
    for est in ['cr','cr_adj']:
        cr_stats[est] = calc_estimator_stats(est, cr_estimates[est], gtes, tau, 
                                                    normalized=normalized_by_lam)

    lr_stats = {}
    for est in ['lr','lr_adj']:
        lr_stats[est] = calc_estimator_stats(est, lr_estimates[est], gtes, tau, 
                                                    normalized=normalized_by_lam)
        
    tsr_opt_stats = {}
    for est in tsr_est_types:
        tsr_opt_stats[est] = calc_estimator_stats(est, tsr_opt_estimates[est], gtes, tau, 
                                                    normalized=normalized_by_lam)


    total_stats_df = pd.concat([cr_stats[est] for est in ['cr', 'cr_adj']]
                                +[lr_stats[est] for est in ['lr', 'lr_adj']]
                                +[tsr_opt_stats[tsr_est] for tsr_est in tsr_est_types])
    total_stats_df.to_csv(file_path+"total_stats"+fname_suffix)
    return total_stats_df

def get_plot(ax, est_stats, stat, title, ylab, xlab, yscale="linear", legend=False, pal="Set3"):

    rgb_0_1_array = np.array(sns.color_palette(pal, est_stats.index.levshape[0]))

    (est_stats
        .unstack(level=0)[stat]
        .plot(kind='bar', color=rgb_0_1_array, ax=ax))

    if yscale:
        ax.set_yscale(yscale)
        if (stat == "bias" or stat == "bias_over_GTE") and yscale == "symlog":
            ax.set_ylim((-10, 10))
        elif (yscale=="symlog"):
            ax.set_ylim((0, 50))

    ax.set_title(title)
    if legend:
        ax.legend(loc=[1.02,0.5])
    else:
        ax.get_legend().remove()
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(tck.MultipleLocator(1))
    sns.despine()

def plot_all_stats(est_stats, suptitle, yscale = "symlog"):

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10))

    get_plot(axes[0][0], 
             est_stats, 
             'bias', 
             "", 
             "Bias", 
             "Relative user arrival rate $\lambda/\\tau$",
             yscale=yscale)

    get_plot(axes[0][1], 
             est_stats, 
             'bias_over_GTE', 
             "", 
             "Bias / GTE", 
             "Relative user arrival rate $\lambda/\\tau$",
             yscale=yscale)

    get_plot(axes[1][0], 
             est_stats, 
             'std', 
             "", 
             "SE", 
             "Relative user arrival rate $\lambda/\\tau$",
             yscale=yscale)

    get_plot(axes[1][1], 
             est_stats, 
             'std_over_GTE', 
             "", 
             "SE / GTE", 
             "Relative user arrival rate $\lambda/\\tau$",
             yscale=yscale)

    get_plot(axes[2][0], 
             est_stats, 
             'rmse', 
             "", 
             "RMSE", 
             "Relative user arrival rate $\lambda/\\tau$",
             yscale=yscale)

    get_plot(axes[2][1], 
             est_stats, 
             'rmse_over_GTE', 
             "", 
             "RMSE / GTE", 
             "Relative user arrival rate $\lambda/\\tau$",
             yscale=yscale,
             legend=True)

    fig.suptitle(suptitle)
    return fig, axes

def plot_and_save(file_path, total_stats_df):
    stat = 'abs_bias_over_GTE'

    (total_stats_df
    .rename({'cr':"Customer-Side", 'lr':"Listing-Side",
             'tsr_est_naive':"TSR-Naive",
             'tsri_1.0':"TSRI-1",
             'tsri_2.0':"TSRI-2"}, 
            level=0)
    .unstack(level=0)[stat]
    .plot(kind='bar'))

    plt.yscale('log')
    plt.xlabel("Relative Demand $\lambda/\\tau$")
    plt.title("Bias")
    plt.legend(loc=[1.02,0.5])
    plt.ylabel("Abs value of Bias/GTE")
    plt.tight_layout()
    sns.despine()
    plt.savefig(file_path+stat+".png")
    
    stat = 'std_over_GTE'

    (total_stats_df
    .rename({'cr':"Customer-Side", 'lr':"Listing-Side",
             'tsr_est_naive':"TSR-Naive",
             'tsri_1.0':"TSRI-1",
             'tsri_2.0':"TSRI-2"}, level=0)
    .unstack(level=0)[stat]
    .plot(kind='bar'))

    plt.yscale('log')
    plt.xlabel("Relative Demand $\lambda/\\tau$")
    plt.title("Standard Error")
    plt.legend(loc=[1.02,0.5])
    plt.ylabel("Standard Error/GTE")
    plt.tight_layout()
    sns.despine()
    plt.savefig(file_path+stat+".png")
    
    stat = 'rmse_over_GTE'
    (total_stats_df
    .rename({'cr':"Customer-Side", 'lr':"Listing-Side",
             'tsr_est_naive':"TSR-Naive",
             'tsri_1.0':"TSRI-1",
             'tsri_2.0':"TSRI-2"}, level=0)
    .unstack(level=0)[stat]
    .plot(kind='bar'))
    
    plt.yscale('log')
    plt.xlabel("Relative Demand $\lambda/\\tau$")
    plt.title("RMSE")
    plt.legend(loc=[1.02,0.5])
    plt.ylabel("RMSE/GTE")
    plt.tight_layout()
    sns.despine()
    plt.savefig(file_path+stat+".png")
    
    return


    

