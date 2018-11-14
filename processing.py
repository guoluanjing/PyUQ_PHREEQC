import os
import errno
import warnings
#import subprocess
#from pyDOE import *
import matplotlib as mpl
mpl.use('PDF')
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import FormatStrFormatter
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit
import seaborn as sns
from math import log10
import csv
import time

# mpl.rcParams['figure.figsize'] = (20.0, 15.0)
# mpl.rcParams['axes.titlepad'] = 20
# mpl.rcParams['axes.titlesize'] = 44
# mpl.rcParams['axes.labelsize'] = 40
# mpl.rcParams['axes.labelpad'] = 14
# mpl.rcParams['axes.grid'] = True
# mpl.rcParams['lines.linewidth'] = 12
# mpl.rcParams['lines.markersize'] = 12
# mpl.rcParams['xtick.labelsize'] = 40
# mpl.rcParams['ytick.labelsize'] = 40
# mpl.rcParams['xtick.major.size'] = 10
# mpl.rcParams['ytick.major.size'] = 10
# mpl.rcParams['legend.fontsize'] = 40
# mpl.rcParams['grid.color'] = '#E6E6E6'
# mpl.rcParams['grid.linestyle'] = '-'
# mpl.rcParams['grid.linewidth'] = 0.5
# mpl.rcParams['boxplot.boxprops.linewidth'] = 8.0
# mpl.rcParams['boxplot.whiskerprops.linewidth'] = 8.0
# mpl.rcParams['boxplot.capprops.linewidth'] = 8.0
# mpl.rcParams['boxplot.medianprops.linewidth'] = 10.0
# mpl.rcParams.update({'figure.autolayout': True})

color_list = []
color_list += ['#8D7B14', '#2C694F', '#774576']
pal = sns.color_palette("husl", 7)
color_list += pal.as_hex()
color_list.append('#666666')

mpl.rcParams['axes.prop_cycle'] = cycler('color', color_list)
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

mpl.rcParams['figure.figsize'] = (20.0, 15.0)
mpl.rcParams['axes.titlepad'] = 20
mpl.rcParams['axes.titlesize'] = 64
mpl.rcParams['axes.labelsize'] = 60
mpl.rcParams['axes.labelpad'] = 14
mpl.rcParams['axes.edgecolor'] = 'gray'
mpl.rcParams['axes.linewidth'] = 8
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['axes.grid'] = True

mpl.rcParams['lines.linewidth'] = 18
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.dashed_pattern'] = (2, 0.5)
mpl.rcParams['hatch.linewidth'] = 4.0
mpl.rcParams['xtick.labelsize'] = 60
mpl.rcParams['ytick.labelsize'] = 60
mpl.rcParams['xtick.color'] = 'gray'
mpl.rcParams['ytick.color'] = 'gray'
mpl.rcParams['xtick.major.size'] = 20
mpl.rcParams['ytick.major.size'] = 20
mpl.rcParams['xtick.major.width'] = 10
mpl.rcParams['ytick.major.width'] = 10
mpl.rcParams['xtick.minor.size'] = 15
mpl.rcParams['ytick.minor.size'] = 15
mpl.rcParams['xtick.minor.width'] = 10
mpl.rcParams['ytick.minor.width'] = 10
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['legend.fontsize'] = 46
mpl.rcParams['grid.color'] = '#E6E6E6'
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['grid.linewidth'] = 3.0
mpl.rcParams['boxplot.boxprops.linewidth'] = 15
mpl.rcParams['boxplot.whiskerprops.linewidth'] = 15
mpl.rcParams['boxplot.capprops.linewidth'] = 10.0
mpl.rcParams['boxplot.medianprops.linewidth'] = 15.0
mpl.rcParams.update({'figure.autolayout': True})


def process():
    import settings

    starting_time = time.time()


    '''
    ===============
    Data Processing
    ===============
    
    Collecting data from output files, plotting, and data analysis 
    '''

    # Collect the species concentration data from output files
    # sp_names = ["HgCl+", "HgCl2", "HgCl3-", "HgCl4-2", "HgOHCl", "HgOH+", "Hg(OH)2", "Hg(OH)3-", "HgS", "HgHS2-", "Hg(HS)2",
    # "HgS2-2", "HgHS+", "Hg+2"]
    # sp_names = ["HgS", "HgHS2-", "Hg(HS)2"]
    # plotting_sp_names = [r'${HgS}^{0}$', r'${HgHS_2}^-$', r'$Hg(HS)_2$']
    sp_names = ["HgHS2-", "Hg(HS)2", "HgS2-2"]
    plotting_sp_names = [r'$\mathregular{{HgS_2H}^-}$', r'$\mathregular{Hg(SH)_2}$', r'$\mathregular{{HgS_2}^{2-}}$']
    num_sp = len(sp_names)
    # print(num_sp)

    c_sp_0 = []
    c_sp_1 = []
    c_sp_2 = []
    c_sp_3 = []

    logk = settings.logk_samples

    dir_base = settings.dir_base
    base_dir = settings.var_ph_dirbase
    newdir_base = settings.dir_base
    new_dir = settings.new_dir
    ph_list = []

    lb = settings.lb
    ub = settings.ub
    inc = settings.inc

    # new_dir = settings.new_dir
    ph_range = np.arange(lb, ub, inc)
    ph_id = ['{:.1f}'.format(j) for j in ph_range]
    # ph_float = [float(j) for j in ph_id]

    num_dir = 0
    # for ph in ph_float:
    for ph in ph_id:

        # pH_str = "pH " + str(ph)
        # new_dir.append(newdir_base + "ph" + str(ph) + "/")
        new_dir.append(newdir_base + "ph" + ph + "/")
        directory = os.path.dirname(new_dir[num_dir])
        # Ensure that an error message will display even if
        # it is for reasons other than the directory already exists
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        num_dir += 1

    # print(new_dir)
    k = 0

    assemble_p_ph = []
    assemble_p_x = []
    assemble_p = []

    assemble_1q = np.empty((0, num_sp))
    assemble_2q = np.empty((0, num_sp))
    assemble_3q = np.empty((0, num_sp))
    # print(assemble_1q.shape)

    iq_skewness = np.empty((0, num_sp))
    rel_uncertainty = np.empty((0, num_sp))
    dist_skewness = np.empty((0, num_sp))
    # print(rel_uncertainty.shape)
    stdvs = np.empty((0, num_sp))

    dominant_sps = []
    dominance_probs = []

    ### For Spearman test of ru and skew correlation ###
    rho_all = []
    p_all = []

    stack_box_id = 0

    for ph in ph_id:
        ph_list.append(ph)
        print('*************** pH = ' + ph + ' ***************')

        if ph[-1] == '0':
            assemble_p_ph.append(ph)

        ######################################################
        # Collecting the basecase species concentration data #
        ######################################################
        output_file = base_dir + "ph" + ph + "/" + "Hg-H2O-Cl-HS-ph" + ph + ".out"

        base_sp_c = []

        lookup = 'batch-reaction'
        file_as_one_string = []
        individual_word_string = []
        with open(output_file, 'r') as myfile2:
            for num, line in enumerate(myfile2, 1):
                if lookup in line:
                    starting_line = num
        with open(output_file, 'r') as myfile2:
            for num, line in enumerate(myfile2, 1):
                if num > starting_line:
                    file_as_one_string.append(line)
        # print(file_as_one_string)
        for entry in file_as_one_string:
            individual_word_string += entry.split()

        # with open(output_file, 'r') as myfile2:
        #     file_as_one_string = myfile2.read()
        # individual_word_string = file_as_one_string.split()

        for j in xrange(len(individual_word_string)):
            if individual_word_string[j] == sp_names[0]:
                base_sp_c.append(individual_word_string[j + 1])
                break
        for j in xrange(len(individual_word_string)):
            if individual_word_string[j] == sp_names[1]:
                base_sp_c.append(individual_word_string[j + 1])
                break
        for j in xrange(len(individual_word_string)):
            if individual_word_string[j] == sp_names[2]:
                base_sp_c.append(individual_word_string[j + 1])
                break
        # for j in xrange(len(individual_word_string)):
        #     if individual_word_string[j] == sp_names[3]:
        #         base_sp_c.append(individual_word_string[j + 1])
        #         break

        base_sp_c = [float(i) for i in base_sp_c]
        del file_as_one_string[:], individual_word_string[:]

        '''
        for i in xrange(len(logk)):  # number of sample sets
            # print(new_dir[k])
            datafile = new_dir[k] + "Hg-H2O-Cl-HS-" + str(i) + ".out"

            file_as_one_string = []
            individual_word_as_string = []
            with open(datafile, 'r') as myfile:
                for num, line in enumerate(myfile):
                    if lookup in line:
                        starting_line = num
            with open(datafile, 'r') as myfile:
                for num, line in enumerate(myfile):
                    if num > starting_line:
                        file_as_one_string.append(line)

            # print(file_as_one_string)
            for entry in file_as_one_string:
                individual_word_as_string += entry.split()

            # with open(datafile, 'r') as myfile:
            #     file_as_one_long_string = myfile.read()
            # individual_word_as_string = file_as_one_long_string.split()

            for j in xrange(len(individual_word_as_string)):
                if individual_word_as_string[j] == sp_names[0]:
                    c_sp_0.append(individual_word_as_string[j + 1])
                    break
            for j in xrange(len(individual_word_as_string)):
                if individual_word_as_string[j] == sp_names[1]:
                    c_sp_1.append(individual_word_as_string[j + 1])
                    break
            for j in xrange(len(individual_word_as_string)):
                if individual_word_as_string[j] == sp_names[2]:
                    c_sp_2.append(individual_word_as_string[j + 1])
                    break
            # for j in xrange(len(individual_word_as_string)):
            #     if individual_word_as_string[j] == sp_names[3]:
            #         c_sp_3.append(individual_word_as_string[j + 1])
            #         break
            del file_as_one_string[:], individual_word_as_string[:]

        # Analyze species concentration data
        # results_block = np.array([c_sp_0[:], c_sp_1[:], c_sp_2[:], c_sp_3[:]]).astype(np.float)  # converting string to float type
        results_block = np.array([c_sp_0[:], c_sp_1[:], c_sp_2[:]]).astype(np.float)  # converting string to float type
        # print(dir_base)
        # print(ph)
        np.savetxt(dir_base + 'results_block_pH'+ph+'.csv', results_block, delimiter=',')
        '''
        results_block = np.loadtxt(dir_base + 'results_block_pH'+ph+'.csv', delimiter=',')
        results_block = np.log10(results_block)

        max_idx = results_block.argmax(axis=0)
        max_sp, counts = np.unique(max_idx, return_counts=True)
        # print(max_sp, counts)

        dominant_sps.append(list(max_sp))
        dominance_probs.append([i / float(results_block.shape[1]) for i in list(counts)])

        # print(dominance_probs)

        # print(variable_names)
        # print(variable_names.shape)
        # print(results_block.shape)
        first_quantiles = np.percentile(results_block, 25, axis=1)
        median = np.percentile(results_block, 50, axis=1)
        third_quantiles = np.percentile(results_block, 75, axis=1)

        assemble_1q = np.append(assemble_1q, [first_quantiles], axis=0)
        assemble_2q = np.append(assemble_2q, [median], axis=0)
        assemble_3q = np.append(assemble_3q, [third_quantiles], axis=0)

        # '''
        ind_rel_uncertainty = (third_quantiles -  first_quantiles) / abs(median)
        # ind_rel_uncertainty = (third_quantiles -  first_quantiles) / median
        ind_iq_skewness = (third_quantiles - 2.0 * median + first_quantiles) / (third_quantiles - first_quantiles)
        # ind_skewness = st.skew(np.log10(results_block), axis=1)
        ind_skewness = st.skew(results_block, axis=1)
        # ind_skewness = np.abs(st.skew(results_block, axis=1))

        rel_uncertainty = np.append(rel_uncertainty, [ind_rel_uncertainty], axis=0)
        iq_skewness = np.append(iq_skewness, [ind_iq_skewness], axis=0)
        dist_skewness = np.append(dist_skewness, [ind_skewness], axis=0)
        stdvs = np.append(stdvs, [np.std(results_block, axis=1)], axis=0)
        # '''

        '''
        Finding distribution pdfs and plotting
        =======================================
    
        '''

        figure_path = dir_base + "ph" + ph + "/figures"

        # Ensure that an error message will display even if
        # it is for reasons other than the directory already exists
        try:
            os.makedirs(figure_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for i in xrange(num_sp):
            data = results_block[i, :]
            # data = np.log10(results_block[i, :])
            n = data.shape[0]
            skewness = st.skew(data)
            sigma = np.sqrt(6.0*(n-2)/(n+1)/(n+3))
            bin_number = int(round(1 + np.log2(n) + np.log2(1 + abs(skewness) / sigma)))
            print("Number of bins by Doane's rule for {} species: {}".format(sp_names[i], bin_number))
            # bin_number *= 2

            p, x = np.histogram(data, bins=bin_number, density=True)
            x_grid = (x + np.roll(x, -1))[:-1] / 2.0

            if ph[-1] == '0':
                assemble_p_x.append(x_grid)
                assemble_p.append(p)

            # Display
            fig, ax = plt.subplots()
            ax.plot(x_grid, p, lw=12)
            # ax.set_xscale('log')
            ax.axvline(log10(base_sp_c[i]), color=colors[-1], linewidth=10, label='Mean logK Only')
            ax.legend(frameon=False)
            ax.set_title('log10 '+plotting_sp_names[i] + ' concentration distribution')
            ax.set_xlabel(u'log10(Concentration), mol/kgw')
            ax.set_ylabel('Probability Density')

            plt.savefig(new_dir[k] + "figures/" + "distribution_" + sp_names[i])

            #######################################
            ### Plotting Raw Concentration Data ###
            #######################################
            fig, ax = plt.subplots()
            # ax.semilogy(xrange(results_block.shape[1]), results_block[i], 'o', color=colors[6])
            ax.plot(xrange(results_block.shape[1]), results_block[i], 'o', color=colors[6])
            ax.legend(frameon=False)
            ax.set_title(plotting_sp_names[i] + ' concentrations')
            ax.set_xlabel('Sample Number')
            # ax.set_ylabel(u'log10(Concentration), mol/kgw')
            ax.set_ylabel(u'Concentration, mol/kgw')

            plt.savefig(new_dir[k] + "figures/" + "concentrations_" + sp_names[i] +"_T", transparent=True)

            ########################################
            ### Boxplots with and without fliers ###
            ########################################
            fig, ax = plt.subplots()
            ax.boxplot([results_block[i, :]],medianprops=dict(color=colors[0]),flierprops=dict(markersize=12))  # with fliers
            ax.set_xticklabels([plotting_sp_names[i]])
            # ax.set_yscale('log')
            ax.set_title(u'Box plot of ' + plotting_sp_names[i] + ' concentrations')
            ax.set_ylabel(u'log10(Concentration), mol/kgw')
            # ax.set_ylabel('Concentration, mol/kgw')
            # ax.axhline(base_sp_c[i], ls='--', linewidth=8.0, label='Mean logK Only', color=colors[4])
            ax.axhline(log10(base_sp_c[i]), ls='--', linewidth=8.0, label='Mean logK Only', color=colors[4])
            ax.legend(frameon=False)
            plt.savefig(new_dir[k] + "figures/" + "boxplot_" + sp_names[i])

            fig, ax = plt.subplots()
            ax.boxplot([results_block[i, :]], showfliers=False,medianprops=dict(color=colors[0]))  # without fliers
            ax.set_xticklabels([plotting_sp_names[i]])
            # ax.set_yscale('log')
            ax.set_title(u'Box plot of ' + plotting_sp_names[i] + ' concentrations')
            ax.set_ylabel(u'log10(Concentration), mol/kgw')
            ax.axhline(log10(base_sp_c[i]), ls='--', linewidth=8.0, label='Mean logK Only', color=colors[4])
            # ax.set_ylabel('Concentration, mol/kgw')
            # ax.axhline(base_sp_c[i], ls='--', linewidth=6.0, label='Mean logK Only', color=colors[4])
            ax.legend(frameon=False)
            plt.savefig(new_dir[k] + "figures/" + "boxplot_" + sp_names[i] + '_nf' +"_T", transparent=True)

        if ph[-1] == '0':
            stack_box_id += 1

        k += 1
        del base_sp_c[:], c_sp_0[:], c_sp_1[:], c_sp_2[:], c_sp_3[:]

    ph_array = np.array([float(j) for j in ph_list])

    np.savetxt(dir_base + 'ph_array.csv', ph_array, delimiter=',')

    with open(dir_base + 'assembled_pdf.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(assemble_p_x)
        wr.writerows(assemble_p)

    with open(dir_base + 'sp_dominance_probability.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(dominant_sps)
        wr.writerows(dominance_probs)

    np.savetxt(dir_base + 'rel_uncertainy_pH.csv', rel_uncertainty, delimiter=',')
    np.savetxt(dir_base + 'SpH_vs_pH.csv', stdvs, delimiter=',')
    np.savetxt(dir_base + 'iq_skewness_pH.csv', iq_skewness, delimiter=',')
    np.savetxt(dir_base + 'skewness_pH.csv', dist_skewness, delimiter=',')
    np.savetxt(dir_base + '1st_quartiles.csv', assemble_1q, delimiter=',')
    np.savetxt(dir_base + '2nd_quartiles.csv', assemble_2q, delimiter=',')
    np.savetxt(dir_base + '3rd_quartiles.csv', assemble_3q, delimiter=',')

    # rel_uncertainty = np.loadtxt(dir_base + 'rel_uncertainy_pH.csv', delimiter=',')
    # iq_skewness = np.loadtxt(dir_base + 'iq_skewness_pH.csv', delimiter=',')

    for i in xrange(len(sp_names)):
        rho, p = st.spearmanr(rel_uncertainty[:, i], dist_skewness[:, i])
        # rho, p = st.spearmanr(rel_uncertainty[:, i], assemble_2q[:, i])
        rho_all.append(rho)
        p_all.append(p)

    print("Spearman correlation test statistics for species {} are: {}".format(sp_names, rho_all))
    print("Spearman correlation test p-value for species {} are: {}".format(sp_names, p_all))

    dirbase_figure_path = dir_base + "/figures"

    # Ensure that an error message will display even if
    # it is for reasons other than the directory already exists
    try:
        os.makedirs(dirbase_figure_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    figs_p_plt =[]
    axes_p_plt =[]

    for i in xrange(num_sp):
        fig, ax = plt.subplots()
        axes_p_plt.append(ax)
        figs_p_plt.append(fig)

    # print(len(assemble_p))
    for i in xrange(len(assemble_p_ph)):
        for j in xrange(num_sp):
            ind = i*num_sp+j
            axes_p_plt[j].plot(assemble_p_x[ind], assemble_p[ind], lw=10, label='pH='+assemble_p_ph[i])
            # axes_p_plt[j].set_xscale('log')
            axes_p_plt[j].set_xlabel('log10(Concentration), mol/kgw')
            axes_p_plt[j].set_ylabel('Probability Density')
            axes_p_plt[j].set_title('Probability distributions of log10 ' + plotting_sp_names[j] + ' concentration')

    for j in xrange(num_sp):
        axes_p_plt[j].legend(bbox_to_anchor=(0.5, -0.2), loc=9, ncol=2,frameon=False)
        # axes_p_plt[j].legend(bbox_to_anchor=(0.98, 0.5), loc=6, frameon=False)
        figs_p_plt[j].savefig(dir_base + "figures/p-"+sp_names[j]+"_vs_pH_T", transparent=True)

    del figs_p_plt[:], axes_p_plt[:]
    for i in xrange(num_sp):
        fig, ax = plt.subplots()
        axes_p_plt.append(ax)
        figs_p_plt.append(fig)

    for i in xrange(len(assemble_p_ph)):
        for j in xrange(num_sp):
            ind = i*num_sp+j
            axes_p_plt[j].plot(assemble_p_x[ind], assemble_p[ind], lw=10)
            # axes_p_plt[j].set_xscale('log')
            axes_p_plt[j].set_xlabel('log10(Concentration), mol/kgw')
            axes_p_plt[j].set_ylabel('Probability Density')
            axes_p_plt[j].set_title('Probability distributions of log10 ' + plotting_sp_names[j] + ' concentration')

    for j in xrange(num_sp):
        # axes_p_plt[j].legend(bbox_to_anchor=(0.98, 0.5), loc=6, frameon=False)
        figs_p_plt[j].savefig(dir_base + "figures/p-"+sp_names[j]+"_vs_pH_T_nolegend", transparent=True)

    fig, ax = plt.subplots()
    # plt.figure(figsize=(25, 20))
    # no_mineral_2q = np.loadtxt("/Users/qng/Documents/python/new_cases/var_ph/13logk_hgs16.4/2nd_quartiles.csv",
    #                            delimiter=',')
    hatches = ['/', '-', '|']

    # for i in xrange(num_sp):
    #     ax.plot(ph_array, assemble_2q[:, i], ls='-', color=colors[i], label=plotting_sp_names[i])

    for i in xrange(num_sp):
        ax.plot(ph_array, assemble_2q[:, i], ls='-', color=colors[i], label=plotting_sp_names[i])
        # ax.plot(ph_array, assemble_2q[:, i], ls='-', color=colors[i], label=plotting_sp_names[i]+'_with_mineral')
        # ax.plot(ph_array, no_mineral_2q[:, i], ls='--', color=colors[i], label=plotting_sp_names[i]+'_no_mineral')
        ax.plot(ph_array, assemble_1q[:, i], ls='None', color=colors[i], label=None)
        ax.plot(ph_array, assemble_3q[:, i], ls='None', color=colors[i], label=None)
        ax.fill_between(ph_array, assemble_1q[:, i], assemble_3q[:, i], facecolor=colors[i], alpha=0.45,
                        hatch=hatches[i], edgecolor='w', lw=0)

    ax.set_xlabel('pH')
    ax.set_ylabel(u'log10 Concentration')
    # ax.set_ylabel(u'log10(Concentration), mol/kgw')
    # ax.set_ylabel('Concentration, mol/kgw')
    # plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    # ax.legend(frameon=False)
    ax.set_ylim(-19, -10)
    ax.set_yticks([-19, -16, -13, -10])
    ax.legend(loc='lower right', bbox_to_anchor=(0.98, -0.04),frameon=False)
    # fig.savefig(dirbase_figure_path + "/percentile_vs_pH" +"_T", transparent=True)
    fig.savefig(dirbase_figure_path + "/percentile_vs_pH")

    fig, ax = plt.subplots()

    for i in xrange(num_sp):
        ax.plot(ph_array, rel_uncertainty[:,i], color=colors[i], label=plotting_sp_names[i])

    ax.set_xlabel('pH')
    # ax.set_yticks([x for x in xrange(5,8)])
    ax.set_ylabel('Relative Uncertainty', multialignment='center')
    # ax.legend(loc="lower right", frameon=False)
    ax.legend(frameon=False)

    fig.savefig(dirbase_figure_path + "/rel_uncertainty_vs_pH" +"_T", transparent=True)

    fig, ax = plt.subplots()

    for i in xrange(num_sp):
        ax.plot(ph_array, stdvs[:,i], color=colors[i], label=plotting_sp_names[i])

    ax.set_xlabel('pH')
    # ax.set_yticks([x for x in xrange(5,8)])
    ax.set_ylabel('Standard Deviation', multialignment='center')
    # ax.legend(loc="upper left", frameon=False)
    ax.legend(loc="best", frameon=False)

    fig.savefig(dirbase_figure_path + "/S_vs_pH" +"_T", transparent=True)

    fig, ax = plt.subplots()
    # plt.figure(figsize=(25,20))
    for i in xrange(num_sp):
        ax.plot(ph_array, iq_skewness[:,i], color=colors[i], label=plotting_sp_names[i])

    ax.set_xlabel('pH')
    ax.set_ylabel('Interquartile Skewness', multialignment='center')
    # ax.set_yticklables(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(bbox_to_anchor=(0.95,0.45), frameon=False)

    fig.savefig(dirbase_figure_path + "/iq_skew_vs_pH" +"_T", transparent=True)

    fig, ax = plt.subplots()
    # plt.figure(figsize=(25,20))
    for i in xrange(num_sp):
        ax.plot(ph_array, dist_skewness[:,i], color=colors[i], label=plotting_sp_names[i])

    ax.set_xlabel('pH')
    ax.set_ylabel('Skewness', multialignment='center')
    # ax.set_yticklables(yticks)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.legend(loc='upper left', bbox_to_anchor=(0.15,0.65), frameon=False)
    # ax.legend(loc='center left', frameon=False)
    ax.legend(loc='best', frameon=False)

    fig.savefig(dirbase_figure_path + "/skew_vs_pH" +"_T", transparent=True)

    print("Data Processing time = %.f seconds." % (time.time() - starting_time))

