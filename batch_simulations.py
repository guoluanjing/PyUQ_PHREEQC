import os
import errno
import shutil
import subprocess
import time

import settings
import numpy as np


def run_simulations():
    start_time = time.time()

    base_dir = settings.basecase_dir
    base_inputfile = base_dir + "Hg-H2O-Cl-HS.phrq"
    base_dbsfile = base_dir + "Phreeqc-scb-lg.dat"

    newdir_base = settings.dir_base
    new_dir = settings.new_dir
    # new_dir = []  # List of directories for batch runs
    new_inputfile = []  # List of input file paths for batch runs
    new_dbsfile = []  # List of database file paths for batch runs

    lb = settings.lb
    ub = settings.ub
    inc = settings.inc

    i = 0  # index for file directories

    ph_range = np.arange(lb, ub, inc)
    ph_id = ['{:.1f}'.format(j) for j in ph_range]
    # ph_float = [float(j) for j in ph_id]

    # for ph in ph_float:
    for ph in ph_id:

        # pH_str = "pH " + str(ph)
        # new_dir.append(newdir_base + "ph" + str(ph) + "/")
        new_dir.append(newdir_base + "ph" + ph + "/")
        new_inputfile.append(new_dir[i] + "Hg-H2O-Cl-HS.phrq")
        file_path = new_inputfile[i]
        directory = os.path.dirname(file_path)
        # Ensure that an error message will display even if
        # it is for reasons other than the directory already exists
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        i += 1

    # print(new_dir)

    inputfile_f_in = open(base_inputfile).readlines()
    dbsfile_f_in = open(base_dbsfile).readlines()
    # print(dbsfile_f_in)

    inputfile_f_out = []
    dbsfile_f_out = []

    logk_str = []

    logk = settings.logk_samples

    j = 0  # index for file directories

    total_prepare_dbs_time = 0
    simulation_time = 0

    # Loop over directories defined by the pH value
    # for ph in ph_float:
    for ph in ph_id:
        # pH_str = "pH " + str(ph) + "\r\n"
        # pH_str = "pH " + ph + "\r\n"
        pH_str = "Fix_H+   -" + ph + '   NaOH    10.0\r\n'

        for i in inputfile_f_in:
            if i == '    Fix_H+   -7.0   NaOH    10.0\r\n':
                i = pH_str
            # if i == "pH 7\r\n":
            #     i = pH_str
            inputfile_f_out.append(i)

        # Create the input files and place them in their
        # corresponding directories
        open(new_inputfile[j], 'w').writelines(inputfile_f_out)

        #   database_file = base_dir + "Phreeqc-scb.dat"
        shutil.copy2(base_dbsfile, new_dir[j])
        # print(new_dir[j])

        # Loop over sets of sampled log values
        prepare_dbs_time = time.time()

        # sp#= [ 0,    1,      2,     3,    4,    5,      6,      7,     8,      9,     10,    11,   12,  13
        # sp = [HgOH,Hg(OH)2,Hg(OH)3,HgCl+,HgCl2,HgCl3-,HgCl4-2,HgOHCl,Hg(SH)2,HgS2H-,HgS2-2,(HgSH)+,HgS,HgS(s)]
        for m in xrange(len(logk)):  # number of sample sets
            for n in xrange(len(logk[0])):  # number of variables being sampled in each set
                # print(logk[i,j])
                logk_str.append("     log_k " + str(logk[m, n]) + "\r\n")
            # print(logk_str)
            new_dbsfile.append(new_dir[j] + "Phreeqc-scb-" + str(m) + ".dat")
            # print(new_dbsfile)
            for phrase in dbsfile_f_in:
                if phrase == "     log_k -3.4\t\t#modified by LG 9/1/2017\r\n":
                    phrase = logk_str[0]
                elif phrase == "     log_k -5.98\t\r\n":
                    phrase = logk_str[1]
                elif phrase == "     log_k -21.1\t#modified by LG 9/1/2017\r\n":
                    phrase = logk_str[2]
                elif phrase == "     log_k 7.31\t\r\n":
                    phrase = logk_str[3]
                elif phrase == "     log_k 14\t\r\n":
                    phrase = logk_str[4]
                elif phrase == "     log_k 14.925\t\r\n":
                    phrase = logk_str[5]
                elif phrase == "     log_k 15.535\t\r\n":
                    phrase = logk_str[6]
                elif phrase == "     log_k 4.27\t\t#modified by LG 9/1/2017\r\n":
                    phrase = logk_str[7]
                # if phrase == "     log_k 39.1\t\t\t#modified by LG 9/1/2017\r\n":
                # elif phrase == "     log_k 39.1\t\t\t#modified by LG 9/1/2017\r\n":
                #     phrase = logk_str[8]
                # if phrase == "     log_k 32.5\t\t\t#modified by LG 9/1/2017\r\n":
                elif phrase == "     log_k 32.5\t\t\t#modified by LG 9/1/2017\r\n":
                    phrase = logk_str[9]
                # if phrase == "     log_k 23.2\t\t\t#modified by LG 9/1/2017\r\n":
                # elif phrase == "     log_k 23.2\t\t\t#modified by LG 9/1/2017\r\n":
                #     phrase = logk_str[10]
                # elif phrase == "     log_k 20.01\t\t#modified by LG 03/29/2018\r\n":
                #     phrase = logk_str[11]
                # elif phrase == "     log_k 28.6\t\r\n":
                # if phrase == "     log_k 16.4\t\r\n":
                # elif phrase == "     log_k 16.4\t\r\n":
                #     phrase = logk_str[12]
                # if phrase == "\tlog_k -36.8 #modified by LG 05/31/2018\r\n":
                # elif phrase == "\tlog_k -36.8 #modified by LG 05/31/2018\r\n":
                #     phrase = logk_str[13]

                dbsfile_f_out.append(phrase)
            del logk_str[:]  # empty the list before taking in the next sample set
            open(new_dbsfile[m], 'w').writelines(dbsfile_f_out)

            del_time = time.time() - prepare_dbs_time
            total_prepare_dbs_time += del_time

            # Running the batch simulations
            batch_start_time = time.time()
            cmd = ["phreeqc", new_inputfile[j], new_dir[j] + "Hg-H2O-Cl-HS-" + str(m) + ".out", new_dbsfile[m]]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            for line in process.stdout:
                print(line)
            simulation_time += time.time() - batch_start_time

            del dbsfile_f_out[:]

        del new_dbsfile[:]
        del inputfile_f_out[:]

        j += 1

    print("Database file preparation time = %.2f seconds." % total_prepare_dbs_time)
    print("Simulation time = %.2f seconds." % simulation_time)
    print("Total batch simulation time = %.2f seconds." % (time.time() - start_time))
