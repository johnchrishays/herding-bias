import csv
import sys
import pandas as pd
import numpy as np

def process_csv(file_path, row_number, nfiles=300):
    if "sample" in file_path:
        total = np.zeros(10)
    else:
        total = np.zeros(31)
    for i in range(1,nfiles+1):
        stats = pd.read_csv(file_path + f"{i}" + ".csv")
        try:
            if "sample" in file_path:
                specified_row = np.array(stats.iloc[row_number,1:])
            else:
                specified_row = np.array(stats.iloc[row_number,])

        except StopIteration:
            print(f"Error: Row {row_number} does not exist in the CSV file.")
            return
        total = total + specified_row

    return total / nfiles

def print_tuples(specified_row):
    # Zip and print the results
    for header, value in enumerate(specified_row):
        print(f"({header}, {value})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file_path> <row_number>")
    else:
        csv_file_path = sys.argv[1]
        which = sys.argv[2]
        if which == "t":
            row_number = 0
            stats = process_csv(csv_file_path, row_number)
            print_tuples(stats)
        elif which == "c":
            row_number = 1
            stats = process_csv(csv_file_path, row_number)
            print_tuples(stats)
        elif "t_sd" in which:
            row_number = 4
            specified_row_sd = process_csv(csv_file_path, row_number)
            specified_row= process_csv(csv_file_path, 0)
            if  which == "t_sd_upper":
                print_tuples(specified_row + specified_row_sd)
            if  which == "t_sd_lower":
                print_tuples(specified_row - specified_row_sd)
        elif "c_sd" in which:
            row_number = 5
            specified_row_sd = process_csv(csv_file_path, row_number)
            specified_row= process_csv(csv_file_path, 1)
            if  which == "c_sd_upper":
                print_tuples(specified_row + specified_row_sd)
            if  which == "c_sd_lower":
                print_tuples(specified_row - specified_row_sd)
        elif which == "expt":
            row_number = 2
            stats = process_csv(csv_file_path, row_number)
            print_tuples(stats)
        elif which == "expc":
            row_number = 3
            specified_row = process_csv(csv_file_path, row_number)
            print_tuples(specified_row)
        elif "expt_sd" in which:
            row_number = 6
            specified_row_sd = process_csv(csv_file_path, row_number)
            specified_row= process_csv(csv_file_path, 2)
            if  which == "expt_sd_upper":
                print_tuples(specified_row + specified_row_sd)
            if  which == "expt_sd_lower":
                print_tuples(specified_row - specified_row_sd)
        elif "expc_sd" in which:
            row_number = 7
            specified_row_sd = process_csv(csv_file_path, row_number)
            specified_row= process_csv(csv_file_path, 3)
            if  which == "expc_sd_upper":
                print_tuples(specified_row + specified_row_sd)
            if  which == "expc_sd_lower":
                print_tuples(specified_row - specified_row_sd)

        elif which == "gate":
            stats_t = process_csv(csv_file_path, 0)
            stats_c = process_csv(csv_file_path, 1)
            print_tuples(stats_t - stats_c)

        elif which == "expgate":
            stats_t = process_csv(csv_file_path, 2)
            stats_c = process_csv(csv_file_path, 3)
            print_tuples(stats_t - stats_c)
        elif which == "sample_cr":
            stats_01 = process_csv(csv_file_path, 0)
            stats_1 = process_csv(csv_file_path, 1)
            stats_10 = process_csv(csv_file_path, 2)
            print(stats_01[4])
            print(stats_1[4])
            print(stats_10[4])
        elif which == "sample_lr":
            stats_01 = process_csv(csv_file_path, 6)
            stats_1 = process_csv(csv_file_path, 7)
            stats_10 = process_csv(csv_file_path, 8)
            print(stats_01[4])
            print(stats_1[4])
            print(stats_10[4])
        elif which == "causalband":
            lower_stats_01 = process_csv(csv_file_path, 33)
            lower_stats_1 = process_csv(csv_file_path, 34)
            lower_stats_10 = process_csv(csv_file_path, 35)
            upper_stats_01 = process_csv(csv_file_path, 36)
            upper_stats_1 = process_csv(csv_file_path, 37)
            upper_stats_10 = process_csv(csv_file_path, 38)
            print("Band")
            print(f"({lower_stats_01[1]}, 1) ({upper_stats_01[1]}, 1)")
            print(f"({lower_stats_1[1]}, 2) ({upper_stats_1[1]}, 2)")
            print(f"({lower_stats_10[1]}, 3) ({upper_stats_10[1]}, 3)")
            print("GATE")
            print(f"({lower_stats_01[2]}, 1)")
            print(f"({lower_stats_1[2]}, 2)")
            print(f"({lower_stats_10[2]}, 3)")
        elif which == "causalband2":
            lower_stats_01 = process_csv(csv_file_path, 21)
            lower_stats_1 = process_csv(csv_file_path, 22)
            lower_stats_10 = process_csv(csv_file_path, 23)
            upper_stats_01 = process_csv(csv_file_path, 36)
            upper_stats_1 = process_csv(csv_file_path, 37)
            upper_stats_10 = process_csv(csv_file_path, 38)
            print("Band")
            print(f"({lower_stats_01[1]}, 1) ({upper_stats_01[1]}, 1)")
            print(f"({lower_stats_1[1]}, 2) ({upper_stats_1[1]}, 2)")
            print(f"({lower_stats_10[1]}, 3) ({upper_stats_10[1]}, 3)")
            print("GATE")
            print(f"({lower_stats_01[2]}, 1)")
            print(f"({lower_stats_1[2]}, 2)")
            print(f"({lower_stats_10[2]}, 3)")


