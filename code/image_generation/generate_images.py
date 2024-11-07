#from reading_shifts import extract_shifts as extract_shifts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

raw_data = pd.read_csv("../../data/datasets/qm9/picked_peaks/picked_peaks.csv")
shifts = raw_data.iloc[0]["shifts"]

# a function which is able to extract a shift from the format outputted by mestrenova into two lists of coupled carbon and protons
def extract_shifts(shifts):
    carbon_shifts, proton_shifts = shifts.split("1H")
    #splits carbon shifts into a list
    carbon_shifts = carbon_shifts.split("?")[1].split("\n")[0].split(", ")
    carbon_shifts[0] = carbon_shifts[0].rstrip(" ")
    #removes spaces from data
    carbon_shifts = [shift.strip() for shift in carbon_shifts]
    carbon_shifts = [float(shift) for shift in carbon_shifts] 

    #splits proton shifts into a list f
    proton_shifts = proton_shifts.split("?")[1].split("\n")[0].split(", ")
    # removes unwanted spaces from data points
    proton_shifts = [shift.strip() for shift in proton_shifts]
    #removes full stop from the end of the report 
    proton_shifts[-1] = proton_shifts[-1].rstrip('.')
    proton_shifts = [float(shift) for shift in proton_shifts] 
    shift_dict = {"proton":proton_shifts, "carbon":carbon_shifts}
    return shift_dict

def generate_hsqc(hsqc_peaks, common_name, functional_group_name, functional_group_present):
    functional_group_present = bool(functional_group_present)
    #scatter plots the proton and carbon hsqc shifts
    fig, ax = plt.subplots( nrows=1, ncols=1)
    ax.scatter(hsqc_peaks["proton"], hsqc_peaks["carbon"], c = "blue")

    #sets limits and removes axes
    plt.xlim(-2, 12)
    plt.ylim(-2, 250)
    plt.axis('off')

    #inverts axes to emulate the backwards nature of NMR plots
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    if functional_group_present:
        #saves the figure against its name
        if not os.path.exists(f"../../data/datasets/qm9/{functional_group_name}_images/{functional_group_name}"):
            os.makedirs(f"../../data/datasets/qm9/{functional_group_name}_images/{functional_group_name}")
        fig.savefig(f"../../data/datasets/qm9/{functional_group_name}_images/{functional_group_name}/{common_name}HSQC.png")
    else:
        if not os.path.exists(f"../../data/datasets/qm9/{functional_group_name}_images/non_{functional_group_name}"):
            os.makedirs(f"../../data/datasets/qm9/{functional_group_name}_images/non_{functional_group_name}")
        #saves the figure against its name
        fig.savefig(f"../../data/datasets/qm9/{functional_group_name}_images/non_{functional_group_name}/{common_name}HSQC.png")
    plt.close("all")

def main(functional_group_name = "ether"):
    for index, spectrum in raw_data.iterrows():
        #checks that the spectrum is not nan and then extracts the peaks
        print(spectrum)
        if pd.notna(spectrum["shifts"]):
            #extract the hsqc peaks from the spectrum 
            hsqc_peaks = extract_shifts(spectrum["shifts"])
            common_name = spectrum["common_name"]
            functional_group_present = spectrum[functional_group_name]
            
            #use the peaks to generate a spectral image
            generate_hsqc(hsqc_peaks, common_name, functional_group_name, functional_group_present)


main()



        

