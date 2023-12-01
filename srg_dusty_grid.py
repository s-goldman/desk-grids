# recompiling: gfortran dustyV2.f -std=legacy -o dusty
# make executable: chmod +x dusty
# command for running on science cluster: nohup nice -19 python srg_dusty_grid.py > log.txt &
# if only getting .inp files recompile in fortran

import os
import json
import copy
import time
import shutil
import datetime
import subprocess
import itertools
from tqdm import tqdm
import numpy as np
from glob import glob
from multiprocessing import cpu_count
from astropy.table import Table, Column, vstack, hstack, unique


def remove_previous_run(grid_name):
    """removes previous run of dusty grid if grid_name in the file name

    Parameters
    ----------
    grid_name : str
        str associated with previous run to remove
    """
    if os.path.exists(grid_name) and os.path.isdir(grid_name):
        shutil.rmtree(grid_name)
    for item in glob(grid_name + "*"):
        os.remove(item)
    os.mkdir(grid_name)


def set_up_input_files(
    config,
):
    """Creates input files for running dusty using srg_dust_grid.json
    file and setting it as the dictionary "config".

    Parameters
    ----------
    config : dict
        dictionary of parameters from srg_dust_grid.json file.
    """
    # Alters the inputs for testing the code in test mode.
    if config["test"] == True:
        config["grid_name"] = "test-mix"
        config["effective_temp_grid"] = [2600]
        config["inner_dust_temp_grid"] = [600]
        config["grain_type_a_grid"] = [0, 24]
        config["grain_type_b_grid"] = [0]
        config["grain_type_d_grid"] = [0]
        config["tau_number"] = 2
        config["tau_max"] = 0.01

    # Sets column names for dust fraction columns
    grain_a_filename = config["grain_a_filename"] + "_frac"
    grain_b_filename = config["grain_b_filename"] + "_frac"
    grain_c_filename = config["grain_c_filename"] + "_frac"
    grain_d_filename = config["grain_d_filename"] + "_frac"

    all_files = []
    counter = 0

    # Columns to be used for matrix generation
    independent_vars = [
        config["effective_temp_grid"],
        config["inner_dust_temp_grid"],
        config["grain_type_a_grid"],
        config["grain_type_b_grid"],
        config["grain_type_d_grid"],
    ]

    # Input table matrix creation
    param_array = Table(
        np.array(list(itertools.product(*independent_vars))),
        names=(
            "teff",
            "tinner",
            grain_a_filename,
            grain_b_filename,
            grain_d_filename,
        ),
    )

    # Figure out remaining fraction of dust
    param_array[grain_c_filename] = (
        100
        - param_array[grain_a_filename]
        - param_array[grain_b_filename]
        - param_array[grain_d_filename]
    )

    param_array.add_column(
        Column([config["grid_name"]] * len(param_array)), name="grid_name", index=0
    )
    param_array.add_column(Column(np.arange(0, len(param_array)), name="grid_idx"))

    # remove rows where third dust component fraction is less than zero
    param_array = param_array[param_array[grain_c_filename] >= 0]

    for item in param_array:
        # if item["teff"] == 3400:
        #     teff_w_g = str(item["teff"]) + "_g-"
        #     z = "0.25"
        # else:
        #     teff_w_g = str(item["teff"]) + "_g+"
        #     z = "0.00"

        filename = (
            str(item["grid_name"])
            + "_"
            + str(item["teff"])
            + "_"
            + str(item["tinner"])
            + "_"
            + grain_a_filename
            + "-"
            + str(item[grain_a_filename])
            + "_"
            + grain_b_filename
            + "-"
            + str(item[grain_b_filename])
            + "_"
            + grain_c_filename
            + "-"
            + str(item[grain_c_filename])
            + "_"
            + grain_d_filename
            + "-"
            + str(item[grain_d_filename])
            + "_"
            + "grid"
            + "-"
            + str(item["grid_idx"])
        )

        all_files.append(filename)
        open(filename + ".inp", "w").write(
            open("template_dusty.inp", "r")
            .read()
            .format(
                # teff=teff_w_g, # change for aringer (currently not working)
                teff=item["teff"],
                tinner=item["tinner"],
                grain_a_nk_filename=config["grain_a_filename"] + ".nk",
                grain_b_nk_filename=config["grain_b_filename"] + ".nk",
                grain_c_nk_filename=config["grain_c_filename"] + ".nk",
                grain_d_nk_filename=config["grain_d_filename"] + ".nk",
                a=item[grain_a_filename],
                b=item[grain_b_filename],
                c=item[grain_c_filename],
                d=item[grain_d_filename],
                # z_val=z,
                tau_number=config["tau_number"],
                tau_max=config["tau_max"],
            )
        )
        counter += 1
    np.savetxt("dusty.inp", all_files, fmt="%s", delimiter="\n")
    param_array.write(config["grid_name"] + "_grid_idx.csv", overwrite=True)


def batch_dusty(config):
    """Runs available {grid_name}_*.inp files with dusty in parallel using 
    inputs from config (srg_dusty_grid.json)

    Parameters
    ----------
    config : dict
        dictionary of parameters from srg_dust_grid.json file.

    """
    file_names = glob(config["grid_name"] + "_*.inp")

    def print_time():
        end = time.time()
        total_time = (end - start) / 60 / 60
        if total_time > 1:
            print("\t\ttotal time = " + str(round(total_time)) + " hours")
        else:
            print("\t\ttotal time = " + str(round(total_time * 60)) + " minutes")

    # inialize variables
    running = {}
    times = []
    starts = {}
    if isinstance(config["n_cores"], int):
        n_cores = config["n_cores"]
    else:
        n_cores = cpu_count() - 1
    print(f"\nUsing a maximum of {n_cores} cores")
    start = time.time()

    # get files
    file_names_without_extension = [x.replace(".inp", "") for x in file_names]

    # get input filenames
    if len(file_names_without_extension) == 0:
        raise Exception("No input files : (")
    elif len(file_names_without_extension) < 1:
        run_file_list = file_names_without_extension[0]
    else:
        run_file_list = file_names_without_extension

    for source_index, source in enumerate(run_file_list):
        # adds source to dusty.inp (input file for individual dusty runs)
        with open("dusty.inp", "w") as f:
            f.write(str(source) + "\n")
            f.close()
            starts[source_index] = time.time()
            print(
                "\t running: "
                + source
                + "\t"
                + str(source_index + 1)
                + " / "
                + str(len(run_file_list))
                + "\t"
                + str(datetime.datetime.now().time())[:-7]
            )

            # dusty run
            process = subprocess.Popen(
                ["nohup nice -19 ./dusty"],
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            running[source_index] = process
            time.sleep(5)

            # if open cores available run another file
            while (len(running) >= n_cores) and (source_index < len(run_file_list) - 1):
                for n_process, process in list(running.items()):
                    process.poll()
                    if process.returncode is not None:
                        print("\t\tfinished: " + source)
                        times.append((time.time() - starts[source_index]) / (60 * 60))
                        process.communicate()
                        del starts[n_process]
                        del running[n_process]
                time.sleep(30)  # checks if process finished every N seconds

            # gets and prints time left
            if len(times) > 0:
                print_time()

    # waits for last process
    while (process.poll() == None) & (len(times) < len(run_file_list)):
        time.sleep(5)  # checks if all process finished every 5 seconds

    print("Complete")
    print_time()


def read_output_file(output_file_name):
    """reads .txt file from remove function (trimmed ".out" file) and
    and removes unnessecary columns.

    Parameters
    ----------
    output_file_name : str
        ".txt" filename.

    Returns
    -------
    output_table
        4 column table of number, odep, mdot, and vexp.

    """
    # reads important columns of output file
    output_table = Table(
        np.genfromtxt(
            output_file_name,
            dtype=[
                ("number", np.int16),
                ("odep", np.float64),
                ("c", np.float64),
                ("d", np.float64),
                ("e", np.float64),
                ("f", np.float64),
                ("g", np.float64),
                ("h", np.float64),
                ("mdot", np.float64),
                ("vexp", np.float64),
                ("i", np.float64),
            ],
            delimiter="",
        )
    )
    output_table.remove_columns(["c", "d", "e", "f", "g", "h", "i"])
    return output_table


def read_spectra(subset_spectra_filenames):
    """Reads in list of spectra files and returns the combined table.

    Parameters
    ----------
    subset_spectra_filenames : array
        names of spectra files.

    Returns
    -------
    output_spectra: astropy table
        Table of (arrays of) wavelength_um and flux_wm2.
    available_numbers: array
        The model numbers found in the used spectra files
    bad_indices: array
        The combined list of missing or bad spectra files
    """
    # initialize lists
    dusty_spectra = []
    empty_index = []
    available_numbers = []

    # append good spectral files
    for i, item in enumerate(subset_spectra_filenames):
        if os.stat(item).st_size != 0:
            a = np.loadtxt(item, usecols=[0, 1], unpack=True)
            dusty_spectra.append(a)
            available_numbers.append(int(item.split(".s")[-1]))
        else:
            empty_index.append(i)

    # get missing indices. some models may be missing in grid?
    spectra_numbers = [int(x.split(".s")[-1]) for x in subset_spectra_filenames]
    missing_index = [x - 1 for x in np.arange(1, 101) if x not in spectra_numbers]

    # return good table
    output_spectra = Table(np.array(dusty_spectra))
    bad_indices = empty_index + missing_index
    return output_spectra, available_numbers, bad_indices


def remove(indiv_grid_name_file):
    """Removes everything but dusty output results.

    Parameters
    ----------
    indiv_grid_name_file : str
        dusty .out filename.
    """
    output_table = []
    results_trigger = False
    results_end = False

    for n, row in enumerate(open(indiv_grid_name_file)):
        # results start
        if "   1 " in row[0:5]:
            results_trigger = True
        # warning within results
        if (
            results_trigger is True
            and results_end is False
            and (
                not (
                    ("Please" in row)
                    or ("There" in row)
                    or ("=" in row)
                    or (" (1)" in row)
                    or ("*" in row)
                )
            )
        ):
            output_table.append(row)
        # end of results, parse file
        if results_trigger is True and (
            (" 100" in row[0:4]) or (" ***" in row) or (" (1)" in row)
        ):
            results_end = True
    f = open(indiv_grid_name_file.replace(".out", ".txt"), "w")
    for line in output_table:
        f.write(line)
    f.close


def dusty_to_grid(config):
    """combining resulting files (.s* and *.out) into two HDF5 files.

    Returns
    -------
    outputs_file.hdf5, spectra_file.hdf5

    """

    full_spectra = Table()

    # remove warning
    out_file_list = glob(config["grid_name"] + "*.out")
    for item in tqdm(out_file_list):
        remove(item)

    # spectra
    output_files = glob(config["grid_name"] + "*.txt*")  # not in idx order!
    output_files_ind = [int(x.split("grid-")[1][:-4]) for x in output_files]
    output_files_sorted = np.array(output_files)[np.argsort(output_files_ind)]

    grid_idx_array = Table.read(config["grid_name"] + "_grid_idx.csv")

    for j, grid_idx_filename in tqdm(enumerate(output_files_sorted)):
        grid_idx = grid_idx_filename.split("grid-")[-1][:-4]

        # spectra
        subset_spectra_filenames = glob("*_grid-" + grid_idx + ".s*")
        subset_spectra_filenames.sort()
        subset_spectra, valid_numbers, bad_indices = read_spectra(
            subset_spectra_filenames
        )

        # defined outputs
        set_outputs = grid_idx_array[grid_idx_array["grid_idx"] == int(grid_idx)]
        set_outputs_large = Table(np.repeat(set_outputs, len(subset_spectra)))

        # resulting outputs
        subset_single_output_file = read_output_file(grid_idx_filename)
        subset_outputs = hstack((set_outputs_large, subset_single_output_file))

        ## for runs with warnings and 100 entries
        # if len(bad_indices) > 0:
        #     subset_single_output_file.remove_rows(bad_indices)

        # construct output table
        if j == 0:
            full_outputs = copy.deepcopy(subset_outputs)
            full_spectra = copy.deepcopy(subset_spectra)

        else:
            # append subset
            full_spectra = vstack((full_spectra, subset_spectra))
            full_outputs = vstack((full_outputs, subset_outputs))

        # #### TESTS
        # make sure spectra and outputs numbers match
        assert np.all(np.array(subset_outputs["number"]) == np.array(valid_numbers)), (
            "Misaligned models and outputs: "
            + str(list(valid_numbers))
            + " vs. "
            + str(list(subset_outputs["number"]))
        )

        # makes sure spectra and outputs are the same size
        assert len(full_outputs) == len(
            full_spectra
        ), "Size of spectra and outputs are different"

        # make sure rows are unique
        assert len(full_outputs) == len(
            unique(full_outputs)
        ), "Duplicated rows in outputs"

    if len(full_outputs) > 0 and len(full_spectra) > 0:
        print(f"Total number of models: {len(full_outputs)}")
    else:
        raise Exception(
            "No models found. Check your input files, or recompile dusty:\n\t gfortran dustyV2.f -std=legacy -o dusty"
        )

    if np.all(full_outputs[config["grain_b_filename"] + "_frac"] == 0):
        full_outputs.remove_column(config["grain_b_filename"] + "_frac")
    if np.all(full_outputs[config["grain_d_filename"] + "_frac"] == 0):
        full_outputs.remove_column(config["grain_d_filename"] + "_frac")

    for file in glob(config["grid_name"] + "*"):
        shutil.move(file, config["grid_name"])

    # save model spectra and outputs
    full_spectra.write(config["grid_name"] + "_models.hdf5", overwrite=True)
    full_outputs.write(config["grid_name"] + "_outputs.hdf5", overwrite=True)
    # return full_outputs, full_spectra


def run_dusty():
    # program starts ---------------------------------------------------------------
    with open("srg_dust_grid.json", "r") as fp:
        config = json.load(fp)

    # remove previous run
    if config["remove_previous"] == True:
        remove_previous_run(config["grid_name"])

    # input dusty files
    set_up_input_files(config)

    # run dusty
    batch_dusty(config); time.sleep(300)

    # collate results
    dusty_to_grid(config)


if __name__ == "__main__":
    run_dusty()
