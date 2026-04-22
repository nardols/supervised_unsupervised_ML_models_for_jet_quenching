#----------------------------------------------
#
# inspiration: Jet substructure observables for jet quenching in Quark Gluon Plasma: a Machine Learning driven analysis - https://arxiv.org/abs/2304.07196
#
#----------------------------------------------

import subprocess
import os
import queue
from threading import Thread
import shutil
import time

#----------------------------------------------#
#                  FUNCTIONS                   #
#----------------------------------------------#

def safe_remove(path, retries=5, delay=1):
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"File {path} deleted.")
                break  
            else:
                break  
        except Exception as e:
            print(f"Attempt {attempt+1}: Could not delete {path}. Error: {e}")
            time.sleep(delay)
    else:
        print(f"Failed to delete {path} after {retries} attempts.")

#----------------------------------------------

def create_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content.strip())
    os.chmod(filepath, 0o644)  #permissions
    print(f"File {filepath} successfully created with default permissions (644).")

#----------------------------------------------

def execute_command(command):
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        print("Command executed successfully.")
        print("Output:", result.stdout)
        print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e)
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return e.stderr, e.stdout

#----------------------------------------------


def analyze_hepmc(hepmc_file, experiment_num, jet_type, background, MLtype):
    analysis_script = "/sampa/llimadas/Jewel2.4/analysis/doStructure"

    if background == 'nobkg':
        background_flag = 'nobkgsub'
    elif background == 'bkg':
        background_flag = 'bkgsub'
    else:
        raise ValueError("Invalid background. Use 'nobkg' or 'bkg'.")


    base_name = os.path.splitext(os.path.basename(hepmc_file))[0]
    output_root_name = f"{base_name}_{MLtype}_structure"
    output_root_path = os.path.join(os.getcwd(), output_root_name + f"_full_{background_flag}.root") 

    # 
    analysis_command = f"{analysis_script} --{jet_type} --{background_flag} {hepmc_file} {output_root_name}"
    execute_command(analysis_command)

    if not os.path.exists(output_root_path):
        raise FileNotFoundError(f"Arquivo ROOT {output_root_path} não foi criado.")

    final_path = "/sampa/llimadas/Jewel2.4/analysis/results_root/"
    move_command = f"mv {output_root_path} {final_path} "
    execute_command(move_command)

    print(f"Done.")
    return


#----------------------------------------------

def create_file_from_template(template_path, output_path, replacements):
    with open(template_path, 'r') as file:
        content = file.read()
    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", str(value))
    with open(output_path, 'w') as file:
        file.write(content)
    os.chmod(output_path, 0o644)


#----------------------------------------------#
#----------------------------------------------#
#         RUNNING jewel default Pb+Pb          #
#----------------------------------------------#
#----------------------------------------------#

def run_simulation(experiment_num, nevent, background, MLtype):
    file_path = '/sampa/archive/llimadas/temp/jewel/substructures/general/outputs_default/'
    os.makedirs(file_path, exist_ok=True)
    
    jewel_dir = "/sampa/leonardo/testing/jewel-2.4.0/"
    
    params_file = os.path.join(file_path, f"params_medium-default_{experiment_num}.dat")
    medium_params_file = os.path.join(file_path, f"medium-default_{experiment_num}.dat")

    create_file_from_template("/sampa/llimadas/general_parameters/par_medium_article-marco.dat", params_file, {
        "NEVENT": nevent,
        "EXPNUM": experiment_num,
        "PAR_MED_FILE": medium_params_file
    })

    create_file_from_template("/sampa/llimadas/general_parameters/medium-params_article-marco.dat", medium_params_file, {
        "EXPNUM": experiment_num 
    })

    jewel_executable = os.path.join(jewel_dir, "jewel-2.4.0-simple")
    if not os.path.isfile(jewel_executable) or not os.access(jewel_executable, os.X_OK):
        print(f"Error: Executable {jewel_executable} is invalid.")
        return

    command = f"{jewel_executable} {params_file} {medium_params_file}"
    execute_command(command)

    hepmc_file = os.path.join(file_path, f"out-pbpb_{experiment_num}.hepmc")
    if not os.path.exists(hepmc_file):
        raise FileNotFoundError(f"HEPMC file {hepmc_file} was not created.")

    analyze_hepmc(hepmc_file, experiment_num, 'fulljets', background, MLtype)
    return 

#----------------------------------------------#
#----------------------------------------------#
#          RUNNING jewel default pp            #
#----------------------------------------------#
#----------------------------------------------#

def run_vacuum_simulation(experiment_num, nevent, background, MLtype):
    file_path = '/sampa/archive/llimadas/temp/jewel/substructures/general/outputs_default/'
    os.makedirs(file_path, exist_ok=True)
    
    jewel_dir = "/sampa/leonardo/testing/jewel-2.4.0/"
    
    params_file = os.path.join(file_path, f"params_vacuum_{experiment_num}.dat")

    create_file_from_template("/sampa/llimadas/general_parameters/par_vacuum_article-marco.dat", params_file, {
        "NEVENT": nevent,
        "EXPNUM": experiment_num
    })

    jewel_executable = os.path.join(jewel_dir, "jewel-2.4.0-vac")
    if not os.path.isfile(jewel_executable) or not os.access(jewel_executable, os.X_OK):
        print(f"Error: Executable {jewel_executable} is invalid.")
        return

    command = f"{jewel_executable} {params_file}"
    execute_command(command)

    hepmc_file = os.path.join(file_path, f"out-pp_{experiment_num}.hepmc")
    if not os.path.exists(hepmc_file):
        raise FileNotFoundError(f"HEPMC file {hepmc_file} was not created.")

    analyze_hepmc(hepmc_file, experiment_num, 'fulljets', background, MLtype)
    return


#*********************************************************************************#
#                                  RUNNING                                        #
#*********************************************************************************#


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JEWEL simulations.")
    parser.add_argument("--experiment_num", type=int, required=True, help="Experiment number")
    parser.add_argument("--nevent", type=int, required=True, help="Number of events")
    parser.add_argument("--sim_type", type=str, required=True, choices=["medium", "vacuum"], help="Simulation type")
    parser.add_argument("--background", type=str, required=True, choices=["nobkg", "bkg"], help="Background off/on")
    parser.add_argument("--MLtype", type=str, required=True, choices=["train", "test", "val"], help="Purpose type of data that will be generated")
    args = parser.parse_args()

    execution_function = run_simulation if args.sim_type == "medium" else run_vacuum_simulation
    output_root = execution_function(args.experiment_num, args.nevent, args.background, args.MLtype)
    print(f"Job completed: {output_root}")

