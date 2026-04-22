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
import sys
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

def create_file_from_template(template_path, output_path, replacements):
    with open(template_path, 'r') as file:
        content = file.read()
    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", str(value))
    with open(output_path, 'w') as file:
        file.write(content)
    os.chmod(output_path, 0o644)
    print(f"Created file {output_path} from template.")

#----------------------------------------------

def create_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content.strip())
    os.chmod(filepath, 0o644)  
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

#----------------------------------------------#

def analyze_hepmc(hepmc_file, experiment_num, jet_type, background, MLtype, average):
    analysis_script = "/sampa/llimadas/Jewel2.4/analysis/doStructure"

    if background == 'nobkg':
        background_flag = 'nobkgsub'
    elif background == 'bkg':
        background_flag = 'bkgsub'
    else:
        raise ValueError("background inválido. Use 'nobkg' ou 'bkg'.")


    base_name = os.path.splitext(os.path.basename(hepmc_file))[0]
    if average == "off":
        output_root_name = f"{base_name}_{MLtype}_structure"
    elif average == "on":
        output_root_name = f"{base_name}_avgmedium_{MLtype}_structure"
        
    output_root_path = os.path.join(os.getcwd(), output_root_name + f"_full_{background_flag}.root") 

    # 
    analysis_command = f"{analysis_script} --{jet_type} --{background_flag} {hepmc_file} {output_root_name}"
    execute_command(analysis_command)

    if not os.path.exists(output_root_path):
        raise FileNotFoundError(f"File ROOT {output_root_path} not found.")


    final_path = "/sampa/llimadas/Jewel2.4/analysis/results_root/"
    move_command = f"mv {output_root_path} {final_path} "
    execute_command(move_command)

    print(f"Done.")
    return



#----------------------------------------------

def append_line(file_path, line):
    with open(file_path, "a") as f:
        f.write(line + "\n")

#----------------------------------------------#
#----------------------------------------------#
#        RUNNING jewel-vUSPhydro Pb+Pb         #
#----------------------------------------------#
#----------------------------------------------#

def run_simulation(experiment_num, nevent, background, MLtype, average):
    jewel_dir = "/sampa/leonardo/USP-JEWEL/./"

    CENT = "0-10"
    CODE = "vusp"
    NJOB = experiment_num
    njob_data = lambda x: x//10 if x<=99 else x//100
    if average == "on":
        NJOB_MODIFIED = njob_data(NJOB)  
    elif average == "off":
        NJOB_MODIFIED = NJOB
    
    if average == "off":
        os.environ["MED_DIR"] = f"/sampa/archive/leonardo/Jets/IC/vusphydro/trento/PbPb_5020_flow_GeV/profiles/{CENT}"
        os.environ["TEMP_DIR"] = f"/sampa/archive/llimadas/temp/jewel/substructures/general/outputs_{CODE}/"
        os.environ["JAKI_DIR"] = "/sampa/archive/leonardo/Jets/IC/vusphydro/trento/PbPb_5020_flow_GeV"
    elif average == "on":
        os.environ["MED_DIR"] = f"/sampa/archive/leonardo/Jets/IC/vusphydro/avgtrento/PbPb_5020_flow_GeV/profiles/{CENT}"
        os.environ["TEMP_DIR"] = f"/sampa/archive/llimadas/temp/jewel/substructures/general/outputs_{CODE}_average/"
        os.environ["JAKI_DIR"] = "/sampa/archive/leonardo/Jets/IC/vusphydro/trento/PbPb_5020_flow_GeV"

    TEMP_DIR = os.environ["TEMP_DIR"]
    MED_DIR = os.environ["MED_DIR"]

    os.makedirs(TEMP_DIR, exist_ok=True)

    PAR_FILE = os.path.join(TEMP_DIR, f"params_medium-{CODE}_{NJOB}.dat")
    PAR_MED_FILE = os.path.join(TEMP_DIR, f"medium-{CODE}_{NJOB}.dat")


    create_file_from_template(
        "/sampa/llimadas/general_parameters/par_medium_vUSPhydro_article-marco.dat",
        PAR_FILE,
        {
            "NEVENT": nevent,
            "CENT": CENT,
            "TEMP_DIR": TEMP_DIR,
            "CODE": CODE,
            "NJOB": NJOB,
            "PAR_MED_FILE": PAR_MED_FILE
        }
    )


    BASE_MED_DIR = "/sampa/llimadas/general_parameters/"
    BASE_MED_FILE = os.path.join(BASE_MED_DIR, "medium-params_vUSPhydro_article-marco.dat")  
    if os.path.isfile(BASE_MED_FILE):
        shutil.copy(BASE_MED_FILE, PAR_MED_FILE)  
    else:
        print(f"Erro: arquivo medium-params_vUSPhydro_article-marco.dat não encontrado em {os.path.join(BASE_MED_DIR, CENT)}")
        sys.exit(1)

    append_line(PAR_MED_FILE, f"MEDFILE {os.path.join(MED_DIR, f'{NJOB_MODIFIED}.dat')}") 

    jewel_executable = os.path.join(jewel_dir, "usp-jewel")
    if not os.path.isfile(jewel_executable) or not os.access(jewel_executable, os.X_OK):
        print(f"Error: Executable {jewel_executable} is invalid.")
        return

    command = f"{jewel_executable} {PAR_FILE}"
    execute_command(command)

    hepmc_file = os.path.join(TEMP_DIR, f"out-pbpb_{CODE}_{NJOB}.hepmc")

        


    if not os.path.exists(hepmc_file):
        raise FileNotFoundError(f"HEPMC file {hepmc_file} was not created. Check JEWEL execution.")

    analyze_hepmc(hepmc_file, experiment_num, 'fulljets', background, MLtype, average)  
    return



#*********************************************************************************#
#                                  RUNNING                                        #
#*********************************************************************************#

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JEWEL v-USPHydro simulations.")
    parser.add_argument("--experiment_num", type=int, required=True, help="Experiment number")
    parser.add_argument("--nevent", type=int, required=True, help="Number of events")
    parser.add_argument("--background", type=str, required=True, choices=["nobkg", "bkg"], help="Background off/on")
    parser.add_argument("--MLtype", type=str, required=True, choices=["train", "test", "val"])
    parser.add_argument("--average_medium", type=str, required=False, choices=["off", "on"], default="off", help="Average medium off/on")
    args = parser.parse_args()

    execution_function = run_simulation 
    output_root = execution_function(args.experiment_num, args.nevent, args.background, args.MLtype, args.average_medium)
    print(f"Job completed: {output_root}")
