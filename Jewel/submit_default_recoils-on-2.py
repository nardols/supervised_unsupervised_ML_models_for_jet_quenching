#----------------------------------------------
#
# artigo de referência das simulações: Jet substructure observables for jet quenching in Quark Gluon Plasma: a Machine Learning driven analysis - https://arxiv.org/abs/2304.07196
#
#----------------------------------------------

import subprocess
import os
import queue
from threading import Thread
import shutil
import time

#----------------------------------------------#
#                  FUNÇÕES                     #
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
#----------------------------------------------

"""
def convert_hepmc_to_pu14(hepmc_file, sim_type):
    base_dir = os.getcwd()

    root_file = os.path.join(base_dir, os.path.basename(hepmc_file.replace(".hepmc", ".root")))
    pu14_file = os.path.join(base_dir, os.path.basename(hepmc_file.replace(".hepmc", ".pu14")))

    # Convert HEPMC to ROOT
    if os.path.exists(hepmc_file):
        root_command = f"./hepmc_to_root {hepmc_file} {root_file}"
        execute_command(root_command)
        if not os.path.exists(root_file):
            raise FileNotFoundError(f"ROOT file {root_file} was not created.")
        safe_remove(hepmc_file)  # Delete HEPMC after successful ROOT creation
        print(f"File {hepmc_file} deleted after conversion to ROOT.")
    else:
        raise FileNotFoundError(f"HEPMC file {hepmc_file} does not exist.")

    # Convert ROOT to PU14
    root_to_pu14_command = f"./root_to_pu14 {root_file} {pu14_file}"
    execute_command(root_to_pu14_command)
    if not os.path.exists(pu14_file):
        raise FileNotFoundError(f"PU14 file {pu14_file} was not created.")
    safe_remove(root_file)  # Delete ROOT after successful PU14 creation
    print(f"File {root_file} deleted after conversion to PU14.")

    return pu14_file

#----------------------------------------------

def analyze_pu14(pu14_file, experiment_num, background):
    # Definir o nome do arquivo de saída como ROOT
    output_root = pu14_file.replace(".pu14", "_structure.root")
    analysis_script = "/sampa/llimadas/Jewel2.4/analysis/doStructure.py"

    if (background == "nobkg"):
        analysis_command = f"python {analysis_script} -i {pu14_file} -o {output_root} -n 2000"
    elif (background == "bkg"):
        background_file = f"/sampa/llimadas/Jewel2.4/analysis/thermalEvent/results/ThermalEventsMult7000PtAv1.20_{experiment_num}.pu14"
        analysis_command = f"python {analysis_script} -i {pu14_file} -b{background_file} -o {output_root} -n 2000"
        
    execute_command(analysis_command)
    if not os.path.exists(output_root):
        raise FileNotFoundError(f"Output ROOT file {output_root} was not created.")
    else:
        os.chmod(output_root, 0o644)
        shutil.move(output_root, f"/sampa/llimadas/Jewel2.4/analysis/results/{os.path.basename(output_root)}") # -> salva o ROOT em uma pasta de resultados

    # Excluir o arquivo .pu14 após a conversão
    if os.path.exists(pu14_file):
        safe_remove(pu14_file)
        print(f"File {pu14_file} deleted after analysis.")

    print(f"Analysis completed for {pu14_file}. Output ROOT file generated at {output_root}.")
    return output_root
"""

#----------------------------------------------
#----------------------------------------------

def analyze_hepmc(hepmc_file, experiment_num, jet_type, background, MLtype):
    analysis_script = "/sampa/llimadas/Jewel2.4/analysis/doStructure"

    if background == 'nobkg':
        background_flag = 'nobkgsub'
    elif background == 'bkg':
        background_flag = 'bkgsub'
    else:
        raise ValueError("background inválido. Use 'nobkg' ou 'bkg'.")


    base_name = os.path.splitext(os.path.basename(hepmc_file))[0]
    output_root_name = f"{base_name}_{MLtype}_structure"
    output_root_path = os.path.join(os.getcwd(), output_root_name + f"_full_{background_flag}.root") # doStructure adiciona estes nomes

    # 
    analysis_command = f"{analysis_script} --{jet_type} --{background_flag} {hepmc_file} {output_root_name}"
    execute_command(analysis_command)

    # confere se o arquivo foi criado
    if not os.path.exists(output_root_path):
        raise FileNotFoundError(f"Arquivo ROOT {output_root_path} não foi criado.")

    # muda pro destino final
    final_path = "/sampa/llimadas/Jewel2.4/analysis/results_root/"
    move_command = f"mv {output_root_path} {final_path} "
    execute_command(move_command)

    print(f"Análise finalizada.")
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

    #pu14_file = convert_hepmc_to_pu14(hepmc_file, sim_type="medium")
    #output_root = analyze_pu14(pu14_file, experiment_num, background) #final file -> results are here!
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

    #pu14_file = convert_hepmc_to_pu14(hepmc_file, sim_type="medium")
    #output_root = analyze_pu14(pu14_file, experiment_num, background) #final file -> results are here!
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

