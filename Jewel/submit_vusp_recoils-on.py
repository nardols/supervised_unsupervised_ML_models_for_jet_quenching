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
import sys
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
    os.chmod(filepath, 0o644)  # Permissão padrão: leitura e escrita para o dono
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

    root_file = os.path.join(base_dir, os.path.basename(hepmc_file.replace(".hepmc", "vUSP.root")))
    pu14_file = os.path.join(base_dir, os.path.basename(hepmc_file.replace(".hepmc", "vUSP.pu14")))

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
    output_root = pu14_file.replace(".pu14", "_structure_vUSP.root")
    analysis_script = "/sampa/llimadas/Jewel2.4/analysis/doStructure.py"

    if (background == "nobkg"):
        analysis_command = f"python {analysis_script} -i {pu14_file} -o {output_root} -n 10000"
    elif (background == "bkg"):
        background_file = f"/sampa/llimadas/Jewel2.4/analysis/thermalEvent/results/ThermalEventsMult7000PtAv1.20_{experiment_num}.pu14"
        analysis_command = f"python {analysis_script} -i {pu14_file} -b{background_file} -o {output_root} -n 10000"
        
    execute_command(analysis_command)
    if not os.path.exists(output_root):
        raise FileNotFoundError(f"Output ROOT file {output_root} was not created.")
    os.chmod(output_root, 0o644)

    # Excluir o arquivo .pu14 após a conversão
    if os.path.exists(pu14_file):
        safe_remove(pu14_file)
        print(f"File {pu14_file} deleted after analysis.")

    print(f"Analysis completed for {pu14_file}. Output ROOT file generated at {output_root}.")
    return output_root

"""

#----------------------------------------------
#----------------------------------------------

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

def append_line(file_path, line):
    """Acrescenta uma linha ao final do arquivo especificado."""
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
        NJOB_MODIFIED = njob_data(NJOB)  # os arquivos pro meio médio vao de 0 a 9 -> aqui eu pego o numero inteiro dos q vao de 0 a 999
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

    # Gera os arquivos de parametros a partir dos templates
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

    # Copia o arquivo base do meio e adiciona o MEDFILE (q é específico meio a meio)
    BASE_MED_DIR = "/sampa/llimadas/general_parameters/"
    BASE_MED_FILE = os.path.join(BASE_MED_DIR, "medium-params_vUSPhydro_article-marco.dat")  # meus parametros (seguindo artigo do Marco)
    if os.path.isfile(BASE_MED_FILE):
        shutil.copy(BASE_MED_FILE, PAR_MED_FILE)  # copia aquele .dat para a minha saída no arquive
    else:
        print(f"Erro: arquivo medium-params_vUSPhydro_article-marco.dat não encontrado em {os.path.join(BASE_MED_DIR, CENT)}")
        sys.exit(1)

    append_line(PAR_MED_FILE, f"MEDFILE {os.path.join(MED_DIR, f'{NJOB_MODIFIED}.dat')}")  # pega o arquivo dentro do dir. do Leo e junta com o meu

    jewel_executable = os.path.join(jewel_dir, "usp-jewel")
    if not os.path.isfile(jewel_executable) or not os.access(jewel_executable, os.X_OK):
        print(f"Error: Executable {jewel_executable} is invalid.")
        return

    command = f"{jewel_executable} {PAR_FILE}"
    execute_command(command)

    hepmc_file = os.path.join(TEMP_DIR, f"out-pbpb_{CODE}_{NJOB}.hepmc")

        

    # Verifica se o arquivo HEPMC foi criado
    if not os.path.exists(hepmc_file):
        raise FileNotFoundError(f"HEPMC file {hepmc_file} was not created. Check JEWEL execution.")

    #pu14_file = convert_hepmc_to_pu14(hepmc_file, sim_type="medium")
    #output_root = analyze_pu14(pu14_file, experiment_num, background)
    analyze_hepmc(hepmc_file, experiment_num, 'fulljets', background, MLtype, average)   # passando avergae flag pro analyze pq fica mais fácil mudar o nome lá
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
    parser.add_argument("--MLtype", type=str, required=True, choices=["train", "test", "val"], help="Purpose type of data that will be generated")
    parser.add_argument("--average_medium", type=str, required=False, choices=["off", "on"], default="off", help="Average medium off/on")
    args = parser.parse_args()

    execution_function = run_simulation 
    output_root = execution_function(args.experiment_num, args.nevent, args.background, args.MLtype, args.average_medium)
    print(f"Job completed: {output_root}")
