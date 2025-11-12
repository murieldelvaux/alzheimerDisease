import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CAMINHOS DOS ARQUIVOS ---
# (Colocamos os dois caminhos aqui no início para organizar)
csv_path = "/Users/murieldaher/Desktop/mestrado/alzheimerDisease/venv_alzheimer/share/dataset/ADNIMERGE_11Nov2025.csv"
path_to_save = "data_cleaner.csv"

print("--- Iniciando a Análise e Preparação em Python ---")

try:
    print("Lendo o arquivo CSV...")
    df = pd.read_csv(csv_path, low_memory=False)

    total_unique_patients = df['PTID'].nunique()
    print(f"Total de pacientes únicos no dataset: {total_unique_patients}")
    
    print("Arquvivo CSV lido com sucesso!");
    
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em: {file_path}")
except KeyError:
    print("ERRO: Coluna 'RID' não encontrada. O nome da coluna mudou?")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
