import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CAMINHO DO ARQUIVO ---
csv_path = "/Users/murieldaher/Desktop/mestrado/alzheimerDisease/venv_alzheimer/share/dataset/ADNIMERGE_11Nov2025.csv"

print("--- Iniciando a Análise de Progressão por Idade e Gênero ---")

try:
    print("Lendo o arquivo CSV...")
    df = pd.read_csv(csv_path, low_memory=False)
    print("Arquivo CSV lido com sucesso!")

    # --- 1. "CAÇANDO" OS CONVERSORES ---
    
    # --- Grupo 1: Buscando Conversores [CN -> MCI] ---
    cn_starters_ids = df[df['DX_bl'] == 'CN']['PTID'].unique()
    cn_starters_history = df[df['PTID'].isin(cn_starters_ids)]
    cn_to_mci_converters_ids = cn_starters_history[
        cn_starters_history['DX'] == 'MCI'
    ]['PTID'].unique()
    print(f"Encontrados {len(cn_to_mci_converters_ids)} pacientes [CN -> MCI]")

    # --- Grupo 2: Buscando Conversores [CN -> Dementia] ---
    cn_to_dementia_converters_ids = cn_starters_history[
        cn_starters_history['DX'] == 'Dementia'
    ]['PTID'].unique()
    print(f"Encontrados {len(cn_to_dementia_converters_ids)} pacientes [CN -> Dementia]")

    # --- Grupo 3: Buscando Conversores [MCI -> Dementia] ---
    
    # Pegando pacientes que começaram como 'EMCI' OU 'LMCI'
    mci_labels_baseline = ['EMCI', 'LMCI']
    mci_starters_ids = df[df['DX_bl'].isin(mci_labels_baseline)]['PTID'].unique()
    print(f"Total de pacientes que começaram como MCI (EMCI ou LMCI): {len(mci_starters_ids)}")

    # Criando o histórico só deles
    mci_starters_history = df[df['PTID'].isin(mci_starters_ids)]
    
    # Procurando por quem virou 'Dementia'
    mci_to_dementia_converters_ids = mci_starters_history[
        mci_starters_history['DX'] == 'Dementia'
    ]['PTID'].unique()
    print(f"Pacientes que progrediram de MCI (EMCI/LMCI) para Dementia: {len(mci_to_dementia_converters_ids)}")


    # --- 2. PREPARANDO OS DADOS PARA O GRÁFICO (ETIQUETAGEM) ---
    
    baseline_df = df[df['VISCODE'] == 'bl'].drop_duplicates(subset='PTID', keep='first').copy()
    
    baseline_df['Progression'] = 'Nao-Conversor' # Rótulo padrão
    
    baseline_df.loc[baseline_df['PTID'].isin(cn_to_mci_converters_ids), 'Progression'] = 'CN -> MCI'
    baseline_df.loc[baseline_df['PTID'].isin(mci_to_dementia_converters_ids), 'Progression'] = 'MCI -> Dementia'
    baseline_df.loc[baseline_df['PTID'].isin(cn_to_dementia_converters_ids), 'Progression'] = 'CN -> Dementia'

    conversores_df = baseline_df[baseline_df['Progression'] != 'Nao-Conversor']
    
    if conversores_df.empty:
        print("\nNenhum paciente conversor encontrado. Verifique os dados.")
    else:
        print(f"\nTotal de {len(conversores_df)} pacientes conversores encontrados. Gerando 3 gráficos...")
        sns.set_theme(style="whitegrid")

        # --- GRÁFICO 1: Contagem Total de Conversores ---
        plt.figure(figsize=(10, 6))
        sns.countplot(data=conversores_df, x='Progression', hue='Progression', palette='viridis', legend=False)
        plt.title('Gráfico 1: Contagem Total de Pacientes por Grupo de Progressão')
        plt.ylabel('Número de Pacientes')
        plt.xlabel('Tipo de Progressão')
        plt.show()

        # --- GRÁFICO 2: Divisão por Gênero ---
        plt.figure(figsize=(12, 7))
        sns.countplot(data=conversores_df, x='Progression', hue='PTGENDER', palette='pastel')
        plt.title('Gráfico 2: Divisão de Gênero (F/M) por Grupo de Progressão')
        plt.ylabel('Número de Pacientes')
        plt.xlabel('Tipo de Progressão')
        plt.legend(title='Gênero')
        plt.show()
        
        # --- GRÁFICO 3: Distribuição de Idade por Gênero ---
        plt.figure(figsize=(14, 8))
        sns.violinplot(
            data=conversores_df, 
            x='Progression',
            y='AGE',
            hue='PTGENDER',
            split=True,
            inner='quart',
            palette='Set2'
        )
        plt.title('Gráfico 3: Distribuição de Idade (Baseline) por Tipo de Progressão e Gênero', fontsize=16)
        plt.xlabel('Tipo de Progressão', fontsize=12)
        plt.ylabel('Idade (na Baseline)', fontsize=12)
        plt.legend(title='Gênero')
        plt.show()

except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em: {csv_path}")
except KeyError as e:
    print(f"ERRO: A coluna {e} não foi encontrada. Verifique o nome da coluna no CSV.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")