import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from alzheimer import config
from alzheimer.data_io import load_merge_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory analysis pipeline for the ADNIMERGE dataset.",
    )
    parser.add_argument(
        "--csv-path",
        help=(
            "Override the ADNIMERGE CSV path."
            f" Defaults to {config.DEFAULT_MERGE_CSV} or the value set in"
            f" {config.MERGE_CSV_ENV_VAR}."
        ),
    )
    return parser.parse_args()

args = parse_args()
csv_path = config.get_merge_csv_path(args.csv_path)

print("--- Iniciando a Análise de Progressão por Idade e Gênero ---")

try:
    print(f"Lendo o arquivo CSV em {csv_path}...")
    df = load_merge_table(csv_path)
    print("Arquivo CSV lido e pré-processado com sucesso!")

    # --- 1. "CAÇANDO" OS CONVERSORES ---
    
    # --- Grupo 1: Buscando Conversores [CN -> MCI] ---
    cn_starters_ids = df[df['DX_bl'] == 'CN']['PTID'].unique()

    # Pergunta 1: Quantos pacientes chegaram sem a doença (CN)?
    print(f"\n[Insight 1] Total de pacientes que começaram o estudo como CN: {len(cn_starters_ids)}")

    # Pegando o histórico completo de pacientes que começaram como 'CN'
    cn_starters_history = df[df['PTID'].isin(cn_starters_ids)]

    # Pergunta 2: Quantos desses CN desenvolveram a doença (MCI ou Demência) DEPOIS?
    progressed_ids = cn_starters_history[
        cn_starters_history['DX'].isin(['MCI', 'Dementia'])
    ]['PTID'].unique()
    
    print(f"\n[Insight 2] Total de pacientes CN que progrediram (para MCI ou Demência): {len(progressed_ids)}")

    cn_to_mci_converters_ids = cn_starters_history[
        cn_starters_history['DX'] == 'MCI'
    ]['PTID'].unique()
    print(f"Encontrados {len(cn_to_mci_converters_ids)} pacientes [CN -> MCI]")

    # Pergunta 3: "Alguém pulou o estágio de MCI?" (Foi de CN direto para Demência)

    # --- Grupo 2: Buscando Conversores [CN -> Dementia] e obtendo insights ---
    cn_to_dementia_converters_ids = cn_starters_history[
        cn_starters_history['DX'] == 'Dementia'
    ]['PTID'].unique()

    # pegando o histórico (tabela filtrada) de pacientes que foram de CN para Demência
    cn_converters_history = cn_starters_history[cn_starters_history['PTID'].isin(cn_to_dementia_converters_ids)]

    cn_first_dementia_visit = cn_converters_history[
        cn_converters_history['DX'] == 'Dementia'
    ].groupby('PTID').first()

    # extraindo os tempos de progressão
    cn_progression_times = cn_first_dementia_visit['Month']

    if len(cn_progression_times) > 0:
        print(f"Encontrados {len(cn_progression_times)} pacientes [CN -> Dementia].")
        print(f"Tempo MÉDIO de progressão: {cn_progression_times.mean()/12:.2f} anos")
        print(f"Tempo MEDIANO de progressão: {cn_progression_times.median()/12:.2f} anos")
        print(f"Progressão mais RÁPIDA: {cn_progression_times.min()/12:.2f} anos")
        print(f"Progressão mais LENTA: {cn_progression_times.max()/12:.2f} anos")
    else:
        print("Nenhum paciente [CN -> Dementia] encontrado.")

    # Convertemos as listas do pandas para "sets" do Python
    set_dementia = set(cn_to_dementia_converters_ids)
    set_mci = set(cn_to_mci_converters_ids)
    
    # Encontra a diferença (quem está em 'set_dementia' e NÃO está em 'set_mci')
    set_skippers = set_dementia.difference(set_mci)
    
    print(f"\n[RESPOSTA FINAL] Pacientes que foram de CN -> Demência (sem nunca passar por MCI): {len(set_skippers)}")

    if len(set_skippers) > 0:
        print(f"Os IDs desses pacientes 'saltadores' são: {list(set_skippers)}")
    print("\n")
    # --- Grupo 3: Buscando Conversores [MCI -> Dementia] ---
    
    # Pegando pacientes que começaram como 'EMCI' OU 'LMCI'
    mci_labels_baseline = ['EMCI', 'LMCI']
    mci_starters_ids = df[df['DX_bl'].isin(mci_labels_baseline)]['PTID'].unique()
    print(f"Total de pacientes que começaram como MCI (EMCI ou LMCI): {len(mci_starters_ids)}")

    # Criando o histórico só deles
    mci_starters_history = df[df['PTID'].isin(mci_starters_ids)]
    
    # Procurando por quem virou 'Dementia'
    mci_to_dementia_converters_ids = mci_starters_history[
        (mci_starters_history['DX'] == 'Dementia') & (mci_starters_history['VISCODE'] != 'bl')
    ]['PTID'].unique()
    print(f"Pacientes que progrediram de MCI (EMCI/LMCI) para Dementia: {len(mci_to_dementia_converters_ids)}")

    # pegando o histórico (tabela filtrada) de pacientes que foram de MCI para Demência
    mci_converters_history = mci_starters_history[mci_starters_history['PTID'].isin(mci_to_dementia_converters_ids)]
    # 5. Encontrar a PRIMEIRA visita de Demência para cada paciente
    mci_first_dementia_visit = mci_converters_history[
        (mci_converters_history['DX'] == 'Dementia') & (mci_converters_history['VISCODE'] != 'bl')
    ].groupby('PTID').first()
    
    # 6. Extrair a lista de meses
    mci_progression_times = mci_first_dementia_visit['Month']
    print("\n--->", mci_progression_times.min())
    
    if len(mci_progression_times) > 0:
        print(f"Encontrados {len(mci_progression_times)} pacientes [MCI -> Dementia].")
        print(f"Tempo MÉDIO de progressão: {mci_progression_times.mean()/12:.2f} anos")
        print(f"Tempo MEDIANO de progressão: {mci_progression_times.median()/12:.2f} anos")
        print(f"Progressão mais RÁPIDA: {mci_progression_times.min()/12:.2f} anos")
        print(f"Progressão mais LENTA: {mci_progression_times.max()/12:.2f} anos")
    else:
        print("Nenhum paciente [MCI -> Dementia] encontrado.")


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

# Gráfico para ilustrarevolução dos pacientes para o alzheimer ao longo do tempo

# Criar dataframes "arrumados" (tidy) para o seaborn
    cn_plot_df = pd.DataFrame({
        'Tempo (Anos)': cn_progression_times/12,
        'Grupo': 'CN -> Dementia'
    })
    
    mci_plot_df = pd.DataFrame({
        'Tempo (Anos)': mci_progression_times/12,
        'Grupo': 'MCI -> Dementia'
    })
    
    # Combinar os dois dataframes em um só
    plot_df = pd.concat([cn_plot_df, mci_plot_df])

    # --- Bloco 4: Geração do Gráfico para o insight: "Depois de quanto tempo em média as pessoas evoluíram para alzheimer?" ---
    
    if plot_df.empty:
        print("\nNenhum dado de progressão para plotar.")
    else:
        print("\nGerando gráfico... (Feche a janela do gráfico para o script terminar)")
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 7))
        
        # 1. Desenha o Boxplot (caixas)
        sns.boxplot(
            data=plot_df, 
            x='Grupo', 
            y='Tempo (Anos)', 
            palette='pastel',
            hue='Grupo',       
            legend=False
        )
        
        # 2. Sobrepõe os pontos de dados individuais
        sns.swarmplot(
            data=plot_df, 
            x='Grupo', 
            y='Tempo (Anos)',
            color=".25",
            size=3.0,
            alpha=0.6
        )
        
        plt.title('Tempo de Progressão para Alzheimer (em Anos)', fontsize=16)
        plt.xlabel('Grupo de Progressão', fontsize=12)
        plt.ylabel('Anos desde a Baseline até o Diagnóstico', fontsize=12)
        
        plt.show()

    # --- Tempo de Progressão Separado por Sexo ---
    print("\nGerando gráfico de Tempo de Progressão por Sexo...")

    # Criar dataframes "arrumados" (tidy), agora incluindo a coluna GÊNERO
    # O truque: cn_first_dementia_visit já tem a coluna 'PTGENDER' porque usamos .first()
    
    cn_plot_df = pd.DataFrame({
        'Tempo (Anos)': cn_progression_times / 12,
        'Grupo': 'CN -> Dementia',
        'Gênero': cn_first_dementia_visit['PTGENDER'] # <--- Adicionamos o Gênero aqui!
    })
    
    mci_plot_df = pd.DataFrame({
        'Tempo (Anos)': mci_progression_times / 12,
        'Grupo': 'MCI -> Dementia',
        'Gênero': mci_first_dementia_visit['PTGENDER'] # <--- E aqui!
    })
    
    # Combinar os dois dataframes em um só
    plot_df = pd.concat([cn_plot_df, mci_plot_df])

    if plot_df.empty:
        print("\nNenhum dado de progressão para plotar.")
    else:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # 1. Desenha o Boxplot dividido por Gênero
        sns.boxplot(
            data=plot_df, 
            x='Grupo', 
            y='Tempo (Anos)', 
            hue='Gênero',       # <--- A MÁGICA: Separa as caixas por sexo
            palette='pastel'
        )
        
        # 2. Sobrepõe os pontos (com dodge=True para eles não ficarem misturados)
        sns.swarmplot(
            data=plot_df, 
            x='Grupo', 
            y='Tempo (Anos)',
            hue='Gênero',       # Precisamos dizer o hue aqui também
            dodge=True,         # <--- Importante: Separa os pontos "Male" dos "Female"
            color=".25",
            size=3.0,
            alpha=0.6,
            legend=False        # Esconde a legenda extra dos pontos
        )
        
        plt.title('Tempo de Progressão para Alzheimer (Separado por Sexo)', fontsize=16)
        plt.xlabel('Grupo de Progressão', fontsize=12)
        plt.ylabel('Anos desde a Baseline até o Diagnóstico', fontsize=12)
        plt.legend(title='Gênero') # Garante que a legenda apareça bonitinha
        
        plt.show()

    # Em pacientes que progridem para o alzheimer, o que declina primeiro?
    #  O volume do hipocampo ou a pontuação do teste MSE?  

    print("\n--- Analisando a Velocidade de Declínio (Normalizada) ---")

    # 1. Pegar histórico dos conversores
    history_df = df[df['PTID'].isin(mci_to_dementia_converters_ids)].copy()

    # 2. Converter para numérico (forçando erros a virar NaN)
    cols_to_numeric = ['Hippocampus', 'MMSE', 'Month']
    for col in cols_to_numeric:
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce')

    # 3. Criar Tabela de Baseline (CORREÇÃO DE ROBUSTEZ)
    # Pegamos apenas as linhas 'bl'
    bl_data = history_df[history_df['VISCODE'] == 'bl'][['PTID', 'Hippocampus', 'MMSE']]
    
    # Renomeamos explicitamente
    bl_data = bl_data.rename(columns={
        'Hippocampus': 'Hippo_BASELINE', 
        'MMSE': 'MMSE_BASELINE'
    })

    # 4. Juntar (Merge) de volta no histórico principal
    # Usamos 'left' join para garantir que não perdemos ninguém
    merged_df = pd.merge(history_df, bl_data, on='PTID', how='left')
    
    # Verificação de depuração (caso dê erro de novo, saberemos por quê)
    if 'MMSE_BASELINE' not in merged_df.columns:
        print("ERRO CRÍTICO: Coluna de baseline não foi criada corretamente.")
        print("Colunas disponíveis:", merged_df.columns)
    
    # 5. CALCULAR A MUDANÇA PERCENTUAL
    # Agora usamos os nomes novos e garantidos: 'Hippo_BASELINE' e 'MMSE_BASELINE'
    merged_df['Mudança Hipocampo'] = (merged_df['Hippocampus'] - merged_df['Hippo_BASELINE']) / merged_df['Hippo_BASELINE']
    merged_df['Mudança MMSE'] = (merged_df['MMSE'] - merged_df['MMSE_BASELINE']) / merged_df['MMSE_BASELINE']

    # 6. Preparar para plotagem (Melt)
    plot_data = pd.melt(
        merged_df, 
        id_vars=['PTID', 'Month', 'PTGENDER'], 
        value_vars=['Mudança Hipocampo', 'Mudança MMSE'],
        var_name='Métrica', 
        value_name='Mudança Percentual'
    )

    # --- GRÁFICO 1: Visão Geral ---
    print("Gerando gráfico geral de declínio...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=plot_data, 
        x='Month', 
        y='Mudança Percentual', 
        hue='Métrica', 
        style='Métrica',
        markers=True, 
        dashes=False,
        palette="tab10" # Paleta segura
    )
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('Trajetória de Declínio: Hipocampo vs. MMSE (Conversores MCI -> AD)')
    plt.ylabel('Mudança em relação ao Início (0.0 = 0%, -0.2 = -20%)')
    plt.xlabel('Meses de Estudo')
    plt.xlim(0, 48)
    plt.show()

    # --- GRÁFICO 2: Separado por Sexo ---
    print("Gerando gráfico separado por sexo...")
    g = sns.relplot(
        data=plot_data, 
        x='Month', 
        y='Mudança Percentual', 
        hue='Métrica', 
        col='PTGENDER',
        kind='line',
        markers=True,
        palette="tab10"
    )
    g.map(plt.axhline, y=0, color="black", linestyle="--", linewidth=0.8)
    g.set_axis_labels("Meses de Estudo", "Mudança Percentual (Declínio)")
    g.fig.suptitle('Comparação de Declínio por Sexo', y=1.03)
    plt.xlim(0, 48)
    plt.show() 

except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em: {csv_path}")
    print(
        "Use --csv-path ou a variável de ambiente "
        f"{config.MERGE_CSV_ENV_VAR} para informar o local correto."
    )
except KeyError as e:
    print(f"ERRO: A coluna {e} não foi encontrada. Verifique o nome da coluna no CSV.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
