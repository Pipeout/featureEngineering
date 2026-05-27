import logging
import os
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml


def mapping_disciplines_names(
    df: pd.DataFrame, column: str, new_column: str, curso: str
) -> pd.DataFrame:
    """
    Still unsure of the usefulness of this function
    """
    if curso == "CC":
        regex_mapping = {
            r".*calculo.*|.*Cálculo.*|.*Calculo.*|.*geometria analítica.*"
            r"|.*VGA.*|.*Vetores e Geometria Analítica.*|.*Geometria Analítica.*"
            r"|.*álgebra linear.*|.*Álgebra Linear.*|.*Matemática.*|.*Geometria Analítica.*"
            r"|.*Vetorial.*|.*Estatística.*": "Núcleo_de_Matemática",
            r".*programação.*|.*Programação.*|.*Redes.*|.*Software.*|.*Compiladores.*"
            r"|.*Laboratório.*|.*Dados.*|.*Sistemas.*|.*Computação.*"
            r"|.*Gráfica.*|.*Computadores.*|.*Inteligência Artificial.*|.*Microcontroladores.*"
            r"|.*Projeto.*|.*Processamento.*|.*Autômatos.*|.*Eletrônica Básica.*": "Núcleo_de_Computação",
            r".*Trabalho de Curso.*": "Núcleo_Trabalho_de_Curso",
            r".*Eletromagnetismo.*|.*Mecânica.*": "Núcleo_de_Física",
            r".*metodolodia.*|.*filosofia.*|.*Práticas de Leitura e Produção de Texto.*"
            r"|.*Empreendedorismo.*|.*Libras.*|.*Inglês Instrumental.*|.*Informática Aplicada à Educação.*": "Núcleo_de_Humanidades",
            r".*Lógica Digital.*|.*Arquitetura de Computadores.*"
            r"|.*Organização de Computadores.*": "Núcleo_de_Hardware",
        }
    else:
        regex_mapping = {}
    try:
        df[new_column] = df[column].apply(normalize_names)

        def apply_regex(name):
            if not isinstance(name, str):
                return name
            for pattern, correct_name in regex_mapping.items():
                if re.search(pattern, name, re.IGNORECASE):
                    return correct_name
            return name

        if column in df.columns:
            df[new_column] = df[column].apply(apply_regex)
            return df
        raise KeyError(f"Column {column} does not exist")
    except KeyError as e:
        print(f"Column {e} does not exist in dataset.")
        raise
    except RuntimeError as e:
        print(f"Can't fill data{e}")
        raise


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CONFIG_PATH = os.path.join(BASE_DIR, "../configs/cleaning.yaml")


def get_config_file():
    try:
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "configs" / "cleaning.yaml"
        return path
    except NameError:  # if it is a jupyter file
        return Path("/training-app/configs/training.yaml")


CONFIG_PATH = get_config_file()


# this function takes two dates in string format and calculate their difference in years  (date1 - date2)
def calculate_age(
    df: pd.DataFrame, date1: str, date2: str, new_column_name: str
) -> pd.DataFrame:
    """
    Users native datetime to calculate date1 - date2
    """
    try:
        # this one is "Data de Ocorrência"
        df[date1] = pd.to_datetime(
            df[date1], format="%d/%m/%Y %H:%M:%S", errors="raise"
        )
        df[date2] = pd.to_datetime(df[date2], format="%m/%d/%Y", errors="raise")
        df.loc[:, new_column_name] = (df[date1] - df[date2]).dt.days / 365.25
    except Exception as e:
        logging.info(f"\n[ERRO]: {e}")
        raise
    return df


###################### Inserting new functions ######################


def assert_unique(df, cols):
    duplicates = df.duplicated(subset=cols, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Dataset is NOT unique on {cols}. {duplicates.sum()} duplicate rows found."
        )
    return df


def calculate_ano_sem(df: pd.DataFrame) -> pd.DataFrame:
    df["Ano"] = df["AnoSem"].astype("int")
    df["Parcela"] = df["Semestre"] / 10
    if "AnoSem" in df.columns:
        df = df.drop(columns={"AnoSem"})
    df["AnoSem"] = df["Ano"] + df["Parcela"]
    df = df.drop("Parcela", axis=1)
    return df


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=cols)
    return df


def merge_dfs(
    df: pd.DataFrame, df_to_merge: pd.DataFrame, cols: list[str], how: str, key: str
) -> pd.DataFrame:
    return df.merge(df_to_merge[cols], on=key, how=how)


def standardize_column_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = (
        df[col]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.upper()
        .str.replace(r"\(OPTATIVA\)", "", regex=True)
        .str.strip()
    )
    return df


def load_config(CONFIG_PATH):
    """
    Selects the current dataset's config file we are interest in.
    """
    with open(CONFIG_PATH, "r") as f:
        full_config = yaml.safe_load(f)

    try:
        current_dataset = full_config["CURRENT_DATASET"]
        logging.info(f"\nloading current dataset: {current_dataset}")
        if current_dataset not in full_config["DATASETS"]:
            raise ValueError(f"\nDataset {current_dataset} not found!")

        return full_config["DATASETS"][current_dataset]

    except Exception as e:
        logging.exception(
            f"There was an error handling the config cleaning.yaml file {e}"
        )
        raise


def load_datasets(CONFIG_PATH: str) -> pd.DataFrame:
    """
    Loads the datasets and separetes them
    """

    dfs = load_config(CONFIG_PATH)

    df_active = pd.read_csv(dfs["ACTIVE_DATASET"])
    df_deactive = pd.read_csv(dfs["EVADED_DATASET"])
    df_history = pd.read_csv(dfs["HISTORY_DATASET"])

    return df_active, df_deactive, df_history


def eliminating_duplicates_ap_ae(df: pd.DataFrame) -> pd.DataFrame:
    """
    AE and AP are duplicated in the dataset.
    Therefore it must be converted to one single row.
    """

    ae_ap_pairs = df.groupby(["RGA_Anon", "Nome_Disciplina"]).filter(
        lambda g: {"AP", "AE"}.issubset(set(g["Situação"]))
    )

    df_adjust = df.copy()

    nota_ap = (
        ae_ap_pairs[ae_ap_pairs["Situação"] == "AP"]
        .groupby(["RGA_Anon", "Nome_Disciplina"])["Nota"]
        .max()
    )

    mask_ae = (df_adjust["Situação"] == "AE") & (
        df_adjust.set_index(["RGA_Anon", "Nome_Disciplina"]).index.isin(nota_ap.index)
    )

    df_adjust.loc[mask_ae, "Nota"] = (
        df_adjust.loc[mask_ae]
        .set_index(["RGA_Anon", "Nome_Disciplina"])
        .index.map(nota_ap)
    )

    mask_ap_to_drop = (df_adjust["Situação"] == "AP") & (
        df_adjust.set_index(["RGA_Anon", "Nome_Disciplina"]).index.isin(nota_ap.index)
    )

    df_adjust = df_adjust[~mask_ap_to_drop].copy()
    return df_adjust


def setting_subject_failures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert subject status into a binary outcome:
    - 1 = failure in a subject
    - 0 = non-failure
    Rows with 'MA' status are excluded.
    """
    df = df.copy()
    df = df[df["Situação"] != "MA"]
    failures = ["RMF", "RM", "RP", "RF"]

    df["Situação"] = np.where(df["Situação"].isin(failures), 1, 0)
    return df


def calculate_failure_ratio(df: pd.DataFrame) -> pd.DataFrame:

    df["Reprovacao_Ponderada_Semestral"] = df["Crédito"] * df["Situação"]

    df["Reprovacao_Ponderada_Semestral"] = df.groupby(["AnoSem", "RGA_Anon"])[
        "Reprovacao_Ponderada_Semestral"
    ].transform("sum")

    total_credit = df.groupby(["AnoSem", "RGA_Anon"])["Crédito"].transform("sum")
    total_credit = total_credit.astype("float")
    df["Reprovacao_Ponderada_Semestral"] = df["Reprovacao_Ponderada_Semestral"].astype(
        "float"
    )

    df["Reprovação_Media_Semestral"] = (
        df["Reprovacao_Ponderada_Semestral"] / total_credit
    )
    df = df.drop(columns={"Reprovacao_Ponderada_Semestral"})

    return df


def selecting_valid_period(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["AnoSem"] >= 2009.1].copy()


def calculate_permanence_period_in_semesters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Período ingresso is float and convert to decimal
    df["Período ingresso"] = df["Período ingresso"].astype(float) / 10

    # Cap any fractional semester > 2 to 2 (e.g., 2024.3 -> 2024.2)
    def cap_sem(x):
        int_part = int(x)
        frac_part = int(round((x - int_part) * 10))
        frac_part = min(frac_part, 2)  # cap to 2
        return float(f"{int_part}.{frac_part}")

    df["AnoSemCap"] = df["AnoSem"].apply(cap_sem)
    df["Período ingresso Cap"] = df["Período ingresso"].apply(cap_sem)

    # Create chronological mapping
    all_values = (
        pd.concat([df["AnoSemCap"], df["Período ingresso Cap"]]).dropna().unique()
    )
    mapping = {val: i + 1 for i, val in enumerate(sorted(all_values))}

    # Map to ordered IDs
    df["AnoSemIdOrdered"] = df["AnoSemCap"].map(mapping)
    df["PeriodoIngressoIdOrdered"] = df["Período ingresso Cap"].map(mapping)

    # Calculate permanence in semesters
    df["Tempo_Permanencia_Em_Semestres"] = (
        df["AnoSemIdOrdered"] - df["PeriodoIngressoIdOrdered"] + 1
    )

    # Drop helper columns
    df = df.drop(
        columns=[
            "AnoSemIdOrdered",
            "PeriodoIngressoIdOrdered",
            "AnoSemCap",
            "Período ingresso Cap",
        ]
    )

    return df


def calculate_total_accumulated_credits(df: pd.DataFrame) -> pd.DataFrame:

    mapping = {20241: 210, 20191: 200, 20091: 211}
    df = df.copy()

    df["Total_creditos_estrutura"] = df["Estrutura"].map(mapping)

    resumo_creditos = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])["Crédito"]
        .sum()
        .reset_index()
    )

    resumo_creditos["Total_Creditos_Acumulados"] = resumo_creditos.groupby("RGA_Anon")[
        "Crédito"
    ].cumsum()
    df = df.merge(
        resumo_creditos[
            ["RGA_Anon", "Tempo_Permanencia_Em_Semestres", "Total_Creditos_Acumulados"]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )

    return df


def calculate_normalized_academic_age(df):
    # 1. Definir os limites por estrutura (ajuste os valores conforme sua realidade)
    metas = {
        20091: {"min_credits": 211, "ideal_semesters": 8},
        20191: {"min_credits": 200, "ideal_semesters": 8},
        20241: {"min_credits": 210, "ideal_semesters": 8},
    }
    df = df.copy()

    def get_age(row):
        struct = row["Estrutura"]
        if struct not in metas:
            return row["Tempo_Permanencia_Em_Semestres"]  # fallback

        meta = metas[struct]
        progresso = min(1.0, row["Total_Creditos_Acumulados"] / meta["min_credits"])

        return progresso * meta["ideal_semesters"]

    df["Idade_Academica"] = df.apply(get_age, axis=1)
    df["Estrutura"] = df["Estrutura"].astype(int)

    return df


def calculate_academic_lag_in_semesters(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
    df["Tempo_Permanencia_Em_Semestres"] = df["Tempo_Permanencia_Em_Semestres"].astype(
        float
    )
    df["Idade_Academica"] = df["Idade_Academica"].astype(float)
    df["Lag_Academico_Em_Semestres"] = (
        df.groupby("RGA_Anon")
        .apply(lambda g: g["Tempo_Permanencia_Em_Semestres"] - g["Idade_Academica"])
        .reset_index(level=0, drop=True)
    )
    return df


def log_and_pipe(
    df: pd.DataFrame, success_msg: str, func: Callable, *args: Any, **kwargs: Any
) -> pd.DataFrame:
    """Logs the start, executes the function, logs success, and returns the DataFrame."""
    logging.info(f"\n\n[INFO]: Starting operation: {func.__name__}...")

    df_transformed = func(df, *args, **kwargs)

    logging.info(f"\n[OK]: {success_msg}")
    return df_transformed


def calculate_lag_delta(df: pd.DataFrame, lag_column: str) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
    df["Lag_Academico_Delta"] = df.groupby("RGA_Anon")[lag_column].diff().fillna(0)
    return df


def calculate_deltas_fixed(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    # 1. Ensure chronological order
    df = df.sort_values(by=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])

    sem_snapshot = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
        .agg({"Lag_Academico_Em_Semestres": "first", "MG_Semestre": "first"})
        .reset_index()
    )

    sem_snapshot["Lag_Academico_Delta"] = (
        sem_snapshot.groupby("RGA_Anon")["Lag_Academico_Em_Semestres"].diff().fillna(0)
    )

    df = df.drop(
        columns=["Lag_Academico_Delta"], errors="ignore"
    )  # Clean up if they exist
    df = df.merge(
        sem_snapshot[
            ["RGA_Anon", "Tempo_Permanencia_Em_Semestres", "Lag_Academico_Delta"]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )

    return df


def calculate_coefficient(df: pd.DataFrame) -> pd.DataFrame:
    """Now we will calculate the CR - Coeficiente de Rendimento - per semester
    Fórmula:  (Soma de Nota x Crédito) / (Soma dos Créditos)
    """

    df = df.copy()
    df = df.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
    # numerator of the division
    df["NC_Materia"] = df["Nota"] * df["Crédito"]

    # a summary for each semester
    resumo_semestral = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
        .agg({"NC_Materia": "sum", "Crédito": "sum"})
        .reset_index()
    )

    # now we sum all the previous NC and the credits from previous semesters
    resumo_semestral["NC_Acumulado"] = resumo_semestral.groupby("RGA_Anon")[
        "NC_Materia"
    ].cumsum()
    resumo_semestral["Creditos_Acumulados"] = resumo_semestral.groupby("RGA_Anon")[
        "Crédito"
    ].cumsum()

    # coefficient up until that semester
    resumo_semestral["Coeficiente_Rendimento"] = (
        resumo_semestral["NC_Acumulado"] / resumo_semestral["Creditos_Acumulados"]
    )

    if "Coeficiente_Rendimento" in df.columns:
        df = df.drop(columns=["Coeficiente_Rendimento"])

    df = df.merge(
        resumo_semestral[
            ["RGA_Anon", "Tempo_Permanencia_Em_Semestres", "Coeficiente_Rendimento"]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )

    # Limpeza de colunas técnicas
    df = df.drop(columns=["NC_Materia"])

    return df


def calculate_coefficient_delta(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])

    snapshot = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"]).first().reset_index()
    )
    snapshot["Coeficiente_Rendimento_Delta"] = (
        snapshot.groupby("RGA_Anon")["Coeficiente_Rendimento"].diff().fillna(0)
    )

    if "Coeficiente_Rendimento_Delta" in df.columns:
        df = df.drop("Coeficiente_Rendimento_Delta", axis=1)

    df = df.merge(
        snapshot[
            [
                "RGA_Anon",
                "Tempo_Permanencia_Em_Semestres",
                "Coeficiente_Rendimento_Delta",
            ]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )
    return df


def mark_pandemic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def classify(x):
        if 2020.1 <= x <= 2021.2:
            return "Remoto"
        elif 2022.1 <= x <= 2022.2:
            return "Hibrido"
        else:
            return "Presencial"

    df["Modalidade_Ensino"] = df["AnoSem"].apply(classify)

    return df


def academic_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df["Eficiencia_Academica"] = (
        df["Idade_Academica"] / df["Tempo_Permanencia_Em_Semestres"]
    )
    return df


""" Lag feature for the acdemic efficiency of the student"""


def calculate_different_academic_efficiency_lags(df: pd.DataFrame) -> pd.DataFrame:

    pula_Semestre = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])[
            "Eficiencia_Academica"
        ]
        .first()
        .reset_index()
    )
    grouped = pula_Semestre.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
    ## shift 1 - lag 1
    grouped["Eficiencia_Academica_Lag_01"] = grouped.groupby("RGA_Anon")[
        "Eficiencia_Academica"
    ].shift(1)
    grouped["Eficiencia_Academica_Lag_02"] = grouped.groupby("RGA_Anon")[
        "Eficiencia_Academica"
    ].shift(2)
    grouped["Eficiencia_Academica_Lag_03"] = grouped.groupby("RGA_Anon")[
        "Eficiencia_Academica"
    ].shift(3)

    cols_to_drop = [
        "Eficiencia_Academica_Lag_03",
        "Eficiencia_Academica_Lag_02",
        "Eficiencia_Academica_Lag_01",
    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.merge(
        grouped[
            [
                "RGA_Anon",
                "Tempo_Permanencia_Em_Semestres",
                "Eficiencia_Academica_Lag_03",
                "Eficiencia_Academica_Lag_02",
                "Eficiencia_Academica_Lag_01",
            ]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )
    return df


def calculate_rolling_failure(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    resumo = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])[
            "Reprovação_Media_Semestral"
        ]
        .first()
        .reset_index()
    )
    resumo = resumo.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])

    # rolling mean
    resumo["Rolling_Reprovacao_Media_3_Semestres"] = resumo.groupby("RGA_Anon")[
        "Reprovação_Media_Semestral"
    ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    if "Rolling_Reprovacao_Media_3_Semestres" in df.columns:
        df = df.drop(columns=["Rolling_Reprovacao_Media_3_Semestres"])

    df = df.merge(
        resumo[
            [
                "RGA_Anon",
                "Tempo_Permanencia_Em_Semestres",
                "Rolling_Reprovacao_Media_3_Semestres",
            ]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )
    return df


"""
Here we calculate if the student has fallen behind any gatekeeper subjects
"""


def apply_gatekeeper_feature(df, gatekeepers):
    # 1. Identificação individual (Linha a linha)
    df["Eh_Gatekeeper"] = df["Nome_Disciplina"].isin(gatekeepers)
    df["Reprovou_Gatekeeper_Puro"] = (
        (df["Eh_Gatekeeper"] == True) & (df["Situação"] == 1)
    ).astype(int)

    # 2. Total de falhas NESTE semestre
    # Se ele reprovou em 4, aqui teremos o número 4 para todas as linhas do semestre
    df["Qtd_Falhas_Gatekeeper_No_Semestre"] = df.groupby(
        ["RGA_Anon", "Tempo_Permanencia_Em_Semestres"]
    )["Reprovou_Gatekeeper_Puro"].transform("sum")

    # 3. Criar resumo para o cálculo acumulado (1 linha por semestre)
    resumo_semestral = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])[
            "Qtd_Falhas_Gatekeeper_No_Semestre"
        ]
        .first()
        .reset_index()
    )
    resumo_semestral = resumo_semestral.sort_values(
        ["RGA_Anon", "Tempo_Permanencia_Em_Semestres"]
    )

    # 4. Soma Acumulada Real
    # Aqui, se o Semestre 1 teve 4 falhas e o Semestre 2 teve 1, o acumulado será 5.
    resumo_semestral["Total_Falhas_Gatekeeper_Acumulado"] = resumo_semestral.groupby(
        "RGA_Anon"
    )["Qtd_Falhas_Gatekeeper_No_Semestre"].cumsum()

    # 5. Merge de volta para o DF principal
    if "Total_Falhas_Gatekeeper_Acumulado" in df.columns:
        df = df.drop(columns=["Total_Falhas_Gatekeeper_Acumulado"])

    df = df.merge(
        resumo_semestral[
            [
                "RGA_Anon",
                "Tempo_Permanencia_Em_Semestres",
                "Total_Falhas_Gatekeeper_Acumulado",
            ]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )

    # Limpeza
    df = df.drop(columns={"Eh_Gatekeeper", "Qtd_Falhas_Gatekeeper_No_Semestre"})
    return df


""" The attendance trend of a student """


def calculate_attendance_trends(df: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])["Frequencia"]
        .mean()
        .reset_index()
    )
    resumo = resumo.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])

    # 2. Lag 1 (Frequência do semestre passado)
    resumo["Frequencia_Lag_01"] = resumo.groupby("RGA_Anon")["Frequencia"].shift(1)

    # 3. Tendência (Delta): Negativo = Aluno está faltando mais
    resumo["Frequencia_Trend"] = resumo["Frequencia"] - resumo["Frequencia_Lag_01"]

    resumo["Frequencia_Rolling_3S"] = resumo.groupby("RGA_Anon")[
        "Frequencia"
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    df = df.merge(
        resumo[
            [
                "RGA_Anon",
                "Tempo_Permanencia_Em_Semestres",
                "Frequencia_Trend",
                "Frequencia_Rolling_3S",
            ]
        ],
        on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
        how="left",
    )
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    return df


def cleaning_pipeline():
    """
    This is the main function. It defines and processes datasets
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("\n\n[INFO]: Starting application...")

    try:
        logging.info("\n\n[INFO]: Loading datasets...")
        (
            df_active,
            df_deactive,
            df_history,
        ) = load_datasets(CONFIG_PATH)

        df_merged = df_history.copy()
        all_students = pd.concat([df_active, df_deactive], axis=0)

        logging.info("\n\n[OK]: Sucessfully loaded and merged the datasets")
    except Exception as e:
        logging.exception(f"[ERROR]: Could not load datasets properly{e}")

    try:
        df_merged = (
            df_merged.pipe(
                log_and_pipe, "[OK]: Creating AnoSem column", calculate_ano_sem
            )
            .pipe(
                log_and_pipe,
                "[OK]: Dropping first useless columns",
                drop_columns,
                cols=[
                    "Faltas",
                    "Codigo_Turma",
                    "Equivalencia",
                    "Codigo_Disciplina",
                    "Curso_Ofertante",
                    "Observacao",
                    "Ano",
                    "Idade_Academica",
                ],
            )
            .pipe(
                log_and_pipe,
                "[OK]: Merging all the datasets",
                merge_dfs,
                all_students,
                ["RGA_Anon", "Período ingresso", "Estrutura", "Situação atual"],
                "left",
                "RGA_Anon",
            )
            .pipe(
                log_and_pipe,
                "[OK]: Standardizing the column text",
                standardize_column_text,
                "Nome_Disciplina",
            )
            .pipe(
                log_and_pipe,
                "[OK]: Eliminated duplicates of AE and AP",
                eliminating_duplicates_ap_ae,
            )
            .pipe(
                log_and_pipe, "[OK]: Setting subject failures", setting_subject_failures
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating the failure ratio per semester",
                calculate_failure_ratio,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Selecting valid period - 2009.1 plus",
                selecting_valid_period,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Creating permance time",
                calculate_permanence_period_in_semesters,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Summing all the accumulated credits",
                calculate_total_accumulated_credits,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating academic age",
                calculate_normalized_academic_age,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating academic lag in semesters",
                calculate_academic_lag_in_semesters,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating academic lag delta",
                calculate_deltas_fixed,
            )
            .pipe(log_and_pipe, "[OK]: Calculating cofficient", calculate_coefficient)
            .pipe(
                log_and_pipe,
                "[OK]: alculating cofficient delta",
                calculate_coefficient_delta,
            )
            .pipe(log_and_pipe, "[OK]: Marking  the pandemic period", mark_pandemic)
            .pipe(
                log_and_pipe,
                "[OK]: Calculating academic efficiency",
                academic_efficiency,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating academic efficiency lags",
                calculate_different_academic_efficiency_lags,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating rolling failure of past semesters",
                calculate_rolling_failure,
                window=3,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating gatekeeper failure",
                apply_gatekeeper_feature,
                gatekeepers=[
                    "ESTRUTURA DE DADOS I",
                    "CALCULO I",
                    "CALCULO DIFERENCIAL E INTEGRAL I",
                    "GEOMETRIA ANALITICA E VETORIAL",
                    "PROGRAMACAO I",
                    "PROGRAMACAO DE COMPUTADORES",
                    "LÓGICA DIGITAL",
                    "LOGICA MATEMATICA E ELEMENTOS DE LOGICA DIGITAL",
                    "ARQUITETURA DE COMPUTADORES",
                    "FUNDAMENTOS DE MATEMATICA",
                    "MECANICA",
                    "ARQUITETURA E ORGANIZACAO DE COMPUTADORES",
                ],
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating attendance trend",
                calculate_attendance_trends,
            )
            .pipe(
                log_and_pipe,
                "[OK]: Calculating attendance trend",
                drop_columns,
                cols=[
                    "Reprovou_Gatekeeper_Puro",
                    "Frequencia",
                    "Percentual_Faltas",
                    "CH",
                    "Semestre",
                    "Falta_Excessiva",
                    "Nome_Disciplina",
                    "Nota",
                    "Situação",
                    "Qtd_Disciplinas_Semestre",
                    "MG_Semestre",
                    "Total_CH_Semestre",
                    "Crédito",
                ],
            )
        )

        df_merged = df_merged.drop_duplicates()

    except Exception as e:
        logging.exception(
            f"[ERROR]: There was an exception creating new columns in the dataset {e}"
        )
        raise
    try:
        # Intermediate dataset after cleaning the dataset

        dfs = load_config(CONFIG_PATH)
        path_to_preprocessed = dfs["PREPROCESSED_DATASET"]
        print(path_to_preprocessed)
        df_merged.to_csv(path_to_preprocessed, index=False)
        logging.info("Successfully saved preprocessed dataset")

    except Exception as e:
        logging.exception(
            f"[ERROR]: There was an exception resolving name conflics{e} "
        )
        raise


if __name__ == "__main__":
    start_time = time.time()
    cleaning_pipeline()
    total_time = time.time() - start_time
    print(f"\ntotal time taken: {total_time:.2f}s\n")
