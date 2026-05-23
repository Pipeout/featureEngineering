import logging
import os
import time

import numpy as np
import pandas as pd
import yaml
from cleaning import load_datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "../configs/cleaning.yaml")


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


def _(df_all_students: pd.DataFrame) -> pd.DataFrame:

    df_all_students = df_all_students.rename(columns={"Período": "Periodo_Atual"})

    df_all_students["Periodo_Atual"] = df_all_students["Periodo_Atual"] / 10

    df_all_students["Estrutura"] = df_all_students["Estrutura"].astype("int")

    df_all_students["Data Nascimento"] = pd.to_datetime(
        df_all_students["Data Nascimento"]
    )

    df_all_students["Período ingresso"] = df_all_students["Período ingresso"] / 10

    df_all_students["Ano_Ingresso"] = np.floor(df_all_students["Período ingresso"])

    df_all_students["Idade_Ingresso"] = (
        df_all_students["Ano_Ingresso"] - df_all_students["Data Nascimento"].dt.year
    )

    df_all_students = df_all_students.drop(
        columns={
            "Coeficiente",
            "Estrangeiro",
            "Nacionalidade",
            "Estado Civil",
            "Data Nascimento",
            "Data ocorrência",
            "Ano_Ingresso",
        }
    )

    return df_all_students


def selection_pipeline():
    logging.basicConfig(level=logging.INFO)

    config = load_config(CONFIG_PATH)

    df_preprocessed = pd.read_csv(config["PREPROCESSED_DATASET"])
    logging.info("\n\n[INFO]: Starting to drop useless columns...")

    try:
        (
            df_active,
            df_deactive,
            df_history,
        ) = load_datasets(CONFIG_PATH)

        all_students = pd.concat([df_active, df_deactive], axis=0)

        all_students = _(all_students)

        df_full = df_preprocessed.merge(
            all_students[
                [
                    "RGA_Anon",
                    "Período ingresso",
                    "Estrutura",
                    "Sexo",
                    "Raça",
                    "Periodo_Atual",
                    "Tipo ingresso",
                    "IMI",
                    "Tipo de demanda",
                    "Idade_Ingresso",
                ]
            ],
            on=["RGA_Anon", "Período ingresso", "Estrutura"],
            how="left",
        )

        df_full.info()

    except Exception as e:
        logging.exception({e})

    df_full.to_csv(config["TRAINING_DATASET"], index=False)
    logging.info("\n\n[OK]: Final training dataset saved to Bucket...")


if __name__ == "__main__":
    start_time = time.time()
    selection_pipeline()
    total_time = time.time() - start_time
    print(f"Total time taken:{total_time:.2f}s\n")
