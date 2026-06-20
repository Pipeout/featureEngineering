import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# error handlers
class ErrorLoadingDatasets(Exception):
    pass


class ErrorReadingConfigFile(Exception):
    pass


class MissingColumnInDataFrame(Exception):
    pass


class RequiredKeyNotUnique(Exception):
    pass


class LoadConfig:
    @staticmethod
    def get_config_file(filename: str) -> Path:
        try:
            base_dir = Path(__file__).resolve().parent.parent
            path = base_dir / "configs" / filename
            return path
        except NameError:  # if it is a jupyter file
            return Path("/cleaning-app/configs/cleaning.yaml")

    @staticmethod
    def load_config(CONFIG_PATH: Path) -> Any:
        """
        Selects the current dataset's config file we are interested in.
        """
        with open(CONFIG_PATH, "r") as f:
            full_config = yaml.safe_load(f)

        try:
            current_dataset = full_config["CURRENT_DATASET"]
            logging.info(f"\nloading current dataset: {current_dataset}")
            if current_dataset not in full_config["DATASETS"]:
                raise ValueError(f"\nDataset {current_dataset} not found!")

            return full_config["DATASETS"][current_dataset]

        except Exception:
            raise ErrorReadingConfigFile(
                "Error while reading config file. Check its path and correctness"
            )

    def load_datasets(self, CONFIG_PATH: Path) -> tuple[str, str, str, str, str]:
        """
        Loads the configuration paths for datasets
        """
        dfs = self.load_config(CONFIG_PATH)
        period_range = dfs.get("PERIOD_RANGE", "")
        preprocessed_active = dfs["PREPROCESSED_ACTIVE_DATASET"]
        preprocessed_deactive = dfs["PREPROCESSED_DEACTIVE_DATASET"]
        preprocessed_history = dfs["PREPROCESSED_HISTORY_DATASET"]
        training_dataset = dfs.get("TRAINING_DATASET", "training.csv")

        return (
            period_range,
            preprocessed_active,
            preprocessed_deactive,
            preprocessed_history,
            training_dataset,
        )

    def run(self) -> tuple[str, str, str, str, str]:
        CONFIG_PATH = self.get_config_file("cleaning.yaml")
        logging.info("\n\n[INFO]: Loading configuration...")

        return self.load_datasets(CONFIG_PATH)


class BuildingFeatures:
    def __init__(self):
        lc = LoadConfig()
        (
            self.period_range,
            self.active_path,
            self.deactive_path,
            self.history_path,
            self.training_path,
        ) = lc.run()

        # Load preprocessed datasets
        self.df_active = pd.read_csv(self.active_path)
        self.df_deactive = pd.read_csv(self.deactive_path)
        self.df_history = pd.read_csv(self.history_path)

        self.gatekeepers = [
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
        ]

    def calculate_ano_sem(self) -> None:
        df = self.df_history
        df["Ano"] = df["AnoSem"].astype("int")
        df["Parcela"] = df["Semestre"] / 10
        df["AnoSem"] = df["Ano"] + df["Parcela"]
        self.df_history = df.drop(columns=["Parcela", "Ano"])

    def calculate_failure_ratio(self) -> None:
        df = self.df_history
        df["Reprovacao_Ponderada_Semestral"] = df["Crédito"] * df["Situação"]
        df["Reprovacao_Ponderada_Semestral"] = df.groupby(["AnoSem", "RGA_Anon"])[
            "Reprovacao_Ponderada_Semestral"
        ].transform("sum")

        total_credit = (
            df.groupby(["AnoSem", "RGA_Anon"])["Crédito"]
            .transform("sum")
            .astype("float")
        )
        df["Reprovação_Media_Semestral"] = (
            df["Reprovacao_Ponderada_Semestral"].astype("float") / total_credit
        )

        self.df_history = df.drop(columns=["Reprovacao_Ponderada_Semestral"])

    def calculate_permanence_period(self) -> None:
        df = self.df_history
        if "Período ingresso" in df.columns:
            df["Período ingresso"] = df["Período ingresso"].astype(float) / 10

            def cap_sem(x):
                int_part = int(x)
                frac_part = min(int(round((x - int_part) * 10)), 2)
                return float(f"{int_part}.{frac_part}")

            df["AnoSemCap"] = df["AnoSem"].apply(cap_sem)
            df["Período ingresso Cap"] = df["Período ingresso"].apply(cap_sem)

            all_values = (
                pd.concat([df["AnoSemCap"], df["Período ingresso Cap"]])
                .dropna()
                .unique()
            )
            mapping = {val: i + 1 for i, val in enumerate(sorted(all_values))}

            df["Tempo_Permanencia_Em_Semestres"] = (
                df["AnoSemCap"].map(mapping)
                - df["Período ingresso Cap"].map(mapping)
                + 1
            )
            self.df_history = df.drop(columns=["AnoSemCap", "Período ingresso Cap"])

    def calculate_total_accumulated_credits(self) -> None:
        mapping = {20241: 210, 20191: 200, 20091: 211}
        df = self.df_history

        if "Estrutura" in df.columns:
            df["Total_creditos_estrutura"] = df["Estrutura"].map(mapping)

        if "Tempo_Permanencia_Em_Semestres" in df.columns:
            resumo = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])["Crédito"]
                .sum()
                .reset_index()
            )
            resumo["Total_Creditos_Acumulados"] = resumo.groupby("RGA_Anon")[
                "Crédito"
            ].cumsum()
            self.df_history = df.merge(
                resumo[
                    [
                        "RGA_Anon",
                        "Tempo_Permanencia_Em_Semestres",
                        "Total_Creditos_Acumulados",
                    ]
                ],
                on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
                how="left",
            )

    def calculate_normalized_academic_age(self) -> None:
        metas = {
            20091: {"min_credits": 211, "ideal_semesters": 8},
            20191: {"min_credits": 200, "ideal_semesters": 8},
            20241: {"min_credits": 210, "ideal_semesters": 8},
        }
        df = self.df_history

        def get_age(row):
            struct = row.get("Estrutura")
            if struct not in metas:
                return row.get("Tempo_Permanencia_Em_Semestres", 0)
            meta = metas[struct]
            progresso = min(
                1.0, row.get("Total_Creditos_Acumulados", 0) / meta["min_credits"]
            )
            return progresso * meta["ideal_semesters"]

        if "Estrutura" in df.columns and "Total_Creditos_Acumulados" in df.columns:
            df["Idade_Academica"] = df.apply(get_age, axis=1)
            df["Estrutura"] = df["Estrutura"].astype(int)
        self.df_history = df

    def calculate_academic_lag(self) -> None:
        df = self.df_history
        if (
            "Tempo_Permanencia_Em_Semestres" in df.columns
            and "Idade_Academica" in df.columns
        ):
            df = df.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
            df["Lag_Academico_Em_Semestres"] = (
                df.groupby("RGA_Anon")
                .apply(
                    lambda g: (
                        g["Tempo_Permanencia_Em_Semestres"].astype(float)
                        - g["Idade_Academica"].astype(float)
                    )
                )
                .reset_index(level=0, drop=True)
            )
        self.df_history = df

    def calculate_academic_lag_delta(self) -> None:
        df = self.df_history
        if "Lag_Academico_Em_Semestres" in df.columns:
            df = df.sort_values(by=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
            sem_snapshot = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
                .agg({"Lag_Academico_Em_Semestres": "first"})
                .reset_index()
            )
            sem_snapshot["Lag_Academico_Delta"] = (
                sem_snapshot.groupby("RGA_Anon")["Lag_Academico_Em_Semestres"]
                .diff()
                .fillna(0)
            )
            self.df_history = df.merge(
                sem_snapshot[
                    [
                        "RGA_Anon",
                        "Tempo_Permanencia_Em_Semestres",
                        "Lag_Academico_Delta",
                    ]
                ],
                on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
                how="left",
            )

    def calculate_coefficient(self) -> None:
        df = self.df_history
        if "Tempo_Permanencia_Em_Semestres" in df.columns:
            df = df.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
            df["NC_Materia"] = df["Nota"] * df["Crédito"]

            resumo = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
                .agg({"NC_Materia": "sum", "Crédito": "sum"})
                .reset_index()
            )
            resumo["NC_Acumulado"] = resumo.groupby("RGA_Anon")["NC_Materia"].cumsum()
            resumo["Creditos_Acumulados"] = resumo.groupby("RGA_Anon")[
                "Crédito"
            ].cumsum()
            resumo["Coeficiente_Rendimento"] = (
                resumo["NC_Acumulado"] / resumo["Creditos_Acumulados"]
            )

            self.df_history = df.merge(
                resumo[
                    [
                        "RGA_Anon",
                        "Tempo_Permanencia_Em_Semestres",
                        "Coeficiente_Rendimento",
                    ]
                ],
                on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
                how="left",
            ).drop(columns=["NC_Materia"])

    def calculate_coefficient_delta(self) -> None:
        df = self.df_history
        if "Coeficiente_Rendimento" in df.columns:
            df = df.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
            snapshot = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
                .first()
                .reset_index()
            )
            snapshot["Coeficiente_Rendimento_Delta"] = (
                snapshot.groupby("RGA_Anon")["Coeficiente_Rendimento"].diff().fillna(0)
            )
            self.df_history = df.merge(
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

    def mark_pandemic(self) -> None:
        def classify(x):
            if 2020.1 <= x <= 2021.2:
                return "Remoto"
            elif 2022.1 <= x <= 2022.2:
                return "Hibrido"
            return "Presencial"

        self.df_history["Modalidade_Ensino"] = self.df_history["AnoSem"].apply(classify)

    def calculate_academic_efficiency(self) -> None:
        df = self.df_history
        if (
            "Idade_Academica" in df.columns
            and "Tempo_Permanencia_Em_Semestres" in df.columns
        ):
            df["Eficiencia_Academica"] = (
                df["Idade_Academica"] / df["Tempo_Permanencia_Em_Semestres"]
            )
        self.df_history = df

    def calculate_academic_efficiency_lags(self) -> None:
        df = self.df_history
        if "Eficiencia_Academica" in df.columns:
            pula_semestre = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])[
                    "Eficiencia_Academica"
                ]
                .first()
                .reset_index()
            )
            grouped = pula_semestre.sort_values(
                ["RGA_Anon", "Tempo_Permanencia_Em_Semestres"]
            )
            grouped["Eficiencia_Academica_Lag_01"] = grouped.groupby("RGA_Anon")[
                "Eficiencia_Academica"
            ].shift(1)
            grouped["Eficiencia_Academica_Lag_02"] = grouped.groupby("RGA_Anon")[
                "Eficiencia_Academica"
            ].shift(2)
            grouped["Eficiencia_Academica_Lag_03"] = grouped.groupby("RGA_Anon")[
                "Eficiencia_Academica"
            ].shift(3)

            self.df_history = df.merge(
                grouped[
                    [
                        "RGA_Anon",
                        "Tempo_Permanencia_Em_Semestres",
                        "Eficiencia_Academica_Lag_01",
                        "Eficiencia_Academica_Lag_02",
                        "Eficiencia_Academica_Lag_03",
                    ]
                ],
                on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
                how="left",
            )

    def calculate_rolling_failure(self, window: int = 3) -> None:
        df = self.df_history
        if "Reprovação_Media_Semestral" in df.columns:
            resumo = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])[
                    "Reprovação_Media_Semestral"
                ]
                .first()
                .reset_index()
            )
            resumo = resumo.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
            resumo["Rolling_Reprovacao_Media_3_Semestres"] = resumo.groupby("RGA_Anon")[
                "Reprovação_Media_Semestral"
            ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            self.df_history = df.merge(
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

    def calculate_attendance_trends(self) -> None:
        df = self.df_history
        if "Frequencia" in df.columns:
            resumo = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])["Frequencia"]
                .mean()
                .reset_index()
            )
            resumo = resumo.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
            resumo["Frequencia_Lag_01"] = resumo.groupby("RGA_Anon")[
                "Frequencia"
            ].shift(1)
            resumo["Frequencia_Trend"] = (
                resumo["Frequencia"] - resumo["Frequencia_Lag_01"]
            )
            resumo["Frequencia_Rolling_3S"] = resumo.groupby("RGA_Anon")[
                "Frequencia"
            ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            self.df_history = df.merge(
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

    def calculate_admission_age(self) -> None:
        df = self.df_history

        if pd.api.types.is_datetime64_any_dtype(df["Data Nascimento"]):
            df["Ano_Ingresso"] = np.floor(df["Período ingresso"])
            df["Idade_Ingresso"] = df["Ano_Ingresso"] - df["Data Nascimento"].dt.year
        self.df_history = df

    def calculate_age_per_semester(self) -> None:
        df = self.df_history

        if "AnoSem" not in df.columns or "Data Nascimento" not in df.columns:
            logging.warning(
                "Missing 'AnoSem' or 'Data Nascimento'. Cannot calculate dynamic age."
            )
            return

        df["Ano_Atual_Semestre"] = np.floor(df["AnoSem"])
        df["Idade_No_Semestre"] = (
            df["Ano_Atual_Semestre"] - df["Data Nascimento"].dt.year
        )
        self.df_history = df.drop(columns=["Ano_Atual_Semestre"], errors="ignore")

    def apply_gatekeeper_feature(self) -> None:
        df = self.df_history
        if "Tempo_Permanencia_Em_Semestres" in df.columns:
            df["Eh_Gatekeeper"] = df["Nome_Disciplina"].isin(self.gatekeepers)
            df["Reprovou_Gatekeeper_Puro"] = (
                (df["Eh_Gatekeeper"] == True) & (df["Situação"] == 1)
            ).astype(int)

            df["Qtd_Falhas_Gatekeeper_No_Semestre"] = df.groupby(
                ["RGA_Anon", "Tempo_Permanencia_Em_Semestres"]
            )["Reprovou_Gatekeeper_Puro"].transform("sum")

            resumo = (
                df.groupby(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])[
                    "Qtd_Falhas_Gatekeeper_No_Semestre"
                ]
                .first()
                .reset_index()
            )
            resumo = resumo.sort_values(["RGA_Anon", "Tempo_Permanencia_Em_Semestres"])
            resumo["Total_Falhas_Gatekeeper_Acumulado"] = resumo.groupby("RGA_Anon")[
                "Qtd_Falhas_Gatekeeper_No_Semestre"
            ].cumsum()

            self.df_history = df.merge(
                resumo[
                    [
                        "RGA_Anon",
                        "Tempo_Permanencia_Em_Semestres",
                        "Total_Falhas_Gatekeeper_Acumulado",
                    ]
                ],
                on=["RGA_Anon", "Tempo_Permanencia_Em_Semestres"],
                how="left",
            ).drop(
                columns=[
                    "Eh_Gatekeeper",
                    "Qtd_Falhas_Gatekeeper_No_Semestre",
                    "Reprovou_Gatekeeper_Puro",
                ]
            )

    def concat_df_active_deative(self) -> pd.DataFrame:
        allstudents = pd.concat([self.df_active, self.df_deactive], axis=0)
        return allstudents

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logging.info("[INFO]: Building features...")
        self.calculate_ano_sem()
        self.calculate_failure_ratio()

        all_demographics = self.concat_df_active_deative()

        # Parse dates early so calculations relying on them don't fail
        all_demographics["Data Nascimento"] = pd.to_datetime(
            all_demographics["Data Nascimento"], errors="coerce"
        )

        self.df_history = self.df_history.merge(
            all_demographics[
                [
                    "RGA_Anon",
                    "Período ingresso",
                    "Estrutura",
                    "Situação atual",
                    "Data Nascimento",
                    "Sexo",
                    "Raça",
                    "Tipo ingresso",
                    "IMI",
                    "Tipo de demanda",
                ]
            ],
            on="RGA_Anon",
            how="left",
        )

        # Base Features
        self.calculate_permanence_period()
        self.calculate_total_accumulated_credits()
        self.calculate_normalized_academic_age()
        self.calculate_academic_lag()
        self.calculate_coefficient()
        self.mark_pandemic()
        self.calculate_academic_efficiency()
        self.calculate_admission_age()
        self.calculate_age_per_semester()
        self.apply_gatekeeper_feature()
        self.calculate_attendance_trends()

        # Delta & Windowed Features (must run after Base Features)
        self.calculate_academic_lag_delta()
        self.calculate_coefficient_delta()
        self.calculate_academic_efficiency_lags()
        self.calculate_rolling_failure(window=3)

        return self.df_history, all_demographics


class SelectingFeatures:
    def __init__(
        self,
        df_engineered: pd.DataFrame,
        training_path: str,
    ):
        self.df_engineered = df_engineered
        self.training_path = training_path

    def compile_final_dataset(self) -> None:
        logging.info("[INFO]: Compiling final dataset...")

        cols_to_drop_hist = [
            "Faltas",
            "Codigo_Turma",
            "Equivalencia",
            "Codigo_Disciplina",
            "Curso_Ofertante",
            "Observacao",
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
            "Data Nascimento",  # Dropped because we now have the engineered age integers
        ]

        df_clean_hist = self.df_engineered.drop(
            columns=[c for c in cols_to_drop_hist if c in self.df_engineered.columns]
        ).drop_duplicates()

        df_clean_hist.to_csv(self.training_path, index=False)
        logging.info(f"[OK]: Final training dataset saved to {self.training_path}")

    def run(self) -> None:
        self.compile_final_dataset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("\nStarting application...")

    try:
        # Initialize configuration to get paths
        lc = LoadConfig()
        _, _, _, _, training_path = lc.run()

        # Build Features
        builder = BuildingFeatures()
        df_engineered_history, _ = builder.run()

        # Select Features & Compile
        selector = SelectingFeatures(df_engineered_history, training_path)
        selector.run()

    except Exception as e:
        logging.exception(f"[ERROR]: Pipeline failed: {e}")
        raise
