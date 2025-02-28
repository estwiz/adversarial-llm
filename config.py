from dataclasses import dataclass
import numpy as np

@dataclass
class ExperimentParams:
    model_name: str
    classifier_name: str
    max_steps: int
    early_stopping_patience: int
    max_change_per_step: float
    input_filename: str
    rprt_name: str


expt1_config = ExperimentParams(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    classifier_name="Hate-speech-CNERG/dehatebert-mono-english",
    max_steps=50,
    early_stopping_patience=25,
    max_change_per_step=np.inf,
    input_filename="./data-splitted/split_2.tsv",
    rprt_name="XX"
)


expt2_config = ExperimentParams(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    classifier_name="Hate-speech-CNERG/dehatebert-mono-english",
    max_steps=50,
    early_stopping_patience=25,
    max_change_per_step=10,
    input_filename="./data-splitted/split_1.tsv",
    rprt_name="XX"
)


expt_gemma_config = ExperimentParams(
    model_name="google/gemma-2b-it",
    classifier_name="Hate-speech-CNERG/dehatebert-mono-english",
    max_steps=50,
    early_stopping_patience=25,
    max_change_per_step=np.inf,
    input_filename="./data-splitted/split_1.tsv",
    rprt_name="XX"
)