from dataclasses import dataclass
import numpy as np


@dataclass
class ExperimentParams:
    model_name: str
    classifier_name: str
    max_steps: int
    early_stopping_patience: int
    max_change_per_step: float
    rprt_name: str
    input_filename: str = "data-splitted/split_{}.tsv"  # will be dynamically updated
    prompt_template: str = """{prompt}"""

    def set_split_num(self, split_num: int):
        self.input_filename = self.input_filename.format(split_num)


expt1_config = ExperimentParams(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    classifier_name="Hate-speech-CNERG/dehatebert-mono-english",
    max_steps=50,
    early_stopping_patience=25,
    max_change_per_step=np.inf,
    rprt_name="XX",
    prompt_template="""<s>[INST]{prompt}[/INST]""",
)


expt2_config = ExperimentParams(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    classifier_name="Hate-speech-CNERG/dehatebert-mono-english",
    max_steps=50,
    early_stopping_patience=25,
    max_change_per_step=10,
    rprt_name="XX",
)


expt_gemma_config = ExperimentParams(
    model_name="google/gemma-2b-it",
    classifier_name="Hate-speech-CNERG/dehatebert-mono-english",
    max_steps=50,
    early_stopping_patience=25,
    # max_change_per_step=np.inf,
    max_change_per_step=10,
    rprt_name="XX",
)

expt_llama_config = ExperimentParams(
    model_name="meta-llama/Llama-3.1-8b",
    classifier_name="Hate-speech-CNERG/dehatebert-mono-english",
    max_steps=50,
    early_stopping_patience=25,
    max_change_per_step=10,
    rprt_name="XX",
)
