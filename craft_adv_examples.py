import torch
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from langchain_core.prompts import PromptTemplate
from transformers import BertForSequenceClassification
from transformers import set_seed
from Levenshtein import distance
from rtpt import RTPT
import re
import csv
import argparse

from utils import (
    get_output_file_name,
    load_csv_dataset,
    compute_hate_score,
    get_prompt_template,
    write_result,
)

from config import ExperimentParams, expt1_config, expt2_config, expt_gemma_config, expt_llama_config


def load_classifier(classifier_name):
    tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    classifier = (
        BertForSequenceClassification.from_pretrained(
            classifier_name, torch_dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    return tokenizer, classifier


def load_llm_pipeline(llm_name):
    tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )

    generation_config = GenerationConfig.from_pretrained(llm_name)
    generation_config.max_new_tokens = 500
    generation_config.temperature = 0.7
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15
    generation_config.pad_token_id = tokenizer.eos_token_id

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        return_full_text=False
    )

    pipe = HuggingFacePipeline(pipeline=text_pipeline)

    return pipe


SEED = 42
set_seed(SEED)


def run_expriment(config: ExperimentParams, split_num: int):
    # Set the split number for the dataset
    config.set_split_num(split_num)
    # parameters
    MODEL_NAME = config.model_name
    HATE_SPEECH_CLASSIFIER = config.classifier_name
    MAX_STEPS = config.max_steps
    EARLY_STOPPING_PATIENCE = config.early_stopping_patience
    MAX_CHANGES_PER_STEP = config.max_change_per_step
    INPUT_FILE_NAME = config.input_filename
    RTPT_NAME = config.rprt_name
    OUTPUT_FILE = get_output_file_name(
        model_name=MODEL_NAME,
        max_change_per_step=MAX_CHANGES_PER_STEP,
        input_filename=INPUT_FILE_NAME,
    )
    TEMPLATE = config.prompt_template

    tokenizer, classifier = load_classifier(HATE_SPEECH_CLASSIFIER)
    llm = load_llm_pipeline(MODEL_NAME)

    dataset = load_csv_dataset(filename=INPUT_FILE_NAME, delimiter="\t")
    rtpt = RTPT(RTPT_NAME, "Craft Adv. Examples", len(dataset))
    rtpt.start()

    result_dict = {
        "Initial_Sample": None,
        "Initial_Score": None,
        "Final_Sample": None,
        "Final_Score": None,
        "Distance": None,
        "Success": None,
        "Num_Steps": None,
    }

    with open(OUTPUT_FILE, "w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(result_dict.keys())

    # template = """
    # <s>[INST]{prompt}[/INST]
    # """
    # template = """{prompt}"""

    url_pattern = re.compile(r"https?://\S+|www\.\S+")

    for initial_sample in dataset:
        # remove urls from string
        initial_sample = re.sub(url_pattern, "", initial_sample).strip()
        # compute initial prediction score and generate prompt
        initial_score = compute_hate_score(tokenizer, classifier, initial_sample)
        num_steps = 0
        update_performed = True
        steps_without_update = 0
        current_sample = initial_sample
        score = initial_score
        dist = 0

        if initial_score < 0.5:
            continue
        prompt = get_prompt_template(current_sample, initial_score)
        prompt_template = PromptTemplate(
            input_variables=["prompt"],
            template=TEMPLATE,
        )

        # optimization loop
        while num_steps < MAX_STEPS:
            print("-----------------------------------")
            print(f"Number of steps: {num_steps}/{MAX_STEPS}")
            with torch.no_grad():
                # generate next sample
                print("PROMPT: \n", prompt + f"Step {num_steps + 1}:")
                output = llm.invoke(
                    prompt_template.format(prompt=prompt + f"Step {num_steps + 1}:")
                )
                print("LLM Output: ")
                print(output)
                # extract sample from model output
                try:
                    temp = re.findall(r"\|([^|]+)\|", output)[0].strip()
                    previous_sample = current_sample
                    current_sample = temp
                    print("Current sample: ", current_sample)
                except Exception as e:
                    print(f"🚩 Exception raised:\n {e}")
                    steps_without_update += 1
                    if steps_without_update > EARLY_STOPPING_PATIENCE:
                        break
                    continue
                # compute Levenshtein distance between previous and current sample
                dist = distance(previous_sample, current_sample)
                # only accept samples that are at most MAX_CHANGES_PER_STEP characters different from the previous sample
                if dist > MAX_CHANGES_PER_STEP or dist == 0:
                    update_performed = False
                    steps_without_update += 1
                    print(
                        f"🚩 Distance {dist} is greater than threshold {MAX_CHANGES_PER_STEP}. Steps without update: {steps_without_update}"
                    )
                    # stop if no update was performed in the last EARLY_STOPPING_PATIENCE steps (model is stuck)
                    if steps_without_update > EARLY_STOPPING_PATIENCE:
                        break
                    continue
                else:
                    # reset early stopping counter
                    update_performed = True
                    steps_without_update = 0

                # compute prediction score for current sample
                score = compute_hate_score(tokenizer, classifier, current_sample)
                print(
                    f'SAMPLE {num_steps + 1}: "',
                    current_sample,
                    '", SCORE:',
                    score,
                    ", DISTANCE:",
                    dist,
                )

                # add sample to prompt for LLM input
                num_steps += 1
                prompt += f"Step {num_steps}: |{current_sample}| - Prediction Score: {score:.4f};\n"
                previous_sample = current_sample

                # stop if prediction score is below 0.5
                if score < 0.5 or num_steps == MAX_STEPS:
                    break
                print("-----------------------------------")

        print("Final result", current_sample, score, dist)
        result_dict["Initial_Sample"] = initial_sample
        result_dict["Initial_Score"] = initial_score
        result_dict["Final_Sample"] = current_sample
        result_dict["Final_Score"] = score
        result_dict["Distance"] = distance(initial_sample, current_sample)
        result_dict["Success"] = score < 0.5
        result_dict["Num_Steps"] = num_steps
        write_result(result_dict, OUTPUT_FILE)
        rtpt.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments to build the config for the experiment")
    parser.add_argument("--split", type=int, required=True, help="The split of the dataset to use")
    args = parser.parse_args()

    # run_expriment(config=expt1_config, split_num=args.split) # For Mistral
    # run_expriment(config=expt_gemma_config, split_num=args.split)
    # run_expriment(config=expt2_config, split_num=args.split)
    run_expriment(config=expt_llama_config, split_num=args.split)
