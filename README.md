
# Adversarial Example Generation using LLMs
This project is based on this [github repo](https://github.com/LukasStruppek/Adversarial_LLMs/tree/main) and [paper](https://arxiv.org/pdf/2402.09132)

## Creating Environment
Create a virtual Python Environment
```shell
sh ./setup.sh
```

## Run Experiments
1. Create a token in your huggingface account. Follow the instruction [here](https://huggingface.co/docs/hub/en/security-tokens).
    * Make sure you add permission to access the [repostiory of the LLM](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) when creating the token.
2. Set the token as an environment variable.

    ```shell
    huggingface-cli login
    ```
    This command will prompt you to add your token to your environment.

3. Run the python script to generate the adversarial examples in the `results` folder.

    In the `craft_adv_examples.py` file, make sure you reference (defined in the `config.py` file) the right configuration for the experiment you want to execute.

    ```shell
    python3 craft_adv_examples.py --split=1
    ```

    * If you want to run the script on a remote server and not get it interrupted when your remote connection quits, run the command with `nohup` as below:

    ```shell
    nohup python3 craft_adv_examples.py --split=1 > expt.log 2>&1 &
    ``` 

4. Muti-GPU inference support

    The file `craft_adv_examples_multi_gpu.py` supports inference on multiple GPUs. Assuming your have `4` GPUs in your environment, run the following command:

    ```shell
    torchrun --nproc-per-node 4 craft_adv_examples_multi_gpu.py  --split=1
    ```
> [!IMPORTANT]  
> Modify the number of GPU nodes available (--nproc-per-node 4)

5. Generate summary statistics

    Run the following command to generate the summary statistics and visualization of the results of the experiments in the `exp_stat` folder:

    ```shell
    python3 expt_summary.py
    ```