
# Adversarial Example Generation using LLMs
This project is based on this [github repo](https://github.com/LukasStruppek/Adversarial_LLMs/tree/mai) and [paper](https://arxiv.org/pdf/2402.09132)

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

```shell
python3 craft_adv_examples.py 
```

 * If you want to run the script on a remote server and not get it interrupted when your remote connection quits, run the command with `nohup` as below:

    ```shell
    nohup python3 craft_adv_examples.py  > expt.log 2>&1 &
    ``` 