
# Adversarial Example Generation using LLMs
This project is based on this [github repo](https://github.com/LukasStruppek/Adversarial_LLMs/tree/mai) and [paper](https://arxiv.org/pdf/2402.09132)

## Creating Environment
Create a virtual Python Environment
```shell
sh ./setup.sh
```

## Run Experiments
Run the python script to generate the adversarial examples in the `results` folder.

```shell
python3 craft_adv_examples.py 
```

If you want to run the script on a remote server and not get it interrupted when your remote connection quits, run the command with `nohup` as below:
```shell
nohup python3 craft_adv_examples.py  > expt.log 2>&1 &
```