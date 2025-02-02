# minGPT

## Presentation

This project has been forked from Andrej Karpathy (see [project page](https://github.com/karpathy/minGPT)). The idea is to recreate the GPU implementation of GPT-2, a rather small model (125M parameters).

In this labs, the objective is to get a custom transformer running as fast as possible. For simplicity, we focus on a BS=1 use-case. This means that only one sequence is fed into the model at a time.

It first goes through context (pre-fill) with all tokens handled in parallel. Then it goes in the autoregressive mode. One token is sampled, it's fed into the network, and another new token is sampled again.

## Installation

``` bash
python3 -m venv mingpt_env
source mingpt_env/bin/activate
pip install -e .
cmake -S . -B build -G Ninja
cmake --build build

```

## Testing the LLM

A small utility `run.py` is provided to generate sequences. Remember that the model isn't trained to be a chatbot, so the results are just the most probable tokens to appear after your prompt (a bit like a smartphone's autocomplete).

``` bash
python run.py --prompt "The secret for students to pass all the exams" --steps 500
```

## Providing your own implementation

The objective of this lab is first to implement your own layers in CUDA. There are three of them:
1. gelu (activation function)
2. layernorm (normalization)
3. linear (matrix multiplication)

The implementation has to be made in `gpt.cu`. You can check your results by running:
```bash
python test.py
```

Once that's correct, you should see correct output when running 
``` bash
python run.py --prompt "The secret for students to pass all the exams" --steps 500 --use_custom_layers
```

Now you can analyze the runtime with Nsight Systems:

``` bash
nsys profile -t cuda -s none -o profile.nsys-rep -f true python run.py --prompt "The secret for students to pass all the exams" --steps 5 --use_custom_layers
```

And opening the produced file `profile.nsys-rep` with Nsight Systems.

## Optimizing the LLM

Now that you have a working implementation, you can try to optimize it. You should first look at the kernels that have the biggest runtime. You can use Nsight Compute as well to have more information on the bottlenecks of your kernels.

Another optimization you can do, at the algorithm level, is to re-use previously computed KV-cache. Look for the notes in `mingpt/model.py`.

The objective is to get the best token/s throughput!
