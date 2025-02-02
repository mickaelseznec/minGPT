import torch
import numpy as np
from argparse import ArgumentParser
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

set_seed(29)

def generate(prompt='',steps=20, use_custom_layers=False):
    model_type = 'gpt2'
    device = torch.device('cuda')
    model = GPT.from_pretrained(model_type)
    model.to(device)
    model.eval()

    # tokenize the input prompt into integer input sequence
    tokenizer = BPETokenizer()
    if prompt == '':
        x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long, device=device)
    else:
        x = tokenizer(prompt).to(device)

    # forward the model `steps` times to get samples, in a batch
    prompt_length = x.size(1)

    for _ in range(2): # warmups
        model(x)

    y, timings = model.generate(x, max_new_tokens=steps, do_sample=True, top_k=8, use_custom_layers=use_custom_layers)

    # detokenize the output tokens into text
    out = tokenizer.decode(y[0][prompt_length:].cpu().squeeze())
    median = 1000 * np.median(timings[1:])
    print(f"Prompt:\n'''{prompt}'''\nGeneration:\n'''{out}'''\nTime To First Token (TFTT): {1000*timings[0]:.2f}ms\nTime Per Output Tokens (TPOT): {median:.2f}ms,  {1000/median:.2f}TPS")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--use_custom_layers', action="store_true", default=False)
    args = parser.parse_args()
    generate(args.prompt, args.steps, args.use_custom_layers)
