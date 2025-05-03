import torch, pathlib, argparse
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)
from optim_pi.arp_adaptive_pi import ARPAdaptivePi


def tiny_loader(tok, seq_len=256, max_samples=50_000):
    ds = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=42)
    ds = ds.select(range(max_samples))

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding="max_length",
                   max_length=seq_len)

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tok, mlm=False)
    )


def main(steps):
    name = "microsoft/bitnet-b1.58-2B-4T-bf16"
    model = (AutoModelForCausalLM
             .from_pretrained(name, torch_dtype=torch.bfloat16)
             .cuda())
    tok = AutoTokenizer.from_pretrained(name)
    dl = tiny_loader(tok)

    opt = ARPAdaptivePi(model.parameters())
    tb = SummaryWriter("runs/pilot01")

    step = 0
    for epoch in range(999):
        for batch in dl:
            step += 1
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step(); opt.zero_grad()

            if step % 20 == 0:
                tb.add_scalar("loss/train", loss.item(), step)

            if step >= steps:
                pathlib.Path("checkpoints").mkdir(exist_ok=True)
                torch.save(model.state_dict(),
                           f"checkpoints/ckpt_step{step:05d}.pt")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    main(parser.parse_args().steps)
