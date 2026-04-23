from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset


tokenizer = AutoTokenizer.from_pretrained("/Users/taoji/Github/assignment-multilingual-bpe-tokenizer/my_zh_bpe_tokenizer")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config = AutoConfig.from_pretrained("gpt2", vocab_size=2000, n_embd=32, n_layer=2, n_head=4)
my_gpt2 = AutoModelForCausalLM.from_config(config)

# Training loop
ds = load_dataset("AmazonScience/massive", "zh-CN", split="train", trust_remote_code=True)

def preprocess(example):
    out = tokenizer(
        example["utt"],
        truncation=True,
        padding="max_length",
        max_length=64
    )
    out["labels"] = out["input_ids"].copy()
    return out

ds = ds.map(preprocess)
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# 使用huggingface的Trainer API
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
)
trainer = Trainer(
    model=my_gpt2,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)
trainer.train()