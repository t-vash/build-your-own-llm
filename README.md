# build-your-own-llm
Hands-on workshop notebook: build a byte-level BPE tokenizer, a GPT-2-style Transformer (MiniGPT2) in PyTorch, LoRA fine-tuning of DistilGPT2, a RAG pipeline with ChromaDB + LangChain, and a Gradio app. Taught to 71 students.

# Build Your Own LLM: Pretrain, Fine-tune, and Deploy

A full-day hands-on workshop I taught to 71 students as Tech Master of the
Salesforce Campus Club, Amrita Vishwa Vidyapeetham, Bengaluru. Everything runs
in a single Google Colab notebook on a free GPU (or a standard laptop).

## What you build

| Module | Topic | What's in the notebook |
|---|---|---|
| 1 | Subword tokenization | Train a byte-level BPE tokenizer (8k vocab), compare with GPT-2 |
| 2 | Datasets and DataLoaders | Sliding-window next-token dataset, padding, attention masks, the `-100` label trick |
| 3 | Causal self-attention | Single-head and multi-head attention from scratch, causal masking |
| 4 | Pretraining | MiniGPT2 (6 layers, 6 heads, 192-dim, 128 context, tied embeddings), training loop, loss curves, temperature sampling |
| 5 | LoRA fine-tuning | Instruction-tune DistilGPT2 with PEFT (r=32, alpha=64) on Alpaca-format data |
| 6 | RAG | Chunking, MiniLM embeddings, ChromaDB retrieval, grounded prompting with LangChain |
| 7 | Deployment | A three-tab Gradio app (tokenizer, generation, RAG) with a public share link |

## Run it

1. Open `build_your_own_llm.ipynb` in Google Colab.
2. Set the runtime to GPU (Runtime → Change runtime type).
3. Run the cells top to bottom.

**Stack:** PyTorch, Hugging Face (transformers, tokenizers, datasets, peft),
LangChain, ChromaDB, sentence-transformers, Gradio.

## Limitations

MiniGPT2 is trained on a short text corpus, so it learns fluent-looking
English but no real knowledge. It is a teaching model, not a useful LLM.
Evaluation is validation loss plus qualitative samples, with no benchmark suite.

## Credits

The architecture and some data follow the public
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) and
[LLM-workshop-2024](https://github.com/rasbt/LLM-workshop-2024) material by
Sebastian Raschka. I adapted, assembled and extended it for this workshop.

## Author

Tvesha Reddy, [LinkedIn](https://linkedin.com/in/tvesha-reddy)
