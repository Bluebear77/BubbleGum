"""
--------------------------------------------------------------------------------
text-to-sparql batch generation pipeline
--------------------------------------------------------------------------------
This script reads natural language questions from CSV files, uses a Hugging Face T5
model fine‑tuned for English-to-SPARQL translation (model: yazdipour/text-to-sparql-t5-small-qald9),
and generates SPARQL queries suitable for querying knowledge graphs (e.g. Wikidata or DBpedia).

Details:
- The model is a T5-small variant fine‑tuned on the QALD‑9 dataset (with some LC‑QuAD
  mixed in) for semantic parsing tasks, using prefix `"translate english to sparql:"` :contentReference[oaicite:1]{index=1}.
- It outputs SPARQL text that can be executed against a public SPARQL endpoint, such as
  Wikidata or DBpedia, depending on downstream entity linking and prefix definitions.
- Training setup: learning_rate 0.0003, batch_size 8, single epoch, Adam optimizer,
  mixed‑precision AMP; framework: Transformers 4.11 / PyTorch 1.9.0 :contentReference[oaicite:2]{index=2}.
- The model has ~60M parameters and uses Apache‑2.0 license :contentReference[oaicite:3]{index=3}.

Purpose:
- Automates bulk conversion of questions into SPARQL.
- Supports downstream QA or batch SPARQL query generation for evaluation, KB querying,
  and prototyping semantic parsing pipelines.

Usage:
- Reads all `.csv` files in the designated input folder, each expected to contain a `question`
  column.
- Iterates through questions, prefixes with the required task instruction, tokenizes, and
  generates a SPARQL query per input.
- Saves results into an output CSV with columns `question` and `sparql`, preserving filenames.

Prerequisites:
- `transformers`, `torch`, `pandas`, and `tqdm` installed (aligned with version in repo).
- Access to model weights: downloaded from HuggingFace repo `yazdipour/text-to-sparql-t5-small-qald9`.
- If executing queries: a SPARQL endpoint and optional entity linker to convert entity labels to
  KB identifiers.

--------------------------------------------------------------------------------
"""





import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from pathlib import Path

# 🧠 Use CPU
device = torch.device('cpu')

# 🔧 Load SPARQL-trained model
model_name = 'yazdipour/text-to-sparql-t5-small-qald9'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()

# 🔁 SPARQL generation function
def generate_sparql(question):
    prompt = f"translate english to sparql: {question}"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 📥 Load all CSV files from the QASd directory
input_dir = Path('./QASd')
csv_files = list(input_dir.glob('*.csv'))

# 🚨 Check for CSV files
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")

# 🔄 Process each CSV file
for csv_file in csv_files:
    print(f"\n📄 Processing: {csv_file.name}")
    df = pd.read_csv(csv_file)
    if 'question' not in df.columns:
        print(f"⚠️ Skipping {csv_file.name}: No 'question' column found.")
        continue

    questions = df['question'].dropna().tolist()

    outputs = []
    for q in tqdm(questions, desc=f"Generating SPARQL for {csv_file.name}"):
        sparql = generate_sparql(q)
        outputs.append({'question': q, 'sparql': sparql})
        print(f"\nQ: {q}\nSPARQL: {sparql}\n")

    # 💾 Save results in 'SPARQL' folder with same filename
    output_dir = input_dir.parent / 'SPARQL'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / csv_file.name
    df_out = pd.DataFrame(outputs)
    df_out.to_csv(output_file_path, index=False)

    print(f"✅ Output saved to {output_file_path}")
