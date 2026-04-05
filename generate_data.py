import json
import os
import random
import re

from faker import Faker
from vllm import LLM, SamplingParams

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

random.seed(42)
fake = Faker()
Faker.seed(42)

with open("data/prompts.json") as f:
    prompts = json.load(f)

ROUNDS = 3

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", seed=42, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1024)


"""## Stage 1 - Placeholder Generation"""

def build_chat(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

stage1_conversations = []
for _ in range(ROUNDS):
    for domain in prompts["stage_1"]["domains"]:
        stage1_conversations.append(
            build_chat(prompts["stage_1"]["system"], domain)
        )

stage1_outputs = llm.chat(stage1_conversations, sampling_params)

placeholder_sentences = []
for output in stage1_outputs:
    text = output.outputs[0].text.strip()
    for line in text.split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if "<MONEY>" in line and len(line) > 10:
            placeholder_sentences.append(line)

print(f"Stage 1 produced {len(placeholder_sentences)} sentences")


"""## Stage 2 - Distractor Enrichment"""

stage2_conversations = [
    build_chat(prompts["stage_2"]["system"], s)
    for s in placeholder_sentences
]

stage2_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
stage2_outputs = llm.chat(stage2_conversations, stage2_params)

enriched_sentences = []
for orig, output in zip(placeholder_sentences, stage2_outputs):
    rewritten = output.outputs[0].text.strip().split("\n")[0].strip()
    if "<MONEY>" in rewritten and len(rewritten) > 10:
        enriched_sentences.append(rewritten)
    else:
        enriched_sentences.append(orig)

print(f"Stage 2 kept {len(enriched_sentences)} sentences")


"""## Stage 3 - Faker Replacement"""

SYMBOLS = ["$", "€", "£", "¥", "₹"]
ISO_CODES = [
    "USD", "EUR", "GBP", "JPY", "AUD", "CHF", "CAD", "NZD", "INR", "CNY",
    "BRL", "MXN", "KRW", "SEK", "NOK", "DKK", "SGD", "HKD", "PLN", "CZK",
    "THB", "ZAR", "TRY", "ILS", "PHP", "MYR", "IDR", "AED", "SAR", "ETB",
    "TOP", "FJD", "BBD", "BZD", "BSD", "NGN", "KES", "GHS", "CLP", "COP",
]
MAGNITUDE_SUFFIXES = ["K", "M", "B"]
WRITTEN_MAGNITUDES = ["thousand", "million", "billion"]
CURRENCY_WORDS = ["dollars", "euros", "pounds", "yen"]


def _fmt_number(n: float, decimals: bool = True) -> str:
    if decimals and random.random() < 0.5:
        return f"{n:,.2f}"
    return f"{int(n):,}"


def generate_money_string() -> str:
    fmt = random.choices(
        ["symbol", "iso_pre", "iso_post", "shorthand", "written", "bare_short"],
        weights=[35, 15, 10, 15, 10, 15],
    )[0]

    if fmt == "symbol":
        sym = random.choice(SYMBOLS)
        amount = round(random.uniform(0.01, 9_999_999), 2)
        return f"{sym}{_fmt_number(amount)}"

    if fmt == "iso_pre":
        code = random.choice(ISO_CODES)
        amount = round(random.uniform(1, 9_999_999), 2)
        return f"{code} {_fmt_number(amount)}"

    if fmt == "iso_post":
        code = random.choice(ISO_CODES)
        amount = round(random.uniform(1, 9_999_999), 2)
        return f"{_fmt_number(amount)} {code}"

    if fmt == "shorthand":
        sym = random.choice(SYMBOLS)
        base = round(random.uniform(0.1, 999), 1)
        suffix = random.choice(MAGNITUDE_SUFFIXES)
        return f"{sym}{base}{suffix}"

    if fmt == "written":
        sym = random.choice(SYMBOLS + [""])
        base = round(random.uniform(1, 999), 1)
        mag = random.choice(WRITTEN_MAGNITUDES)
        if sym:
            return f"{sym}{base} {mag}"
        cur = random.choice(CURRENCY_WORDS)
        return f"{base} {mag} {cur}"

    base = round(random.uniform(0.1, 999), 1)
    suffix = random.choice(MAGNITUDE_SUFFIXES)
    return f"{base}{suffix}"


MONEY_PATTERN = re.compile(r"<MONEY>")


def replace_placeholders(sentence: str) -> dict:
    """Replace every <MONEY> with a realistic money string and record char-level spans."""
    spans = []
    result_parts = []
    last_end = 0

    for m in MONEY_PATTERN.finditer(sentence):
        result_parts.append(sentence[last_end:m.start()])
        offset = sum(len(p) for p in result_parts)

        money = generate_money_string()
        spans.append([offset, offset + len(money), 1])
        result_parts.append(money)
        last_end = m.end()

    result_parts.append(sentence[last_end:])
    text = "".join(result_parts)

    return {"text": text, "spans": spans, "source": "synthetic"}


positive_records = [replace_placeholders(s) for s in enriched_sentences]
print(f"Stage 3 produced {len(positive_records)} positive records")


"""## Stage 4 - Hard Negatives"""

stage4_conversations = []
for _ in range(ROUNDS):
    for domain in prompts["stage_4"]["domains"]:
        stage4_conversations.append(
            build_chat(prompts["stage_4"]["system"], domain)
        )

stage4_outputs = llm.chat(stage4_conversations, sampling_params)

negative_records = []
for output in stage4_outputs:
    text = output.outputs[0].text.strip()
    for line in text.split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if len(line) > 10 and "<MONEY>" not in line:
            negative_records.append({"text": line, "spans": [], "source": "synthetic"})

print(f"Stage 4 produced {len(negative_records)} negative records")


"""## Combine and Save"""

dataset = positive_records + negative_records
random.shuffle(dataset)

with open("data/synthetic_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(dataset)} records to data/synthetic_dataset.json")
print(f"  Positive (with money): {sum(1 for r in dataset if r['spans'])}")
print(f"  Negative (no money):   {sum(1 for r in dataset if not r['spans'])}")
