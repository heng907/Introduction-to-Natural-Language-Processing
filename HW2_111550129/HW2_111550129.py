import json
import re
import csv
from tqdm import tqdm
import ollama

# ── 設定 ──────────────────────────────────────────────────
MODEL_NAME  = "qwen3:8b"
TRAIN_PATH  = "train.json"
TEST_PATH   = "test.json"
OUTPUT_PATH = "submission_few_shot_v4.csv"

EVAL_MODE = False
EVAL_SIZE = 200
# ──────────────────────────────────────────────────────────

FLIP = {"A": "B", "B": "A", "tie": "tie", "neither": "neither"}


def format_dialog(dialog: list) -> str:
    lines = []
    for msg in dialog:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}]: {msg['content']}")
    return "\n".join(lines)


def build_few_shot_prompt(dialog_a: list, dialog_b: list) -> str:
    return f"""You are a fair and impartial judge evaluating two AI assistant responses.

Study these examples carefully before making your judgment:

=== NEITHER examples (both responses are bad) ===

[Type 1 - Both give wrong answers]
User: Solve this crossword clue: "BLANK, I'm Adam" (palindrome, 5 letters)
Response A: I apologize for the mistake. The correct answer is "SIS."
Response B: "EVEE" is 5 letters long and is a palindrome.
→ Both gave factually wrong answers. VERDICT: neither

[Type 2 - Both too short / unhelpful]
User: The following is a command that the user is about to run in bash. Please determine if it is safe.
Response A: NO
Response B: NO
→ Both responses are too vague and provide no useful explanation. VERDICT: neither

[Type 3 - Both give the same wrong answer]
User: Act as a professor grading proofs on a test, did this proof pass?
Response A: Yes, this proof passed!
Response B: Yes, this proof passed the test
→ Both responses agree but are both incorrect. VERDICT: neither

[Type 4 - Both refuse or dodge the question]
User: Do not include any numeric characters in your reply. What's the smallest non-negative integer?
Response A: I apologize for the mistake. The smallest non-negative integer is not defined.
Response B: You are right, I apologize for the mistake. The smallest non-negative integer is 0.
→ Response A is wrong, Response B violates the constraint by including "0". VERDICT: neither

=== NOT neither examples (at least one response is acceptable) ===

[Counter-example 1 - This is A, not neither]
User: What causes rainbows?
Response A: Rainbows are caused by the refraction, dispersion, and reflection of sunlight inside water droplets, creating a spectrum of colors.
Response B: Rainbows happen because of water in the air.
→ Response A is detailed and correct. Response B is too vague but not wrong. One is clearly better. VERDICT: A

[Counter-example 2 - This is tie, not neither]
User: What is the capital of Japan?
Response A: The capital of Japan is Tokyo.
Response B: Tokyo is Japan's capital city.
→ Both are correct and helpful. VERDICT: tie

=== A / B / tie examples ===

[VERDICT: A]
User: Explain what a for loop does in Python.
Response A: A for loop in Python iterates over a sequence and executes a block of code for each item. Example: for i in [1,2,3]: print(i) prints 1, 2, 3.
Response B: For loops are used in programming to repeat things.
→ Response A is more complete with a concrete example. VERDICT: A

[VERDICT: B]
User: What are the symptoms of dehydration?
Response A: Dehydration happens when you don't drink enough water.
Response B: Symptoms include thirst, dark urine, dizziness, fatigue, dry mouth, and in severe cases, confusion or rapid heartbeat.
→ Response B directly answers with specific symptoms. VERDICT: B

[VERDICT: tie]
User: What is photosynthesis?
Response A: Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.
Response B: Plants use photosynthesis to turn light energy, carbon dioxide, and water into sugar and oxygen.
→ Both are accurate and equally helpful. VERDICT: tie

Now evaluate the following:

<conversation_A>
{format_dialog(dialog_a)}
</conversation_A>

<conversation_B>
{format_dialog(dialog_b)}
</conversation_B>

Follow these steps:
Step 1 - Are BOTH responses wrong, unhelpful, too vague, or failing to answer? → neither
Step 2 - Is one response noticeably more accurate, complete, or helpful? → A or B
Step 3 - Only if both are correct AND the quality difference is truly negligible → tie

Remember:
- Use "neither" when BOTH responses fail, not just one
- Use "A" or "B" whenever one response is noticeably better, even slightly
- Use "tie" ONLY when both are correct and you truly cannot distinguish quality
- Use tie when both responses are correct and cover the same key points, even if one is slightly longer or more detailed

After your analysis, output your final verdict on the last line:
VERDICT: [A or B or tie or neither]

Your analysis:"""


def parse_verdict(raw: str) -> str:
    # 抓最後出現的 VERDICT:（避免範例干擾）
    matches = re.findall(r'VERDICT:\s*([ABab]|tie|neither)', raw, re.IGNORECASE)
    if matches:
        val = matches[-1].strip().lower()
        return val.upper() if val in ("a", "b") else val

    # fallback：抓最後一行
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].lower()
        for label in ["neither", "tie"]:
            if label in last:
                return label
        m = re.search(r'\b([ab])\b', last)
        if m:
            return m.group(1).upper()

    # 最終 fallback：掃後半段避免範例污染
    raw_lower = raw.lower()
    half = len(raw_lower) // 2
    raw_second_half = raw_lower[half:]
    for label in ["neither", "tie"]:
        if label in raw_second_half:
            return label
    m = re.search(r'\b(neither|tie|[ab])\b', raw_second_half)
    if m:
        val = m.group(1)
        return val if val in ("neither", "tie") else val.upper()

    return "tie"


def predict_once(dialog_a: list, dialog_b: list, temperature: float = 0) -> str:
    prompt = build_few_shot_prompt(dialog_a, dialog_b)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": 512},
        think=False,
    )
    return parse_verdict(response["message"]["content"])


def predict_with_debias(sample: dict) -> str:
    d1, d2 = sample["dialog_1"], sample["dialog_2"]

    pred_ab = predict_once(d1, d2, temperature=0)
    pred_ba = FLIP[predict_once(d2, d1, temperature=0)]

    if pred_ab == pred_ba:
        return pred_ab

    pred_3 = predict_once(d1, d2, temperature=0.3)
    if pred_3 == pred_ab:
        return pred_ab
    elif pred_3 == pred_ba:
        return pred_ba
    else:
        return "tie"


def evaluate(data: list):
    correct = 0
    pred_dist = {}
    consistent = 0

    for sample in tqdm(data, desc="Evaluating"):
        d1, d2 = sample["dialog_1"], sample["dialog_2"]
        pred_ab = predict_once(d1, d2, temperature=0)
        pred_ba = FLIP[predict_once(d2, d1, temperature=0)]

        if pred_ab == pred_ba:
            final = pred_ab
            consistent += 1
        else:
            pred_3 = predict_once(d1, d2, temperature=0.3)
            if pred_3 == pred_ab:
                final = pred_ab
            elif pred_3 == pred_ba:
                final = pred_ba
            else:
                final = "tie"

        pred_dist[final] = pred_dist.get(final, 0) + 1
        if final == sample["verdict"]:
            correct += 1

    acc = correct / len(data)
    print(f"\n✅ Accuracy: {correct}/{len(data)} = {acc:.4f} ({acc*100:.2f}%)")
    print(f"   一致率: {consistent}/{len(data)} = {consistent/len(data)*100:.1f}%")
    print(f"\n預測分佈: {dict(sorted(pred_dist.items()))}")
    gold_dist = {}
    for s in data:
        gold_dist[s["verdict"]] = gold_dist.get(s["verdict"], 0) + 1
    print(f"Gold 分佈:  {dict(sorted(gold_dist.items()))}")


def generate_submission(data: list, output_path: str):
    rows = []
    for sample in tqdm(data, desc="Generating submission"):
        final = predict_with_debias(sample)
        rows.append({"id": sample["id"], "verdict": final})

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    print(f"\n✅ Saved to {output_path} ({len(rows)} rows)")
    print(f"預測分佈: {dict(Counter([r['verdict'] for r in rows]))}")


if __name__ == "__main__":
    if EVAL_MODE:
        print("=== EVAL MODE：在 train 前 200 筆評估 ===\n")
        with open(TRAIN_PATH, encoding="utf-8") as f:
            train = json.load(f)
        evaluate(train[:EVAL_SIZE])
    else:
        print("=== SUBMISSION MODE ===\n")
        with open(TEST_PATH, encoding="utf-8") as f:
            test = json.load(f)
        generate_submission(test, OUTPUT_PATH)
