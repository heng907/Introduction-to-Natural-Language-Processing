import json
import csv
import re
import torch
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ========== 設定路徑 ==========
TRAIN_PATH  = "train.jsonl"
TEST_PATH   = "test.jsonl"
OUTPUT_PATH = "submission_listwise_v2.csv"

# ========== 設定 ==========
DENSE_MODEL_NAME    = "/workspace/models/bge-m3"
RERANKER_MODEL_NAME = "/workspace/models/bge-reranker"
LLM_MODEL_NAME      = "/workspace/models/qwen3-8b"
BATCH_SIZE          = 32
TOP_K               = 5
# ── [Fix 2] RETRIEVAL_K 不再使用固定值，改為動態取全部候選 ──────────────────
# 每個 sample 最多 ~16 個候選，把全部丟進 CE 讓它做完整排序
# 保留此變數僅供 LLM listwise 階段限制輸入數量用
CE_TOP_K            = 10   # LLM 仍只看 CE reranker 排序後的 top-10
RRF_K               = 60
MAX_CANDIDATE_CHARS = 400
USE_4BIT            = True
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

# ── [Fix 1] 不再使用全域 N_IMG_MIN，改為 apply_modality_quota 內部自適應 ───


# ========== 工具函數 ==========

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_keywords(question: str) -> str:
    years = re.findall(r'\b(?:FY)?\d{2,4}\b', question)
    cap_phrases = re.findall(r'\b[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)*\b', question)
    stopwords = {
        "What", "Which", "How", "When", "Where", "Who", "Why",
        "The", "This", "That", "These", "Those", "Find", "Give",
        "List", "Show", "Does", "Did", "Has", "Have", "Round",
        "Answer", "According", "Compare", "Based", "Between",
    }
    cap_phrases = [p for p in cap_phrases if p not in stopwords and len(p) > 3]
    return " ".join(dict.fromkeys(years + cap_phrases))


def build_candidates(sample: dict) -> tuple[list[str], list[str]]:
    question = sample["question"]
    keywords = extract_keywords(question)
    quote_ids, texts = [], []

    for tq in sample["text_quotes"]:
        quote_ids.append(tq["quote_id"])
        texts.append(tq["text"])

    for iq in sample["img_quotes"]:
        quote_ids.append(iq["quote_id"])
        desc = iq["img_description"]
        if keywords:
            desc = f"{desc}\nKeywords: {keywords}"
        texts.append(desc)

    return quote_ids, texts


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def is_image_quote(quote_id: str) -> bool:
    return "image" in quote_id or "img" in quote_id


def recall_at_k(predicted: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return len(set(predicted) & set(gold)) / len(gold)


# ========== Step 1: BM25 ==========

def bm25_retrieve(question: str, quote_ids: list[str],
                  texts: list[str], top_k: int) -> list[str]:
    tokenized_corpus = [tokenize(t) for t in texts]
    tokenized_query  = tokenize(question)
    bm25   = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [quote_ids[i] for i in ranked[:top_k]]


# ========== Step 2: Dense Retrieval ==========

def dense_retrieve(question: str, quote_ids: list[str], texts: list[str],
                   model: SentenceTransformer, top_k: int) -> list[str]:
    query_vec   = model.encode("query: " + question, convert_to_numpy=True,
                                normalize_embeddings=True)
    corpus_vecs = model.encode(texts, batch_size=BATCH_SIZE,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=False)
    scores = corpus_vecs @ query_vec
    ranked = np.argsort(scores)[::-1][:top_k]
    return [quote_ids[i] for i in ranked]


# ========== Step 3: RRF 融合 ==========

def rrf_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = {}
    for ranked in ranked_lists:
        for rank, quote_id in enumerate(ranked, start=1):
            scores[quote_id] = scores.get(quote_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


# ========== Step 4: Cross-encoder Reranker ==========

def rerank(question: str, candidate_ids: list[str], texts_map: dict[str, str],
           reranker: CrossEncoder, top_k: int) -> tuple[list[str], dict[str, float]]:
    pairs  = [[question, texts_map[qid]] for qid in candidate_ids]
    scores = reranker.predict(pairs, show_progress_bar=False)
    score_map  = {qid: float(scores[i]) for i, qid in enumerate(candidate_ids)}
    # ── [Fix 2] top_k 現在傳入 len(candidate_ids)，ranked 取全部後再切片 ──
    ranked     = np.argsort(scores)[::-1][:top_k]
    ranked_ids = [candidate_ids[i] for i in ranked]
    return ranked_ids, score_map


# ========== Step 5: LLM Listwise Reranker ==========

def load_llm(model_path: str, use_4bit: bool = True):
    print(f"載入 LLM：{model_path} (4bit={use_4bit})")
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("LLM 載入完成！")
    return model, tokenizer


# ── [Fix 3] 新增 evidence_modality_type 參數，注入 prompt 作為 modality hint ─
def listwise_rerank(question: str, candidate_ids: list[str],
                    texts_map: dict[str, str],
                    llm_model, llm_tokenizer,
                    top_k: int = 5,
                    max_chars: int = 400,
                    evidence_modality_type: list[str] | None = None) -> list[str]:
    n = len(candidate_ids)
    candidates_text = "\n".join(
        f"[{i+1}] {texts_map[cid][:max_chars]}"
        for i, cid in enumerate(candidate_ids)
    )

    # ── [Fix 3] 組合 modality hint 字串 ──────────────────────────────────────
    if evidence_modality_type:
        modality_hint = ", ".join(evidence_modality_type)
    else:
        modality_hint = "text"

    prompt = f"""You are a document retrieval expert.
Given a question and candidate evidence items from a document, \
rank them from most to least relevant to answer the question.
Expected evidence modality: {modality_hint}

Question: {question}

Candidates:
{candidates_text}

Output only a JSON array of candidate numbers (1-{n}) \
from most to least relevant. Example: [3,1,5,2,4]
Output:"""

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    generated = llm_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    try:
        match = re.search(r'\[[\d,\s]+\]', generated)
        if match:
            numbers = json.loads(match.group())
            ranked = [candidate_ids[n_-1] for n_ in numbers
                      if isinstance(n_, int) and 1 <= n_ <= len(candidate_ids)]
            seen = set(ranked)
            ranked += [cid for cid in candidate_ids if cid not in seen]
            return ranked[:top_k]
    except Exception:
        pass

    return candidate_ids[:top_k]


# ========== Step 6: Adaptive Modality Quota ==========
#
# ── [Fix 1] 根據 evidence_modality_type 自適應決定 quota ─────────────────────
#
# 規則（根據訓練集統計得出）：
#   • pure image (table / figure / chart，無 text)
#       → gold 100% 是 image，強制全部 5 個都取 image
#   • mixed (有 image modality 也有 text)
#       → gold 約 50% image，維持至少 2 個 image
#   • text only (僅 text，無 image modality)
#       → gold 全是 text，不做任何替換
#
def apply_modality_quota(
    ranked_ids: list[str],
    all_candidate_ids: list[str],
    reranker_scores: dict[str, float],
    evidence_modality_type: list[str],   # ← 新增，取代舊的 n_img_min 參數
    top_k: int = 5,
) -> list[str]:
    emt = set(evidence_modality_type)
    has_img_modality  = bool(emt & {"table", "figure", "chart"})
    has_text_modality = "text" in emt
    pure_img = has_img_modality and not has_text_modality

    # ── Case 1: pure image → 全部取 image，按 CE score 排序 ─────────────────
    if pure_img:
        img_candidates = [
            qid for qid in all_candidate_ids if is_image_quote(qid)
        ]
        img_sorted = sorted(
            img_candidates,
            key=lambda x: reranker_scores.get(x, -999.0),
            reverse=True,
        )
        return img_sorted[:top_k]

    # ── Case 2: text only → 直接回傳 LLM 排序結果，不做替換 ─────────────────
    if not has_img_modality:
        return list(ranked_ids[:top_k])

    # ── Case 3: mixed → 確保至少 2 個 image ─────────────────────────────────
    n_img_min = 2
    result = list(ranked_ids[:top_k])

    img_in_result = [qid for qid in result if is_image_quote(qid)]
    if len(img_in_result) >= n_img_min:
        return result

    need         = n_img_min - len(img_in_result)
    selected_set = set(result)
    extra_imgs   = [
        qid for qid in all_candidate_ids
        if is_image_quote(qid) and qid not in selected_set
    ]
    extra_imgs_sorted = sorted(
        extra_imgs,
        key=lambda x: reranker_scores.get(x, -999.0),
        reverse=True,
    )
    imgs_to_add = extra_imgs_sorted[:need]

    if not imgs_to_add:
        return result

    # 替換掉 CE score 最低的 text quote
    text_in_result = [
        (i, qid) for i, qid in enumerate(result)
        if not is_image_quote(qid)
    ]
    text_sorted_asc = sorted(
        text_in_result,
        key=lambda x: reranker_scores.get(x[1], -999.0),
    )
    for img_qid, (replace_idx, _) in zip(imgs_to_add, text_sorted_asc):
        result[replace_idx] = img_qid

    return result


# ========== 完整 Pipeline ==========

def hybrid_listwise_pipeline(
    samples: list[dict],
    dense_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    llm_model,
    llm_tokenizer,
    top_k: int = 5,
    ce_top_k: int = 10,
) -> list[list[str]]:
    all_predictions = []

    for sample in tqdm(samples, desc="Processing"):
        question             = sample["question"]
        evidence_modality    = sample.get("evidence_modality_type", [])

        quote_ids, texts = build_candidates(sample)
        texts_map        = dict(zip(quote_ids, texts))
        n_candidates     = len(quote_ids)   # 通常 10 text + 5 img = 15

        # ── [Fix 2] BM25 / Dense 都取全部候選，不再受 RETRIEVAL_K 截斷 ───────
        bm25_top  = bm25_retrieve(question, quote_ids, texts, top_k=n_candidates)
        dense_top = dense_retrieve(question, quote_ids, texts,
                                   dense_model, top_k=n_candidates)

        # RRF 融合後也不截斷，保留全部候選的融合排序
        fused = rrf_fusion([bm25_top, dense_top], k=RRF_K)

        # ── [Fix 2] CE reranker 對全部候選評分，取 top CE_TOP_K 給 LLM ──────
        ce_ranked, score_map = rerank(
            question, fused, texts_map,
            cross_encoder, top_k=n_candidates,  # ← 全部傳入 CE
        )

        # ── [Fix 3] 傳入 evidence_modality_type 讓 LLM 知道期待的 modality ──
        lw_ranked = listwise_rerank(
            question,
            ce_ranked[:ce_top_k],   # LLM 仍只看 CE 篩選後的 top-10
            texts_map,
            llm_model, llm_tokenizer,
            top_k=top_k,
            max_chars=MAX_CANDIDATE_CHARS,
            evidence_modality_type=evidence_modality,
        )

        # ── [Fix 1] 自適應 modality quota，取代固定的 n_img_min=2 ────────────
        final = apply_modality_quota(
            ranked_ids=lw_ranked,
            all_candidate_ids=quote_ids,
            reranker_scores=score_map,
            evidence_modality_type=evidence_modality,
            top_k=top_k,
        )

        all_predictions.append(final)

    return all_predictions


# ========== 評估 ==========

def evaluate(samples: list[dict], predictions: list[list[str]]) -> float:
    total = sum(
        recall_at_k(pred, s.get("gold_quotes", []))
        for s, pred in zip(samples, predictions)
    )
    return total / len(samples)


# ========== 生成提交檔 ==========

def generate_submission(samples: list[dict], predictions: list[list[str]],
                        output_path: str):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["q_id", "gold_quotes"])
        writer.writeheader()
        for sample, predicted in zip(samples, predictions):
            writer.writerow({
                "q_id": sample["q_id"],
                "gold_quotes": " ".join(predicted),
            })
    print(f"已輸出 {len(samples)} 筆到 {output_path}")


# ========== 主程式 ==========

if __name__ == "__main__":
    print(f"使用裝置：{DEVICE}")

    print(f"\n載入 Dense model：{DENSE_MODEL_NAME}")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME, device=DEVICE)
    print("Dense model 載入完成！")

    print(f"\n載入 Cross-encoder：{RERANKER_MODEL_NAME}")
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)
    print("Cross-encoder 載入完成！")

    llm_model, llm_tokenizer = load_llm(LLM_MODEL_NAME, use_4bit=USE_4BIT)

    # ── Train 評估（前 100 筆快速確認）──────────────────────────────────────
    print("\n載入 train.jsonl ...")
    train_samples = load_jsonl(TRAIN_PATH)
    print(f"共 {len(train_samples)} 筆")

    print("\n先跑前 100 筆確認 pipeline 正確...")
    sample_preds = hybrid_listwise_pipeline(
        train_samples[:100], dense_model, cross_encoder,
        llm_model, llm_tokenizer,
        top_k=TOP_K, ce_top_k=CE_TOP_K,
    )
    print(f"前 100 筆 Recall@5: {evaluate(train_samples[:100], sample_preds):.4f}")

    print("\n跑完整 train 評估...")
    train_preds = hybrid_listwise_pipeline(
        train_samples, dense_model, cross_encoder,
        llm_model, llm_tokenizer,
        top_k=TOP_K, ce_top_k=CE_TOP_K,
    )
    print(f"Train Recall@5: {evaluate(train_samples, train_preds):.4f}")

    # ── Test 提交 ─────────────────────────────────────────────────────────────
    print("\n載入 test.jsonl ...")
    test_samples = load_jsonl(TEST_PATH)
    print(f"共 {len(test_samples)} 筆")

    test_preds = hybrid_listwise_pipeline(
        test_samples, dense_model, cross_encoder,
        llm_model, llm_tokenizer,
        top_k=TOP_K, ce_top_k=CE_TOP_K,
    )
    generate_submission(test_samples, test_preds, OUTPUT_PATH)
    print("\n完成！")
