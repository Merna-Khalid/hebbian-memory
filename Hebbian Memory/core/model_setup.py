"""
Local model setup for the Japanese learning Hebbian RAG system.

Models:
    BAAI/bge-m3          — cross-lingual EN+JA embeddings (1024-dim)
                           puts English and Japanese into the SAME vector space
                           so "eat" and "食べる" are neighbours
    LiquidAI/LFM2.5-1.2B-JP — Japanese-optimised chat + summaries
    RobroKools/vad-bert  — loaded inside EmotionClassifier (arme_v2 / emotion_classifier.py)

RAM budget on M4 16GB:
    bge-m3      ~0.6 GB
    LFM2.5-1.2B ~2.5 GB
    vad-bert    ~0.2 GB
    Total       ~3.3 GB  (comfortable on 16 GB)

Usage:
    from model_setup import embedding_fn, summary_fn, chat_fn
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# ── BGE-M3 — cross-lingual EN + JA embeddings ────────────────────────
# Outputs 1024-dim normalised vectors.
# English and Japanese share the same semantic space — critical for a
# bilingual memory system where English questions retrieve Japanese nodes.

print("Loading BAAI/bge-m3 (cross-lingual embeddings)...")
bge = SentenceTransformer("BAAI/bge-m3", device="mps")
print("  → bge-m3 ready")


def embedding_fn(text: str) -> list[float]:
    """
    Embed text with BGE-M3.  Returns a normalised 1024-dim float list.

    No instruction prefix needed for general concept encoding.
    For asymmetric retrieval (query vs. document) you can prepend:
        query:    "Represent this sentence for searching relevant passages: "
        document: no prefix
    For our use case (symmetric — concepts vs. concepts) no prefix is better.
    """
    return bge.encode(text, normalize_embeddings=True).tolist()


# ── LFM2.5-1.2B-JP — summaries and chat ──────────────────────────────
# Japanese-optimised instruction model.
# device_map="auto" lets transformers decide MPS vs CPU layer split.
# bfloat16 halves VRAM vs float32 with negligible quality loss.

print("Loading LiquidAI/LFM2.5-1.2B-JP (chat + summaries)...")
lfm_tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-JP")
lfm_model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-JP",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
lfm_model.eval()
print("  → LFM2.5-1.2B-JP ready")


# ── vad-bert — valence / arousal / dominance ─────────────────────────
# Instantiated once here so IngestPipeline can reuse it without a second
# model load. Loaded lazily on first call inside EmotionClassifier.
from core.emotion_classifier import EmotionClassifier
emotion_classifier = EmotionClassifier(
    device="mps",
    vad_model_path="./core/vad_model"
)
print("  → vad-bert ready (lazy load on first call)")


def summary_fn(text: str) -> str:
    """
    Summarise text in 1-2 sentences using LFM2.5-1.2B-JP.

    Prompt is in Japanese — the model handles English input too and
    will produce a Japanese summary, which is what we want for the
    memory system (Japanese-language concept nodes).

    For English-language concept nodes, swap the prompt to English.
    """
    prompt = (
        f"次のテキストを1〜2文で要約してください。\n\n"
        f"テキスト: {text[:1000]}\n\n要約:"
    )
    # apply_chat_template returns a BatchEncoding dict, not a raw tensor.
    # Extract input_ids before passing to generate().
    encoded = lfm_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )
    input_ids = encoded["input_ids"].to(lfm_model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(lfm_model.device)

    with torch.no_grad():
        output = lfm_model.generate(
            input_ids,
            attention_mask=attention_mask,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=80,
            do_sample=True,
        )
    new_tokens = output[0][input_ids.shape[1]:]
    return lfm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_fn(
    user_message : str,
    history      : list[dict] = None,
    system_prompt: str = None,
) -> str:
    """
    Single call to LFM2.5-1.2B-JP for conversation.

    Parameters
    ----------
    user_message  : the user's current message (English or Japanese)
    history       : list of {"role": "user"/"assistant", "content": "..."}
                    pass the last N turns to keep context
    system_prompt : prepended system message — used to inject retrieved
                    Hebbian memory as RAG context

    Returns
    -------
    The model's response as a plain string.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # apply_chat_template returns a BatchEncoding dict, not a raw tensor.
    # Extract input_ids before passing to generate().
    encoded = lfm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )
    input_ids = encoded["input_ids"].to(lfm_model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(lfm_model.device)

    with torch.no_grad():
        output = lfm_model.generate(
            input_ids,
            attention_mask=attention_mask,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=512,
            do_sample=True,
        )
    new_tokens = output[0][input_ids.shape[1]:]
    return lfm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()