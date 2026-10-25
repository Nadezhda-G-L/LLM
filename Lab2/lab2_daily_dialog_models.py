# -*- coding: utf-8 -*-
#
# Лабораторная работа по разговорному ИИ и большим языковым моделям.
#
# Задача: обучить две диалоговые модели, decoder-only и encoder-decoder,
# чтобы они предсказывали ответ системы на запрос пользователя.
#
# Модель 1, decoder-only:
#     facebook/opt-125m
#
# Модель 2, encoder-decoder:
#     google/flan-t5-small
#
# Датасет:
#     HuggingFaceH4/no_robots
#
# Зависимости:
#     pip install -U datasets transformers accelerate nltk sentencepiece sacrebleu
#

# -- coding: utf-8 --
import os
import tempfile
import shutil
import re
import json
import csv
import gc
import inspect
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

# Переводим кэш Hugging Face во временную папку.
HF_HOME = os.path.join(tempfile.gettempdir(), "hf_home")
HF_DATASETS_CACHE = os.path.join(tempfile.gettempdir(), "hf_datasets")
HF_HUB_CACHE = os.path.join(tempfile.gettempdir(), "hf_hub")
TRANSFORMERS_CACHE = os.path.join(tempfile.gettempdir(), "hf_transformers")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

# Создаем каталоги кэша заранее.
for cache_path in [HF_HOME, HF_DATASETS_CACHE, HF_HUB_CACHE, TRANSFORMERS_CACHE]:
    try:
        os.makedirs(cache_path, exist_ok=True)
    except Exception:
        pass

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import nltk
from nltk.translate.meteor_score import single_meteor_score

SEED = 42

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

set_all_seeds(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECODER_ONLY_MODEL_ID = "facebook/opt-125m"
ENCODER_DECODER_MODEL_ID = "google/flan-t5-small"
DATASET_ID = "HuggingFaceH4/no_robots"
DATASET_NAME_USED = DATASET_ID

TRAIN_DIALOGUE_SAMPLES = 8000
VALID_DIALOGUE_SAMPLES = 1000
TEST_DIALOGUE_SAMPLES = 1000

VALID_SPLIT_SIZE = 0.15
TEST_SPLIT_SIZE = 0.15
MAX_TRAIN_PAIRS = 30000
MAX_VALID_PAIRS = 1000
MAX_TEST_PAIRS = 1000

MAX_HISTORY_TURNS = 6
MAX_SOURCE_WORDS = 240
MAX_TARGET_WORDS = 64

MAX_SOURCE_LEN = 192
MAX_TARGET_LEN = 64

TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
GRAD_ACC_STEPS = 2
EVAL_BATCH_SIZE = 16

DECODER_LEARNING_RATE = 5e-5
ENCODER_LEARNING_RATE = 2e-4

MAX_GRAD_NORM = 1.0

OPTIM = "adamw_torch"
WARMUP_RATIO = 0.05

USE_FP16 = False

MAX_NEW_TOKENS = 64
MIN_NEW_TOKENS = 8

DECODER_BEAM_SIZE = 1
ENCODER_BEAM_SIZE = 4

# Промпты для моделей
def causal_prompt(context: str) -> str:
    """
    Промпт для decoder-only модели.
    Микро-инструкция и уникальный маркер генерации [ASSISTANT].
    """
    return f"Dialogue history (User and Assistant):\n{context}\n\n[ASSISTANT]\n"

def seq2seq_prompt(context: str) -> str:
    """
    Промпт для encoder-decoder модели.
    """
    return (
        "You are a helpful assistant. "
        "Given a conversation between User and Assistant, "
        "generate only the next Assistant response.\n"
        f"Conversation:\n{context}\n"
        "Assistant response:"
    )

# Утилиты для NLTK и метрик
def download_nltk_resources() -> None:
    resources = [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet",
        "wordnet_ic",
        "omw-1.4",
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

def normalize_text(text: str) -> str:
    return " ".join(str(text).split())

def clean_role_prefix(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(
        r"^\s*(?:user|human|prompter|question|input|client)\s*[:.-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:assistant|bot|gpt|chatgpt|answer|response|output|ai)\s*[:.-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*(?:system|sys)\s*[:.-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()

def postprocess_generation(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"</?s>|<pad>|<unk>|<mask>|<eos>", " ", text)
    text = re.sub(
        r"^\s*(?:User|Human|System|Bot|Assistant|Response)\s*[:-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip()
    lower_text = text.lower()
    cut_markers = [
        "\nUser:",
        "\nUser ",
        "\nHuman:",
        "\nSystem:",
        "\nBot:",
        "\nAssistant:",
    ]
    for marker in cut_markers:
        marker_lower = marker.lower()
        if marker_lower in lower_text:
            pos = lower_text.index(marker_lower)
            text = text[:pos].strip()
            lower_text = text.lower()
    match = re.match(r"^\s*([A-Za-z0-9_ .#-]{1,40}):\s+(.+)$", text)
    if match:
        speaker_prefix = match.group(1)
        cleaned_text = match.group(2)
        if len(speaker_prefix.split()) <= 3:
            text = cleaned_text.strip()
    text = " ".join(text.split())
    # Нормализация кавычек
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    words = text.split()
    if len(words) > MAX_TARGET_WORDS + 16:
        text = " ".join(words[: MAX_TARGET_WORDS + 16])
    return text.strip()

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    preds = [normalize_text(p) for p in predictions]
    refs = [normalize_text(r) for r in references]
    try:
        import sacrebleu
        scored = sacrebleu.corpus_bleu(
            preds,
            [refs],
            tokenize="13a",
        )
        return float(scored.score)
    except Exception as exc:
        print(f"BLEU через sacrebleu не считается: {exc}")
        return 0.0

def compute_meteor(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    download_nltk_resources()
    preds = [normalize_text(p) for p in predictions]
    refs = [normalize_text(r) for r in references]
    try:
        scores = []
        for pred, ref in zip(preds, refs):
            if not pred or not ref:
                scores.append(0.0)
                continue
            try:
                score = single_meteor_score(ref.split(), pred.split())
                scores.append(float(score or 0.0))
            except Exception:
                scores.append(0.0)
        if not scores:
            return 0.0
        return float(np.mean(scores)) * 100.0
    except Exception as exc:
        print(f"METEOR через NLTK не считается: {exc}")
        return 0.0


# Работа с сообщениями и ролями

def normalize_role(role: Any) -> Optional[str]:
    if role is None:
        return None
    role = str(role).strip().lower()
    if role in {"user", "human", "prompter", "question", "input", "client"}:
        return "user"
    if role in {
        "assistant",
        "bot",
        "gpt",
        "chatgpt",
        "answer",
        "response",
        "output",
        "ai",
    }:
        return "assistant"
    return None

def first_non_empty(mapping: Dict[str, Any], keys: List[str]) -> Optional[str]:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            if text:
                return text
        if isinstance(value, list):
            parts = []
            for item in value:
                if item is None:
                    continue
                item_text = str(item).strip()
                if item_text:
                    parts.append(item_text)
            text = " ".join(parts)
            if text:
                return text
    return None

def parse_single_line(line: str) -> Optional[Dict[str, str]]:
    text = normalize_text(line)
    if not text:
        return None
    user_match = re.match(
        r"^(user|human|prompter|question|input|client)\s*[:.-]\s*(.+)", text, flags=re.IGNORECASE
    )
    if user_match:
        return {"role": "user", "content": clean_role_prefix(user_match.group(2))}
    assistant_match = re.match(
        r"^(assistant|bot|gpt|chatgpt|answer|response|output|ai)\s*[:.-]\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if assistant_match:
        return {"role": "assistant", "content": clean_role_prefix(assistant_match.group(2))}
    system_match = re.match(
        r"^(system|sys)\s*[:.-]\s*(.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if system_match:
        return None
    return None

def normalize_message(turn: Any) -> Optional[Dict[str, str]]:
    if isinstance(turn, dict):
        role_value = turn.get("role", turn.get("from", turn.get("speaker", turn.get("author", ""))))
        role = normalize_role(role_value)
        text = first_non_empty(
            turn,
            [
                "content",
                "text",
                "value",
                "message",
                "response",
                "answer",
            ],
        )
        if text is None:
            return None
        text = clean_role_prefix(text)
        if not text:
            return None
        if role is None:
            parsed = parse_single_line(text)
            if parsed:
                return parsed
            return None
        return {"role": role, "content": text}
    if isinstance(turn, str):
        return parse_single_line(turn)
    return None

def parse_messages(raw_dialogue: Any) -> List[Dict[str, str]]:
    if raw_dialogue is None:
        return []
    if isinstance(raw_dialogue, dict):
        for key in ["messages", "conversations", "dialogue", "dialog", "turns"]:
            if key in raw_dialogue:
                return parse_messages(raw_dialogue[key])
        return []
    if isinstance(raw_dialogue, list):
        messages = []
        for item in raw_dialogue:
            msg = normalize_message(item)
            if msg:
                messages.append(msg)
        if messages:
            return messages
        if all(isinstance(x, str) for x in raw_dialogue):
            fallback_messages = []
            for idx, line in enumerate(raw_dialogue):
                parsed = parse_single_line(line)
                if parsed:
                    fallback_messages.append(parsed)
                    continue
                content = clean_role_prefix(line)
                if content:
                    role = "user" if idx % 2 == 0 else "assistant"
                    fallback_messages.append({"role": role, "content": content})
            return fallback_messages
        return []
    if isinstance(raw_dialogue, str):
        text = raw_dialogue.strip()
        if not text:
            return []
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, list):
                return parse_messages(parsed_json)
        except Exception:
            pass
        messages = []
        for line in text.splitlines():
            msg = parse_single_line(line)
            if msg:
                messages.append(msg)
        if messages:
            return messages
        content = clean_role_prefix(text)
        if content:
            return [{"role": "user", "content": content}]
        return []
    return []

def format_message(message: Dict[str, str]) -> str:
    role = message["role"].capitalize()
    return f"{role}: {message['content']}"

def build_context_with_limits(messages: List[Dict[str, str]], user_idx: int) -> Optional[str]:
    if user_idx < 0 or user_idx >= len(messages):
        return None
    current_message = messages[user_idx]
    current_line = format_message(current_message)
    current_words = len(current_line.split())
    if current_words > MAX_SOURCE_WORDS:
        return None
    lines = [current_line]
    total_words = current_words
    start = max(0, user_idx - MAX_HISTORY_TURNS + 1)
    for j in range(user_idx - 1, start - 1, -1):
        line = format_message(messages[j])
        line_words = len(line.split())
        if total_words + line_words > MAX_SOURCE_WORDS:
            break
        lines.insert(0, line)
        total_words += line_words
    # Разделяем реплики пустой строкой для лучшей читаемости моделью
    return "\n\n".join(lines)

def build_pairs_from_messages(messages: List[Dict[str, str]]) -> List[tuple]:
    clean_messages = []
    for msg in messages:
        role = msg.get("role")
        content = clean_role_prefix(msg.get("content", ""))
        if role in {"user", "assistant"} and content:
            clean_messages.append({"role": role, "content": content})
    pairs = []
    for i in range(len(clean_messages) - 1):
        current = clean_messages[i]
        nxt = clean_messages[i + 1]
        if current["role"] != "user" or nxt["role"] != "assistant":
            continue
        target = nxt["content"]
        target_words = len(target.split())
        if target_words < 1:
            continue
        if target_words > MAX_TARGET_WORDS:
            continue
        # Фильтр отписок и односложных ответов (Yes, No, Ok, Thanks)
        target_lower = target.lower()
        stopword_answers = {"yes", "no", "ok", "okay", "thanks", "thank you", "sure", "maybe", "idk"}
        if target_lower in stopword_answers:
            continue
        source = build_context_with_limits(clean_messages, i)
        if source is None:
            continue
        if len(source.split()) < 1:
            continue
        pairs.append((source, target))
    return pairs

def has_valid_user_assistant_pair(raw_dialogue: Any) -> bool:
    try:
        messages = parse_messages(raw_dialogue)
        return len(build_pairs_from_messages(messages)) > 0
    except Exception:
        return False

def detect_message_column(dataset) -> str:
    candidate_names = [
        "messages",
        "conversations",
        "dialogue",
        "dialog",
        "turns",
        "utterances",
        "chat",
        "history",
    ]
    lower_columns = {name.lower(): name for name in dataset.column_names}
    for candidate in candidate_names:
        if candidate in lower_columns:
            return lower_columns[candidate]
    for column_name in dataset.column_names:
        lower_name = column_name.lower()
        if (
            "message" in lower_name
            or "conversation" in lower_name
            or "dialog" in lower_name
            or "turn" in lower_name
            or "history" in lower_name
        ):
            return column_name
    raise ValueError(
        f"Не удалось найти колонку с сообщениями. "
        f"Доступные колонки: {dataset.column_names}"
    )

def with_normalized_message_column(dataset):
    column_name = detect_message_column(dataset)
    if column_name != "messages":
        dataset = dataset.rename_column(column_name, "messages")
    return dataset, "messages"


# Подготовка датасета

def load_dataset_all_splits(dataset_id: str):
    try:
        return load_dataset(
            dataset_id,
            cache_dir=HF_DATASETS_CACHE,
            trust_remote_code=False,
        )
    except TypeError:
        return load_dataset(
            dataset_id,
            cache_dir=HF_DATASETS_CACHE,
        )

def make_three_splits_from_single_dataset(dataset):
    split1 = dataset.train_test_split(
        test_size=VALID_SPLIT_SIZE,
        seed=SEED,
    )
    temp_train = split1["train"]
    validation = split1["test"]
    if VALID_SPLIT_SIZE >= 1.0:
        raise ValueError("VALID_SPLIT_SIZE должен быть меньше 1.0")
    second_test_size = TEST_SPLIT_SIZE / (1.0 - VALID_SPLIT_SIZE)
    split2 = temp_train.train_test_split(
        test_size=second_test_size,
        seed=SEED + 1,
    )
    train = split2["train"]
    test = split2["test"]
    return train, validation, test

def get_dataset_splits(dataset_id: str = DATASET_ID):
    global DATASET_NAME_USED
    print(f"\nПытаюсь загрузить dataset: {dataset_id}")
    print(f"cache_dir: {HF_DATASETS_CACHE}")
    ds = load_dataset_all_splits(dataset_id)
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        print(f"DatasetDict загружен. Сплиты: {keys}")
        if len(keys) == 0:
            raise ValueError("DatasetDict пустой.")
        if "train" in ds and len(ds["train"]) > 0:
            train = ds["train"]
        else:
            print(
                "Поле 'train' не найдено или пусто. "
                f"Беру первый сплит как train: {keys[0]}"
            )
            train = ds[keys[0]]
        if len(train) == 0:
            raise ValueError("Train-сплит пустой.")
        print(f"Официальных train-диалогов: {len(train)}")
        if "validation" in ds and len(ds["validation"]) > 0:
            validation = ds["validation"]
            print(f"Использую официальный validation: {len(validation)}")
        else:
            validation = None
            print("Официального validation нет. Буду делать из train.")
        if "test" in ds and len(ds["test"]) > 0:
            test = ds["test"]
            print(f"Использую официальный test: {len(test)}")
        else:
            test = None
            print("Официального test нет.")
        if validation is None:
            split = train.train_test_split(
                test_size=VALID_SPLIT_SIZE,
                seed=SEED,
            )
            train = split["train"]
            validation = split["test"]
            print(f"Сделан validation из train: {len(validation)}")
            print(f"Train после выделения validation: {len(train)}")
        if test is None:
            split = train.train_test_split(
                test_size=TEST_SPLIT_SIZE,
                seed=SEED + 1,
            )
            train = split["train"]
            test = split["test"]
            print(f"Сделан test из train: {len(test)}")
            print(f"Train после выделения test: {len(train)}")
        DATASET_NAME_USED = dataset_id
        return train, validation, test, dataset_id
    print("load_dataset вернул один Dataset, делаю три сплита вручную.")
    train, validation, test = make_three_splits_from_single_dataset(ds)
    DATASET_NAME_USED = dataset_id
    return train, validation, test, dataset_id

def sample_dataset(dataset, max_size: int, seed: int = SEED):
    if dataset is None:
        return None
    if max_size is None or max_size <= 0:
        return dataset
    if len(dataset) <= max_size:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_size))

def expand_to_pairs(
    batch: Dict[str, Any],
    messages_col: str,
) -> Dict[str, List[str]]:
    inputs: List[str] = []
    targets: List[str] = []
    dialogues = batch[messages_col]
    for dialogue in dialogues:
        messages = parse_messages(dialogue)
        pairs = build_pairs_from_messages(messages)
        for source, target in pairs:
            inputs.append(source)
            targets.append(target)
    return {
        "input": inputs,
        "target": targets,
    }

def remove_duplicate_pairs(dataset):
    seen = set()
    indices_to_keep = []
    for i, (source, target) in enumerate(zip(dataset["input"], dataset["target"])):
        key = (source.lower(), target.lower())
        if key not in seen:
            seen.add(key)
            indices_to_keep.append(i)
    return dataset.select(indices_to_keep)

def remove_pairs_in_dataset(target_dataset, source_dataset):
    if target_dataset is None:
        return None
    existing = set()
    for source, target in zip(source_dataset["input"], source_dataset["target"]):
        existing.add((source.lower(), target.lower()))
    indices_to_keep = []
    for i, (source, target) in enumerate(
        zip(target_dataset["input"], target_dataset["target"])
    ):
        if (source.lower(), target.lower()) not in existing:
            indices_to_keep.append(i)
    return target_dataset.select(indices_to_keep)

def prepare_data():
    global DATASET_NAME_USED
    print("Загружаем dataset...")
    train_raw, validation_raw, test_raw, used_dataset_name = get_dataset_splits(
        DATASET_ID
    )
    DATASET_NAME_USED = used_dataset_name
    print("\nНормализую колонку с сообщениями...")
    train_raw, _ = with_normalized_message_column(train_raw)
    if validation_raw is None or len(validation_raw) == 0:
        if len(train_raw) < 10:
            raise ValueError("Train-диалогов слишком мало, чтобы создать validation.")
        split = train_raw.train_test_split(
            test_size=VALID_SPLIT_SIZE,
            seed=SEED,
        )
        train_raw = split["train"]
        validation_raw = split["test"]
        validation_raw, _ = with_normalized_message_column(validation_raw)
    if test_raw is None or len(test_raw) == 0:
        test_raw = None
    messages_col = "messages"
    print("\nФильтрую диалоги без пар User -> Assistant...")
    train_raw = train_raw.filter(
        lambda example: has_valid_user_assistant_pair(example[messages_col]),
        num_proc=1,
    )
    validation_raw = validation_raw.filter(
        lambda example: has_valid_user_assistant_pair(example[messages_col]),
        num_proc=1,
    )
    if test_raw is not None:
        test_raw = test_raw.filter(
            lambda example: has_valid_user_assistant_pair(example[messages_col]),
            num_proc=1,
        )
    print(f"Валидных train-диалогов после фильтрации: {len(train_raw)}")
    print(f"Валидных validation-диалогов после фильтрации: {len(validation_raw)}")
    if test_raw is not None:
        print(f"Валидных test-диалогов после фильтрации: {len(test_raw)}")
    train_dialogues = sample_dataset(
        train_raw,
        TRAIN_DIALOGUE_SAMPLES,
    )
    validation_dialogues = sample_dataset(
        validation_raw,
        VALID_DIALOGUE_SAMPLES,
    )
    test_dialogues = None
    if test_raw is not None and len(test_raw) > 0:
        test_dialogues = sample_dataset(
            test_raw,
            TEST_DIALOGUE_SAMPLES,
        )
    print(f"Диалогов для train: {len(train_dialogues)}")
    print(f"Диалогов для validation: {len(validation_dialogues)}")
    if test_dialogues is not None:
        print(f"Диалогов для test: {len(test_dialogues)}")
    if validation_dialogues is None or len(validation_dialogues) == 0:
        if len(train_dialogues) < 10:
            raise ValueError("Диалогов для train слишком мало, чтобы создать validation.")
        split = train_dialogues.train_test_split(
            test_size=VALID_SPLIT_SIZE,
            seed=SEED,
        )
        train_dialogues = split["train"]
        validation_dialogues = split["test"]
        print("Validation был пуст. Он был создан из train.")
        print(f"Новое количество train-диалогов: {len(train_dialogues)}")
        print(f"Новое количество validation-диалогов: {len(validation_dialogues)}")
    print("\nПревращаю диалоги в пары input/target...")
    train_pairs = train_dialogues.map(
        expand_to_pairs,
        batched=True,
        batch_size=1000,
        fn_kwargs={"messages_col": messages_col},
        remove_columns=train_dialogues.column_names,
        num_proc=1,
    )
    validation_pairs = validation_dialogues.map(
        expand_to_pairs,
        batched=True,
        batch_size=1000,
        fn_kwargs={"messages_col": messages_col},
        remove_columns=validation_dialogues.column_names,
        num_proc=1,
    )
    test_pairs = None
    if test_dialogues is not None and len(test_dialogues) > 0:
        test_pairs = test_dialogues.map(
            expand_to_pairs,
            batched=True,
            batch_size=1000,
            fn_kwargs={"messages_col": messages_col},
            remove_columns=test_dialogues.column_names,
            num_proc=1,
        )
    print(f"Пар до дедупликации, train: {len(train_pairs)}")
    print(f"Пар до дедупликации, validation: {len(validation_pairs)}")
    if test_pairs is not None:
        print(f"Пар до дедупликации, test: {len(test_pairs)}")
    train_pairs = remove_duplicate_pairs(train_pairs)
    validation_pairs = remove_duplicate_pairs(validation_pairs)
    if test_pairs is not None:
        test_pairs = remove_duplicate_pairs(test_pairs)
    validation_pairs = remove_pairs_in_dataset(validation_pairs, train_pairs)
    if test_pairs is not None:
        test_pairs = remove_pairs_in_dataset(test_pairs, train_pairs)
    train_pairs = train_pairs.shuffle(seed=SEED).select(
        range(min(MAX_TRAIN_PAIRS, len(train_pairs)))
    )
    validation_pairs = validation_pairs.shuffle(seed=SEED).select(
        range(min(MAX_VALID_PAIRS, len(validation_pairs)))
    )
    if test_pairs is not None:
        test_pairs = test_pairs.shuffle(seed=SEED).select(
            range(min(MAX_TEST_PAIRS, len(test_pairs)))
        )
    if test_pairs is not None and len(test_pairs) == 0:
        test_pairs = None
    print(f"\nФинальный train pairs: {len(train_pairs)}")
    print(f"Финальный validation pairs: {len(validation_pairs)}")
    if test_pairs is not None:
        print(f"Финальный test pairs: {len(test_pairs)}")
    else:
        print("Финальный test pairs: test evaluation пропущен.")
    print("\nПримеры подготовленных пар:")
    for i in range(min(5, len(train_pairs))):
        print(f"[{i}] input:")
        print(train_pairs[i]["input"])
        print(" target:")
        print(train_pairs[i]["target"])
        print("")
    if len(train_pairs) == 0:
        raise RuntimeError("После подготовки не осталось тренировочных пар.")
    if len(validation_pairs) == 0:
        raise RuntimeError("После подготовки не осталось валидационных пар.")
    return train_pairs, validation_pairs, test_pairs


# Работа с моделями

def get_pad_and_eos_ids(tokenizer):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id
    if eos_id is None:
        eos_id = pad_id
    if pad_id is None:
        pad_id = 0
    if eos_id is None:
        eos_id = pad_id
    return pad_id, eos_id

def load_model_with_eager_fp32(model_cls, model_id: str):
    attempt_kwargs = [
        {"attn_implementation": "eager", "dtype": torch.float32, "trust_remote_code": False},
        {"attn_implementation": "eager", "torch_dtype": torch.float32, "trust_remote_code": False},
        {"dtype": torch.float32, "trust_remote_code": False},
        {"torch_dtype": torch.float32, "trust_remote_code": False},
        {"attn_implementation": "eager", "trust_remote_code": False},
        {"dtype": torch.float32},
        {"torch_dtype": torch.float32},
        {"trust_remote_code": False},
        {},
    ]
    last_exc = None
    for kwargs in attempt_kwargs:
        try:
            return model_cls.from_pretrained(model_id, **kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            last_exc = exc
            continue
        except Exception:
            raise
    raise RuntimeError(f"Не удалось загрузить модель {model_id}") from last_exc

def make_model_attention_stable(model):
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    try:
        if hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1
    except Exception:
        pass
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
    except Exception:
        pass
    return model

def load_model_and_tokenizer(model_id: str, model_type: str):
    print(f"Загружаем модель: {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=False,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
    added_pad = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        added_pad = True
    if tokenizer.eos_token is None and tokenizer.pad_token is not None:
        tokenizer.eos_token = tokenizer.pad_token
    tokenizer.padding_side = "right"
    if model_type == "causal":
        model = load_model_with_eager_fp32(AutoModelForCausalLM, model_id)
    elif model_type == "seq2seq":
        model = load_model_with_eager_fp32(AutoModelForSeq2SeqLM, model_id)
    else:
        raise ValueError(f"Неизвестный model_type: {model_type}")
    model = make_model_attention_stable(model)
    if added_pad:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model_type == "causal":
        if getattr(model.config, "bos_token_id", None) is None:
            model.config.bos_token_id = (
                tokenizer.bos_token_id
                if tokenizer.bos_token_id is not None
                else tokenizer.eos_token_id
            )
    if model_type == "seq2seq":
        if getattr(model.config, "decoder_start_token_id", None) is None:
            model.config.decoder_start_token_id = model.config.pad_token_id
    try:
        model = model.to(dtype=torch.float32)
    except Exception:
        pass
    model = model.to(DEVICE)
    return model, tokenizer

def tokenize_causal_batch(
    batch: Dict[str, List[str]],
    tokenizer,
) -> Dict[str, List[Any]]:
    _, eos_id = get_pad_and_eos_ids(tokenizer)
    bos_id = tokenizer.bos_token_id
    input_ids = []
    labels = []
    for source, target in zip(batch["input"], batch["target"]):
        prompt = causal_prompt(source)
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        )["input_ids"]
        if bos_id is not None:
            if len(prompt_ids) == 0 or prompt_ids[0] != bos_id:
                prompt_ids = [bos_id] + prompt_ids
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_TARGET_LEN - 1,
        )["input_ids"]
        if eos_id is not None:
            target_ids = target_ids + [eos_id]
        combined_ids = prompt_ids + target_ids
        combined_labels = [-100] * len(prompt_ids) + target_ids
        input_ids.append(combined_ids)
        labels.append(combined_labels)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

def tokenize_seq2seq_batch(
    batch: Dict[str, List[str]],
    tokenizer,
) -> Dict[str, List[Any]]:
    _, eos_id = get_pad_and_eos_ids(tokenizer)
    sources = [seq2seq_prompt(source) for source in batch["input"]]
    encoded_sources = tokenizer(
        sources,
        truncation=True,
        max_length=MAX_SOURCE_LEN,
        padding=False,
        add_special_tokens=True,
        return_tensors=None,
    )
    targets = []
    for target in batch["target"]:
        target_ids = tokenizer(
            target,
            truncation=True,
            max_length=MAX_TARGET_LEN - 1,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        if eos_id is not None:
            target_ids = target_ids + [eos_id]
        targets.append(target_ids)
    return {
        "input_ids": encoded_sources["input_ids"],
        "attention_mask": encoded_sources["attention_mask"],
        "labels": targets,
    }

def causal_collate(
    features: List[Dict[str, Any]],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    pad_id, _ = get_pad_and_eos_ids(tokenizer)
    max_len = max(len(feature["input_ids"]) for feature in features)
    batch_input_ids = []
    batch_labels = []
    for feature in features:
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        padding_len = max_len - len(input_ids)
        padded_input_ids = input_ids + [pad_id] * padding_len
        padded_labels = labels + [-100] * padding_len
        batch_input_ids.append(padded_input_ids)
        batch_labels.append(padded_labels)
    input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": torch.ones_like(input_ids_tensor),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }

def seq2seq_collate(
    features: List[Dict[str, Any]],
    tokenizer,
) -> Dict[str, torch.Tensor]:
    pad_id, _ = get_pad_and_eos_ids(tokenizer)
    max_source_len = max(len(feature["input_ids"]) for feature in features)
    max_target_len = max(len(feature["labels"]) for feature in features)
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    for feature in features:
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        labels = feature["labels"]
        source_padding = max_source_len - len(input_ids)
        target_padding = max_target_len - len(labels)
        padded_input_ids = input_ids + [pad_id] * source_padding
        padded_attention_mask = attention_mask + [0] * source_padding
        padded_labels = labels + [-100] * target_padding
        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        batch_labels.append(padded_labels)
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }

def build_training_args(output_dir: Path, learning_rate: float) -> TrainingArguments:
    desired_args = {
        "output_dir": str(output_dir),
        "num_train_epochs": TRAIN_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACC_STEPS,
        "learning_rate": learning_rate,
        "logging_steps": 20,
        "save_strategy": "no",
        "fp16": USE_FP16,
        "bf16": False,
        "report_to": [],
        "seed": SEED,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "optim": OPTIM,
    }
    try:
        allowed_params = set(inspect.signature(TrainingArguments.__init__).parameters)
        allowed_params.discard("self")
        desired_args = {
            key: value
            for key, value in desired_args.items()
            if key in allowed_params
        }
    except Exception:
        desired_args.pop("overwrite_output_dir", None, None)
    if "output_dir" not in desired_args:
        desired_args["output_dir"] = str(output_dir)
    return TrainingArguments(**desired_args)

def tokenized_example_is_valid(example: Dict[str, Any]) -> bool:
    input_ids = example.get("input_ids", [])
    labels = example.get("labels", [])
    if len(input_ids) == 0 or len(labels) == 0:
        return False
    try:
        return any(int(label) != -100 for label in labels)
    except Exception:
        return False

def train_model(
    model,
    tokenizer,
    train_dataset,
    model_type: str,
    output_dir: Path,
    learning_rate: float,
) -> None:
    print(f"\nНачинаем обучение модели: {model_type}")
    print(f"Размер тренировочной выборки: {len(train_dataset)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size per device: {TRAIN_BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRAD_ACC_STEPS}")
    print(f"Effective batch size: {TRAIN_BATCH_SIZE * GRAD_ACC_STEPS}")
    tokenizer.padding_side = "right"
    if model_type == "causal":
        tokenized_dataset = train_dataset.map(
            tokenize_causal_batch,
            batched=True,
            batch_size=1000,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=train_dataset.column_names,
            num_proc=1,
        )
        data_collator = lambda features: causal_collate(features, tokenizer=tokenizer)
    elif model_type == "seq2seq":
        tokenized_dataset = train_dataset.map(
            tokenize_seq2seq_batch,
            batched=True,
            batch_size=1000,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=train_dataset.column_names,
            num_proc=1,
        )
        data_collator = lambda features: seq2seq_collate(features, tokenizer=tokenizer)
    else:
        raise ValueError(f"Неизвестный model_type: {model_type}")
    tokenized_dataset = tokenized_dataset.filter(
        tokenized_example_is_valid,
        num_proc=1,
    )
    if len(tokenized_dataset) == 0:
        raise RuntimeError("После токенизации и фильтрации тренировочный датасет пуст.")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = build_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    print(f"\nСохраняем обученную модель в: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

def safe_generate(model, gen_kwargs: Dict[str, Any]):
    optional_keys = [
        "min_new_tokens",
        "num_beams",
        "no_repeat_ngram_size",
        "do_sample",
        "length_penalty",
    ]
    while True:
        try:
            return model.generate(**gen_kwargs)
        except TypeError as exc:
            msg = str(exc)
            changed = False
            for key in list(optional_keys):
                if key in gen_kwargs and key in msg:
                    gen_kwargs.pop(key, None)
                    changed = True
            if not changed:
                raise exc

def encode_causal_prompts_leftpad(
    texts: List[str],
    tokenizer,
):
    bos_id = tokenizer.bos_token_id
    pad_id, _ = get_pad_and_eos_ids(tokenizer)
    encoded = []
    max_len = 0
    for text in texts:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        )["input_ids"]
        if bos_id is not None:
            if len(ids) == 0 or ids[0] != bos_id:
                ids = [bos_id] + ids
        encoded.append(ids)
        max_len = max(max_len, len(ids))
    input_ids = []
    attention_mask = []
    for ids in encoded:
        pad_len = max_len - len(ids)
        input_ids.append([pad_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

def generate_predictions(
    model,
    tokenizer,
    inputs: List[str],
    model_type: str,
    prompt_func,
    batch_size: int = EVAL_BATCH_SIZE,
) -> List[str]:
    model.eval()
    predictions: List[str] = []
    device = next(model.parameters()).device
    pad_id, eos_id = get_pad_and_eos_ids(tokenizer)
    min_new_tokens = min(MIN_NEW_TOKENS, MAX_NEW_TOKENS)
    if model_type == "causal":
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            prompts = [prompt_func(text) for text in chunk]
            encoded = encode_causal_prompts_leftpad(prompts, tokenizer)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output_ids = safe_generate(
                    model,
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "min_new_tokens": min_new_tokens,
                        "do_sample": True,
                        "top_p": 0.9,
                        "temperature": 0.7,
                        "no_repeat_ngram_size": 2,
                        "pad_token_id": pad_id,
                        "eos_token_id": eos_id,
                    },
                )
            prompt_len = encoded["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_len:]
            decoded = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            for text in decoded:
                predictions.append(postprocess_generation(text))
        tokenizer.padding_side = old_padding_side
    elif model_type == "seq2seq":
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            prompts = [prompt_func(text) for text in chunk]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=MAX_SOURCE_LEN,
                add_special_tokens=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                generated_ids = safe_generate(
                    model,
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "min_new_tokens": min_new_tokens,
                        "do_sample": False,
                        "num_beams": ENCODER_BEAM_SIZE,
                        "no_repeat_ngram_size": 4,
                        "length_penalty": 1.0,
                        "early_stopping": True,
                        "pad_token_id": pad_id,
                        "eos_token_id": eos_id,
                    },
                )
            decoded = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            for text in decoded:
                predictions.append(postprocess_generation(text))
        tokenizer.padding_side = old_padding_side
    else:
        raise ValueError(f"Неизвестный model_type: {model_type}")
    return predictions

def evaluate_model(
    model,
    tokenizer,
    model_type: str,
    validation_dataset,
    stage_name: str,
    prompt_func,
    batch_size: int = EVAL_BATCH_SIZE,
    examples_to_show: int = 8,
) -> Dict[str, Any]:
    print("\n==============================")
    print(f"Оценка: {stage_name}")
    print("==============================")
    if validation_dataset is None or len(validation_dataset) == 0:
        print("Примеров нет, возвращаем нули.")
        return {
            "bleu": 0.0,
            "meteor": 0.0,
            "num_examples": 0,
            "avg_pred_words": 0.0,
            "avg_ref_words": 0.0,
            "examples": [],
        }
    inputs = validation_dataset["input"]
    references = validation_dataset["target"]
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        model_type=model_type,
        prompt_func=prompt_func,
        batch_size=batch_size,
    )
    bleu = compute_bleu(predictions, references)
    meteor = compute_meteor(predictions, references)
    avg_pred_words = float(np.mean([len(p.split()) for p in predictions])) if predictions else 0.0
    avg_ref_words = float(np.mean([len(r.split()) for r in references])) if references else 0.0
    print(f"BLEU = {bleu:.2f}")
    print(f"METEOR = {meteor:.2f}")
    print(f"Avg prediction length, words = {avg_pred_words:.2f}")
    print(f"Avg reference length, words = {avg_ref_words:.2f}")
    examples = []
    print("\nПримеры:")
    for i in range(min(examples_to_show, len(inputs))):
        example = {
            "input": inputs[i],
            "reference": references[i],
            "prediction": predictions[i],
        }
        examples.append(example)
        print("\n" + "-" * 80)
        print(f"[{i}] Context:")
        print(example["input"])
        print("\n Gold response:")
        print(example["reference"])
        print("\n Model response:")
        print(example["prediction"])
    result = {
        "bleu": bleu,
        "meteor": meteor,
        "num_examples": len(inputs),
        "avg_pred_words": avg_pred_words,
        "avg_ref_words": avg_ref_words,
        "examples": examples,
    }
    return result

def save_report(results: Dict[str, Any]) -> None:
    output_json = OUTPUT_DIR / "metrics_report.json"
    output_csv = OUTPUT_DIR / "metrics_table.csv"
    output_md = OUTPUT_DIR / "metrics_table.md"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    rows = []
    for model_name, model_result in results.items():
        for stage_key in ["zero_shot", "fine_tuned", "test"]:
            if stage_key in model_result:
                rows.append(
                    {
                        "model": model_name,
                        "stage": stage_key,
                        "BLEU": float(model_result[stage_key]["bleu"]),
                        "METEOR": float(model_result[stage_key]["meteor"]),
                        "AvgPredWords": float(model_result[stage_key].get("avg_pred_words", 0.0)),
                        "AvgRefWords": float(model_result[stage_key].get("avg_ref_words", 0.0)),
                        "NumExamples": int(model_result[stage_key].get("num_examples", 0)),
                    }
                )
    fieldnames = [
        "model",
        "stage",
        "BLEU",
        "METEOR",
        "AvgPredWords",
        "AvgRefWords",
        "NumExamples",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    md_lines = [
        "# Результаты лабораторной работы",
        "",
        f"Device: {DEVICE}",
        f"Dataset used: {DATASET_NAME_USED}",
        f"Decoder-only model: {DECODER_ONLY_MODEL_ID}",
        f"Encoder-decoder model: {ENCODER_DECODER_MODEL_ID}",
        "",
        "| model | stage | BLEU | METEOR | AvgPredWords | AvgRefWords | NumExamples |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['model']} | {row['stage']} | "
            f"{row['BLEU']:.2f} | {row['METEOR']:.2f} | "
            f"{row['AvgPredWords']:.2f} | {row['AvgRefWords']:.2f} | "
            f"{row['NumExamples']} |"
        )
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nОтчет сохранен в JSON: {output_json}")
    print(f"Таблица метрик сохранена в CSV: {output_csv}")
    print(f"Таблица метрик сохранена в Markdown: {output_md}")

def run_for_model(
    model_name: str,
    model_id: str,
    model_type: str,
    prompt_func,
    train_dataset,
    validation_dataset,
    results: Dict[str, Any],
    learning_rate: float,
    test_dataset=None,
) -> None:
    print("\n\n######################################################################")
    print(f"Модель: {model_name}")
    print(f"model_id: {model_id}")
    print(f"model_type: {model_type}")
    print("######################################################################\n")
    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(model_id, model_type)
    zero_shot_result = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        validation_dataset=validation_dataset,
        stage_name=f"{model_name}: исходная модель, zero-shot",
        prompt_func=prompt_func,
    )
    zero_shot_result["model_id"] = model_id
    zero_shot_result["model_type"] = model_type
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        model_type=model_type,
        output_dir=output_dir,
        learning_rate=learning_rate,
    )
    fine_tuned_result = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        validation_dataset=validation_dataset,
        stage_name=f"{model_name}: дообученная модель, validation",
        prompt_func=prompt_func,
    )
    fine_tuned_result["model_id"] = model_id
    fine_tuned_result["model_type"] = model_type
    test_result = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_result = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            validation_dataset=test_dataset,
            stage_name=f"{model_name}: дообученная модель, test",
            prompt_func=prompt_func,
        )
        test_result["model_id"] = model_id
        test_result["model_type"] = model_type
    model_results = {
        "model_id": model_id,
        "model_type": model_type,
        "learning_rate": learning_rate,
        "output_dir": str(output_dir),
        "zero_shot": zero_shot_result,
        "fine_tuned": fine_tuned_result,
    }
    if test_result is not None:
        model_results["test"] = test_result
    results[model_name] = model_results
    save_report(results)
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    print(f"DEVICE: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA memory total, GB: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
        )
    download_nltk_resources()
    train_pairs, valid_pairs, test_pairs = prepare_data()
    print(f"\nИспользован dataset: {DATASET_NAME_USED}")
    results: Dict[str, Any] = {}
    run_for_model(
        model_name="decoder_only_opt125m",
        model_id=DECODER_ONLY_MODEL_ID,
        model_type="causal",
        prompt_func=causal_prompt,
        train_dataset=train_pairs,
        validation_dataset=valid_pairs,
        results=results,
        learning_rate=DECODER_LEARNING_RATE,
        test_dataset=test_pairs,
    )
    run_for_model(
        model_name="encoder_decoder_flan_t5_small",
        model_id=ENCODER_DECODER_MODEL_ID,
        model_type="seq2seq",
        prompt_func=seq2seq_prompt,
        train_dataset=train_pairs,
        validation_dataset=valid_pairs,
        results=results,
        learning_rate=ENCODER_LEARNING_RATE,
        test_dataset=test_pairs,
    )
    print("\n\nИтоговая таблица метрик:")
    print("| model | stage | BLEU | METEOR |")
    print("|---|---|---:|---:|")
    for model_name, model_result in results.items():
        for stage_key in ["zero_shot", "fine_tuned", "test"]:
            if stage_key in model_result:
                bleu = model_result[stage_key]["bleu"]
                meteor = model_result[stage_key]["meteor"]
                print(
                    f"| {model_name} | {stage_key} | {bleu:.2f} | {meteor:.2f} |"
                )
    print("\nВсе шаги лабораторной завершены.")

if __name__ == "__main__":
    main()