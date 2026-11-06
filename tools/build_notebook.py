"""Собирает notebooks/omni_assistant.ipynb — основной ноутбук задания (запускается в Colab на GPU):

    .venv/bin/python -m tools.build_notebook

Локальная проверка кода на Mac без CUDA (Qwen3-VL-2B без квантования, урезанные эксперименты):

    OMNI_SMOKE=1 .venv/bin/jupyter nbconvert --to notebook --execute notebooks/omni_assistant.ipynb \
        --output omni_assistant_smoke.ipynb --ExecutePreprocessor.timeout=3600
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

GITHUB = "https://github.com/AlexIntellySoft/llm-8"
COLAB = "https://colab.research.google.com/github/AlexIntellySoft/llm-8/blob/main/notebooks/omni_assistant.ipynb"

cells = [
md(f"""# Мультимодальный Omni-ассистент: Whisper → Qwen3-VL → Silero

Прототип голосового помощника, который **слышит** вопрос, **видит** изображение и **отвечает голосом**.

```
 аудио-вопрос ──► [ASR] Whisper small ──► текст вопроса ─┐
   (wav/mic)      + шумоподавление                       ▼
 изображение ───────────────────────────────► [VLM] Qwen3-VL-8B-Instruct (4-bit NF4)
                                                         │  ответ 2–3 предложения
                                                         ▼
                              ответ-аудио ◄── [TTS] Silero v4_ru (kseniya) ◄── нормализация текста
```

Все три модели живут на одной видеокарте (T4, 15 ГБ). Ноутбук выполняется сверху вниз:

| § | что | требование задания |
| --- | --- | --- |
| 0 | окружение, GPU, данные | п. 1 |
| 1 | тестовые пары «изображение + аудио» | п. 1 |
| 2 | «Уши»: Whisper + шумоподавление | п. 2 |
| 3 | «Глаза»: Qwen3-VL в 4-bit, промпт голосового ассистента | п. 2 |
| 4 | «Голос»: Silero TTS | п. 3 |
| 5 | «Сердце»: `run_omni_assistant(image, audio_query)` + latency, демонстрация вход → тексты → аудио | п. 3, сдача |
| 6 | Gradio-приложение и его скриншот | п. 4, п. 5.1 |
| 7 | таблица latency и стриминговый режим | п. 5.2 |
| 8 | «глухота»: шум в аудио | п. 5, вопрос 1 |
| 9 | «слепота»: мелкий текст на этикетке | п. 5, вопрос 2 |
| 10 | выводы | п. 5, вопрос 3 |

Код и данные: [{GITHUB}]({GITHUB}) · [открыть в Colab]({COLAB})"""),

md("""## 0. Окружение

Нужна видеокарта NVIDIA: `bitsandbytes` квантует в 4 бита только на CUDA. В Colab:
*Среда выполнения → Сменить среду выполнения → T4 GPU*.

Переменная `OMNI_SMOKE=1` включает локальную проверку кода на Mac: вместо 8B берётся
Qwen3-VL-2B без квантования, эксперименты урезаны. Сдаётся прогон в Colab, не smoke."""),
code(r'''import os, sys, json, re, time, math, gc, uuid, warnings
from contextlib import contextmanager
from pathlib import Path

SMOKE = os.environ.get("OMNI_SMOKE") == "1"
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
warnings.filterwarnings("ignore")
print(f"Colab: {IN_COLAB}, smoke-режим: {SMOKE}")'''),
code(r'''if IN_COLAB:
    %pip install -q "transformers==5.22.0" "accelerate==1.15.0" "bitsandbytes==0.50.2" qwen-vl-utils openai-whisper noisereduce jiwer num2words omegaconf gradio playwright
    !playwright install --with-deps chromium > /dev/null 2>&1'''),
code(r'''REPO_URL = "https://github.com/AlexIntellySoft/llm-8"
if IN_COLAB:
    if not Path("/content/llm-8").exists():
        !git clone -q {REPO_URL} /content/llm-8
    ROOT = Path("/content/llm-8")
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA, OUT = ROOT / "data", ROOT / ("outputs_smoke" if SMOKE else "outputs")
ANSWERS = OUT / "answers"
ANSWERS.mkdir(parents=True, exist_ok=True)
META = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
PAIRS = META["pairs"]
print(ROOT, "|", len(PAIRS), "тестовых пар")'''),
code(r'''import numpy as np
import pandas as pd
import torch
import transformers

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} ГБ VRAM")
else:
    print("CUDA нет, устройство:", DEVICE)
print("torch", torch.__version__, "| transformers", transformers.__version__)
pd.set_option("display.max_colwidth", None)

def sync():
    """Дождаться GPU, иначе замер времени покажет только постановку ядер в очередь."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

def vram_gb():
    return torch.cuda.memory_allocated() / 2**30 if DEVICE == "cuda" else float("nan")

class Timer:
    """with timer("asr"): ...  → timer.t["asr"] в секундах."""
    def __init__(self):
        self.t = {}

    @contextmanager
    def __call__(self, name):
        sync()
        t0 = time.perf_counter()
        yield
        sync()
        self.t[name] = time.perf_counter() - t0

LOAD = Timer()      # время загрузки моделей
VRAM = {}           # сколько памяти заняла каждая модель'''),

md("""## 1. Тестовые данные: 5 пар «изображение + аудио-вопрос»

* 4 фотографии с Wikimedia Commons (лицензии в `data/images/LICENSES.md`);
* синтетическая этикетка йогурта `label.png`: текст от 64 до 7 px, эталон известен построчно,
  на ней в §9 меряется чтение мелких деталей;
* вопросы озвучены голосом Milena (macOS `say`, 16 кГц) скриптом `tools/make_test_data.py`.
  Голос вопроса намеренно другой, не Silero, чтобы ASR не слушал сам себя.

В `data/pairs.json` для каждой пары лежит эталонный текст вопроса (для WER) и ключевые слова
правильного ответа (для автоматической проверки, что ответ по делу)."""),
code(r'''from IPython.display import Audio, Image as IPImage, Markdown, display
from PIL import Image
import soundfile as sf

for p in PAIRS:
    info = sf.info(str(DATA / p["audio"]))
    display(Markdown(f"**{p['id']}** · вопрос: «{p['question']}» · аудио {info.duration:.1f} с"))
    img = Image.open(DATA / p["image"]); img.thumbnail((360, 360)); display(img)
    display(Audio(str(DATA / p["audio"])))'''),

md("""## 2. «Уши»: Whisper small + шумоподавление

`transcribe(path)` читает файл, приводит его к 16 кГц моно, чистит и отдаёт текст Whisper.

Шумоподавление в два шага:
1. **high-pass 80 Гц**: срезает гул (кондиционер, ветер, вибрация), в котором речи нет;
2. **спектральное вычитание** (`noisereduce`, нестационарный режим): по каждой частотной полосе
   оценивается уровень шума и приглушаются участки ниже порога.

Whisper запускается с `language="ru"` (иначе на коротком шумном вопросе он может решить, что
язык другой), `temperature=0` и `condition_on_previous_text=False`.

**Защита от галлюцинаций.** На тишине и шуме Whisper иногда пишет фразы из обучающих субтитров
(«Продолжение следует…», «Субтитры сделал…»). Поэтому `asr_transcribe` возвращает и признаки
уверенности: `no_speech_prob`, средний `avg_logprob` и совпадение с известными галлюцинациями.
Если расшифровка ненадёжна, ассистент не отвечает на выдуманный вопрос (см. §5 и §8)."""),
code(r'''import whisper
import noisereduce as nr
from scipy.signal import butter, resample_poly, sosfilt

SR = 16000
ASR_SIZE = "small"

with LOAD("asr"):
    asr_model = whisper.load_model(ASR_SIZE, device="cuda" if DEVICE == "cuda" else "cpu")
VRAM["asr"] = vram_gb()
print(f"Whisper {ASR_SIZE}: {LOAD.t['asr']:.1f} с загрузки, VRAM {VRAM['asr']:.2f} ГБ")

def load_audio(path, sr=SR):
    """Любой wav/flac/ogg/mp3 → float32 моно sr Гц. Для экзотики (m4a, webm) — ffmpeg через whisper."""
    try:
        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return whisper.load_audio(str(path), sr=sr)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        g = math.gcd(file_sr, sr)
        wav = resample_poly(wav, sr // g, file_sr // g).astype(np.float32)
    return wav

_HIGHPASS = butter(4, 80, btype="highpass", fs=SR, output="sos")

def denoise(wav, sr=SR):
    wav = sosfilt(_HIGHPASS, wav)
    wav = nr.reduce_noise(y=wav, sr=sr, stationary=False, prop_decrease=0.85)
    peak = np.abs(wav).max()
    return (wav / peak * 0.95 if peak > 0 else wav).astype(np.float32)

HALLUCINATIONS = re.compile(r"субтитр|продолжение следует|спасибо за просмотр|подпис|dimatorzok|редактор", re.I)

def asr_transcribe(audio, denoise_audio=True):
    """audio — путь к файлу или numpy-массив 16 кГц. Возвращает текст и признаки надёжности."""
    wav = load_audio(audio) if isinstance(audio, (str, Path)) else np.asarray(audio, dtype=np.float32)
    if denoise_audio:
        wav = denoise(wav)
    r = asr_model.transcribe(wav, language="ru", task="transcribe", fp16=DEVICE == "cuda",
                             temperature=0.0, condition_on_previous_text=False, verbose=None)
    text = r["text"].strip()
    segs = r["segments"]
    no_speech = float(np.mean([s["no_speech_prob"] for s in segs])) if segs else 1.0
    logprob = float(np.mean([s["avg_logprob"] for s in segs])) if segs else -10.0
    hallucination = bool(HALLUCINATIONS.search(text))
    reliable = bool(text) and no_speech < 0.6 and logprob > -1.0 and not hallucination
    return {"text": text, "no_speech_prob": no_speech, "avg_logprob": logprob,
            "hallucination": hallucination, "reliable": reliable}

def transcribe(audio_path):
    """Функция из задания: путь к аудио → текст на русском (с шумоподавлением)."""
    return asr_transcribe(audio_path)["text"]'''),
code(r'''import jiwer

def norm_text(s):
    s = s.lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())

def wer(ref, hyp):
    ref, hyp = norm_text(ref), norm_text(hyp)
    return jiwer.wer(ref, hyp) if hyp else 1.0

rows = []
for p in PAIRS:
    t0 = time.perf_counter()
    r = asr_transcribe(DATA / p["audio"])
    rows.append({"пара": p["id"], "эталон": p["question"], "Whisper": r["text"],
                 "WER": round(wer(p["question"], r["text"]), 3), "надёжно": r["reliable"],
                 "сек": round(time.perf_counter() - t0, 2)})
asr_clean = pd.DataFrame(rows)
asr_clean'''),

md("""## 3. «Глаза»: Qwen3-VL-8B-Instruct в 4 бита

**Квантование** — `bitsandbytes` NF4 + double quantization: веса 8B-модели занимают ≈6 ГБ вместо ≈17 ГБ
в bf16, поэтому Whisper, Qwen и Silero помещаются на одну T4. Вычисления идут в fp16 на T4
и в bf16 на Ampere+ (A100, L4).

**Разрешение картинки.** Qwen3-VL режет изображение на патчи 16×16 и сливает их 2×2, то есть
один визуальный токен покрывает 32×32 px. По умолчанию ассистент получает до ≈1 Мп
(`IMAGE_MAX_PIXELS`, ≈1000 токенов): это компромисс между скоростью и чтением мелкого текста,
в §9 он проверяется экспериментом. Размер задаётся через `qwen_vl_utils.process_vision_info`
(ключи `max_pixels` / `min_pixels` в сообщении).

**Промпт голосового ассистента.** Ответ будут слушать, а не читать, поэтому промпт требует 2–3
коротких предложения, без списков и markdown. Вопрос пришёл из ASR и может содержать ошибки,
поэтому модели разрешено восстанавливать смысл по картинке. Для пустого или неразборчивого вопроса
задан отдельный сценарий."""),
code(r'''from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

VLM_ID = "Qwen/Qwen3-VL-2B-Instruct" if SMOKE else "Qwen/Qwen3-VL-8B-Instruct"
IMAGE_MAX_PIXELS = 1024 * 1024

before = vram_gb()
with LOAD("vlm"):
    if DEVICE == "cuda":
        COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=COMPUTE_DTYPE)
        vlm = AutoModelForImageTextToText.from_pretrained(VLM_ID, quantization_config=bnb,
                                                          dtype=COMPUTE_DTYPE, device_map="cuda:0")
    else:  # smoke: без CUDA bitsandbytes не работает, берём маленькую модель в bf16
        vlm = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=torch.bfloat16).to(DEVICE)
    vlm.eval()
    processor = AutoProcessor.from_pretrained(VLM_ID)
VRAM["vlm"] = vram_gb() - before
IMAGE_TOKEN_ID = getattr(vlm.config, "image_token_id", None)
print(f"{VLM_ID}: {LOAD.t['vlm']:.1f} с загрузки, VRAM {VRAM['vlm']:.2f} ГБ")'''),
code(r'''SYSTEM_PROMPT = """Ты — голосовой ассистент «Омни». Пользователь показывает изображение и задаёт вопрос голосом.
Вопрос получен автоматическим распознаванием речи и может содержать ошибки: восстанавливай смысл по изображению.
Правила ответа:
- отвечай по-русски, максимум 2–3 коротких предложения: ответ будут слушать, а не читать;
- говори разговорно и по делу, сначала прямой ответ на вопрос;
- никаких списков, markdown, таблиц, эмодзи и ссылок;
- если мелкие детали (текст, цифры) читаются неуверенно, честно скажи об этом;
- если вопрос пустой или бессмысленный, коротко скажи, что видишь на изображении, и попроси повторить вопрос."""

UNCLEAR_QUESTION = ("[Распознать вопрос не удалось: в аудио шум или тишина.] "
                    "Коротко скажи, что видишь на изображении, и попроси повторить вопрос.")

def _vision_inputs(messages):
    try:   # qwen-vl-utils >= 0.0.14 знает про патч 16 px у Qwen3-VL
        return process_vision_info(messages, image_patch_size=16)[:2], False
    except TypeError:
        return process_vision_info(messages)[:2], True

def ask_vlm(image, question, system=SYSTEM_PROMPT, max_new_tokens=160,
            max_pixels=IMAGE_MAX_PIXELS, min_pixels=None, streamer=None):
    """image — PIL или путь. Возвращает (текст ответа, статистика по токенам)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    vis = {"type": "image", "image": image.convert("RGB"), "max_pixels": max_pixels}
    if min_pixels:
        vis["min_pixels"] = min_pixels
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [vis, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    (images, videos), legacy = _vision_inputs(messages)
    inputs = processor(text=[prompt], images=images, videos=videos, do_resize=legacy,
                       return_tensors="pt").to(vlm.device)
    with torch.inference_mode():
        out = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
    new = out[0, inputs["input_ids"].shape[1]:]
    stats = {"prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": int(len(new)),
             "image_tokens": int((inputs["input_ids"] == IMAGE_TOKEN_ID).sum()) if IMAGE_TOKEN_ID else None,
             "image_size": images[0].size}
    return processor.decode(new, skip_special_tokens=True).strip(), stats

def for_speech(text, max_sentences=3):
    """Страховка поверх промпта: убрать разметку и оставить не больше 3 предложений."""
    text = re.sub(r"[*_#`>|]+", " ", text)
    text = re.sub(r"^\s*[-•\d]+[.)]?\s+", "", text, flags=re.M)
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    return " ".join(sentences[:max_sentences])'''),
code(r'''# VLM отдельно, на эталонных текстах вопросов (без ошибок ASR)
rows = []
for p in PAIRS:
    t0 = time.perf_counter()
    answer, st = ask_vlm(DATA / p["image"], p["question"])
    rows.append({"пара": p["id"], "вопрос": p["question"], "ответ Qwen": answer,
                 "токенов картинки": st["image_tokens"], "новых токенов": st["new_tokens"],
                 "сек": round(time.perf_counter() - t0, 2)})
vlm_clean = pd.DataFrame(rows)
vlm_clean'''),

md("""## 4. «Голос»: Silero TTS v4_ru

Silero — маленькая (≈60 МБ) модель, синтезирует быстрее реального времени даже на CPU.
Голоса `v4_ru`: `aidar`, `eugene` (мужские), `baya`, `kseniya`, `xenia` (женские). Основной голос
ассистента — **kseniya**.

Silero читает только кириллицу: цифры, латиницу и символы он пропускает. Поэтому перед синтезом
`normalize_for_tts` разворачивает «14.11.2026» в «четырнадцатого ноября две тысячи двадцать шестого
года», «2,5%» — в «два целых пять десятых процентов», «£2» — в «два фунтов» (с ошибкой падежа, но
понятно на слух), а латиницу грубо транслитерирует («Coca-Cola» → «Кока-Кола»)."""),
code(r'''from num2words import num2words

TTS_SR = 48000
SILERO_SPEAKER = "kseniya"
tts_device = torch.device("cuda" if DEVICE == "cuda" else "cpu")
before = vram_gb()
with LOAD("tts"):
    tts_model, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru",
                                  speaker="v4_ru", trust_repo=True)
    tts_model.to(tts_device)
VRAM["tts"] = vram_gb() - before
print(f"Silero v4_ru: {LOAD.t['tts']:.1f} с загрузки, VRAM {VRAM['tts']:.2f} ГБ, голоса: {tts_model.speakers}")

MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря"]
SYMBOLS = {"%": " процентов", "°C": " градусов", "°": " градусов", "₽": " рублей", "€": " евро",
           "№": " номер ", "&": " и ", "+": " плюс "}
CURRENCY_BEFORE = {"£": "фунтов", "$": "долларов", "€": "евро"}
LATIN = [("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("oo", "у"), ("ee", "и"), ("ck", "к"),
         ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("h", "х"),
         ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
         ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("x", "кс"),
         ("y", "й"), ("z", "з")]

def _genitive_ordinal(n):
    w = num2words(n, lang="ru", to="ordinal")
    return re.sub(r"(ый|ой)$", "ого", re.sub(r"ий$", "его", w))

def _number(m):
    s = m.group(0).replace(",", ".")
    if "." in s:
        whole, frac = s.split(".")
        if set(frac) == {"0"}:
            return num2words(int(whole), lang="ru")
        return num2words(float(s), lang="ru")
    return num2words(int(s), lang="ru")

def _latin(word):
    w = word.group(0).lower()
    for a, b in LATIN:
        w = w.replace(a, b)
    return w

def normalize_for_tts(text):
    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b",
                  lambda m: f"{_genitive_ordinal(int(m[1]))} {MONTHS[int(m[2]) - 1]} {_genitive_ordinal(int(m[3]))} года"
                  if 1 <= int(m[2]) <= 12 else m[0], text)
    for sym, word in CURRENCY_BEFORE.items():
        text = re.sub(re.escape(sym) + r"\s?(\d+(?:[.,]\d+)?)", rf"\1 {word}", text)
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, word)
    text = re.sub(r"\d+(?:[.,]\d+)?", _number, text)
    text = re.sub(r"[A-Za-z]+", _latin, text)
    text = re.sub(r"[^А-Яа-яЁё0-9\s.,!?:;—\-«»()]", " ", text)
    return " ".join(text.split())

def synthesize(text, path=None, speaker=SILERO_SPEAKER):
    """Текст → wav-файл 48 кГц. Длинный текст режется по предложениям (лимит Silero ≈1000 символов)."""
    path = Path(path or ANSWERS / f"reply_{uuid.uuid4().hex[:8]}.wav")
    clean = normalize_for_tts(text) or "Извините, мне нечего ответить."
    chunks, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", clean):
        if len(cur) + len(sentence) > 800:
            chunks.append(cur); cur = ""
        cur = f"{cur} {sentence}".strip()
    chunks.append(cur)
    with torch.inference_mode():
        audio = torch.cat([tts_model.apply_tts(text=c, speaker=speaker, sample_rate=TTS_SR,
                                               put_accent=True, put_yo=True) for c in chunks if c])
    audio = audio.cpu().numpy()
    sf.write(str(path), audio, TTS_SR)
    return str(path), len(audio) / TTS_SR

print(normalize_for_tts("Годен до 14.11.2026, жирность 2,5%, сахар 11,0 г. Coca-Cola за £2."))'''),
code(r'''# Пробуем голоса на одной фразе: выбран kseniya
sample = "Привет! Я голосовой ассистент Омни. Покажите мне фото и задайте вопрос."
for spk in ["aidar", "eugene", "baya", "kseniya", "xenia"]:
    path, dur = synthesize(sample, ANSWERS / f"voice_{spk}.wav", speaker=spk)
    display(Markdown(f"`{spk}` — {dur:.1f} с")); display(Audio(path))'''),

md("""## 5. «Сердце»: `run_omni_assistant(image, audio_query)`

Оркестратор: аудио → Whisper → (вопрос + картинка) → Qwen3-VL → нормализация → Silero → wav.
Каждый этап обёрнут в `Timer` с `torch.cuda.synchronize()`. Возвращает текст ответа и путь к
аудио; с `return_trace=True` ещё и трассу: расшифровку, время этапов, токены.

Если ASR признал расшифровку ненадёжной (шум, тишина, галлюцинация), в Qwen уходит не выдуманный
вопрос, а служебная пометка `UNCLEAR_QUESTION`: ассистент опишет картинку и попросит повторить."""),
code(r'''LATENCY_LOG = []

def run_omni_assistant(image, audio_query, speaker=SILERO_SPEAKER, out_path=None,
                       use_guard=True, return_trace=False):
    timer = Timer()
    with timer("asr"):
        asr = asr_transcribe(audio_query)
    question = asr["text"] if (asr["reliable"] or not use_guard) else UNCLEAR_QUESTION
    with timer("vlm"):
        raw, st = ask_vlm(image, question)
    answer = for_speech(raw)
    with timer("tts"):
        audio_path, audio_dur = synthesize(answer, out_path, speaker=speaker)
    trace = {"transcript": asr["text"], "asr_reliable": asr["reliable"], "question_to_vlm": question,
             "answer": answer, "audio": audio_path, "answer_audio_s": round(audio_dur, 2),
             "asr_s": timer.t["asr"], "vlm_s": timer.t["vlm"], "tts_s": timer.t["tts"],
             "total_s": sum(timer.t.values()), **st,
             "vlm_tok_per_s": st["new_tokens"] / timer.t["vlm"]}
    LATENCY_LOG.append(trace)
    return (answer, audio_path, trace) if return_trace else (answer, audio_path)

# прогрев: первый вызов CUDA-ядер и кэшей заметно медленнее, в статистику не идёт
t0 = time.perf_counter()
run_omni_assistant(DATA / PAIRS[0]["image"], DATA / PAIRS[0]["audio"], out_path=ANSWERS / "warmup.wav")
COLD_START_S = time.perf_counter() - t0
LATENCY_LOG.clear()
print(f"Холодный первый запрос: {COLD_START_S:.1f} с")'''),
md("""### Демонстрация: входные файлы → промежуточные тексты → аудио-ответ"""),
code(r'''demo_rows = []
for p in PAIRS:
    answer, audio_path, tr = run_omni_assistant(DATA / p["image"], DATA / p["audio"],
                                                out_path=ANSWERS / f"answer_{p['id']}.wav", return_trace=True)
    hit = any(k in answer.lower() for k in p["keywords"])
    demo_rows.append({"пара": p["id"], **tr, "по делу": hit})
    display(Markdown(f"---\n### {p['id']}"))
    img = Image.open(DATA / p["image"]); img.thumbnail((420, 420)); display(img)
    display(Markdown("**Вход: аудио-вопрос**")); display(Audio(str(DATA / p["audio"])))
    display(Markdown(f"**Whisper:** {tr['transcript']}  \n"
                     f"**Qwen3-VL:** {answer}  \n"
                     f"**Latency:** ASR {tr['asr_s']:.2f} с · VLM {tr['vlm_s']:.2f} с · "
                     f"TTS {tr['tts_s']:.2f} с · **всего {tr['total_s']:.2f} с** · ключевые слова: {'да' if hit else 'нет'}"))
    display(Markdown("**Выход: голосовой ответ (Silero)**")); display(Audio(audio_path))
with open(OUT / "results.jsonl", "w", encoding="utf-8") as f:
    for r in demo_rows:
        f.write(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}, ensure_ascii=False) + "\n")'''),

md("""## 6. Gradio-приложение

Два входа (изображение; запись с микрофона или загрузка аудио) и два основных выхода (текст ответа;
плеер с озвученным ответом, автовоспроизведение). Дополнительно видны распознанный вопрос и время
этапов: так сразу понятно, ошибся ли ассистент «ушами» или «глазами».

В Colab `share=True` даёт публичную ссылку `*.gradio.live` (живёт 72 часа). Скриншот снимается
автоматически headless-браузером Playwright: он загружает пару «рынок», жмёт «Спросить» и ждёт ответ."""),
code(r'''import gradio as gr

def gradio_handler(image, audio):
    if image is None or audio is None:
        raise gr.Error("Нужны и изображение, и аудио-вопрос.")
    answer, path, tr = run_omni_assistant(image, audio, return_trace=True)
    latency = (f"ASR **{tr['asr_s']:.2f} с** · VLM **{tr['vlm_s']:.2f} с** · TTS **{tr['tts_s']:.2f} с** · "
               f"всего **{tr['total_s']:.2f} с**")
    transcript = tr["transcript"] + ("" if tr["asr_reliable"] else "  ⚠ ненадёжно, вопрос не использован")
    return transcript, answer, path, latency

with gr.Blocks(title="Омни-ассистент") as demo:
    gr.Markdown("# Омни-ассистент\nПокажите фото и задайте вопрос голосом: Whisper → Qwen3-VL → Silero.")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Изображение", elem_id="img_in", height=360)
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                label="Вопрос голосом", elem_id="audio_in")
            ask_btn = gr.Button("Спросить", variant="primary", elem_id="ask_btn")
        with gr.Column():
            transcript_out = gr.Textbox(label="Распознанный вопрос (Whisper)", elem_id="transcript")
            answer_out = gr.Textbox(label="Ответ ассистента", lines=4, elem_id="answer")
            audio_out = gr.Audio(label="Ответ голосом", type="filepath", autoplay=True, elem_id="audio_out")
            latency_out = gr.Markdown(elem_id="latency")
    gr.Examples([[str(DATA / p["image"]), str(DATA / p["audio"])] for p in PAIRS], inputs=[img_in, audio_in],
                label="Тестовые пары")
    ask_btn.click(gradio_handler, [img_in, audio_in], [transcript_out, answer_out, audio_out, latency_out])

demo.queue().launch(share=IN_COLAB, prevent_thread_lock=True, server_port=7860,
                    allowed_paths=[str(ROOT)], quiet=True)
APP_URL = demo.local_url
print("Локально:", APP_URL, "| публично:", getattr(demo, "share_url", None))'''),
code(r'''async def screenshot_app(url, pair, path):
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 1000})
        await page.goto(url)
        await page.wait_for_selector("#ask_btn")
        await page.locator("#img_in input[type=file]").set_input_files(str(DATA / pair["image"]))
        await page.locator("#audio_in input[type=file]").set_input_files(str(DATA / pair["audio"]))
        await page.wait_for_timeout(2000)
        await page.click("#ask_btn")
        await page.wait_for_function(
            "() => { const t = document.querySelector('#answer textarea'); return t && t.value.length > 0; }",
            timeout=300_000)
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(path), full_page=True)
        await browser.close()

SHOT = OUT / "gradio_screenshot.png"
try:
    await screenshot_app(APP_URL, next(p for p in PAIRS if p["id"] == "market"), SHOT)
except Exception as e:
    print("Автоскриншот не получился, сделайте его вручную и сохраните в outputs/gradio_screenshot.png:", e)
if SHOT.exists():
    display(IPImage(str(SHOT)))'''),

md("""## 7. Latency: сколько ждать голосового ответа

Каждая пара прогоняется `REPEATS` раз (после прогрева). Время считается от получения аудио-файла
до готового wav-ответа. Сеть и браузер Gradio сюда не входят."""),
code(r'''REPEATS = 1 if SMOKE else 3
LATENCY_LOG.clear()
for _ in range(REPEATS):
    for p in PAIRS:
        _, _, tr = run_omni_assistant(DATA / p["image"], DATA / p["audio"],
                                      out_path=ANSWERS / f"answer_{p['id']}.wav", return_trace=True)
        tr["пара"] = p["id"]

lat = pd.DataFrame(LATENCY_LOG)
per_pair = lat.groupby("пара", sort=False)[["asr_s", "vlm_s", "tts_s", "total_s", "image_tokens",
                                             "new_tokens", "vlm_tok_per_s", "answer_audio_s"]].mean()
per_pair.loc["среднее"] = per_pair.mean()
latency_table = per_pair.rename(columns={
    "asr_s": "ASR, с", "vlm_s": "VLM, с", "tts_s": "TTS, с", "total_s": "всего, с",
    "image_tokens": "токенов картинки", "new_tokens": "токенов ответа", "vlm_tok_per_s": "VLM ток/с",
    "answer_audio_s": "длина ответа, с"}).round(2)
lat.to_csv(OUT / "latency_runs.csv", index=False)
latency_table.to_csv(OUT / "latency.csv")
latency_table'''),
code(r'''import matplotlib.pyplot as plt

mean = per_pair.loc["среднее"]
share = pd.Series({"ASR (Whisper)": mean.asr_s, "VLM (Qwen3-VL)": mean.vlm_s, "TTS (Silero)": mean.tts_s})
display(Markdown(
    f"**Среднее время до голосового ответа: {mean.total_s:.2f} с.** Самый долгий этап: **{share.idxmax()}** "
    f"({share.max() / share.sum():.0%} времени). Генерация Qwen: {mean.vlm_tok_per_s:.1f} токенов/с. "
    f"Холодный первый запрос: {COLD_START_S:.1f} с."))

fig, ax = plt.subplots(figsize=(9, 3.6))
left = np.zeros(len(per_pair))
colors = {"asr_s": "#4C78A8", "vlm_s": "#E45756", "tts_s": "#54A24B"}
for col, name in [("asr_s", "ASR"), ("vlm_s", "VLM"), ("tts_s", "TTS")]:
    ax.barh(per_pair.index, per_pair[col], left=left, label=name, color=colors[col])
    left += per_pair[col].values
ax.invert_yaxis(); ax.set_xlabel("секунды"); ax.set_title("Время этапов на запрос")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(OUT / "latency.png", dpi=120); plt.show()

mem = pd.DataFrame({"модель": ["Whisper small", VLM_ID.split("/")[-1] + (" (bf16)" if DEVICE != "cuda" else " (NF4)"), "Silero v4_ru"],
                    "VRAM, ГБ": [VRAM["asr"], VRAM["vlm"], VRAM["tts"]],
                    "загрузка, с": [LOAD.t["asr"], LOAD.t["vlm"], LOAD.t["tts"]]}).round(2)
if DEVICE == "cuda":
    display(Markdown(f"Пик VRAM за ноутбук: **{torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ** из "
                     f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}"))
mem'''),
md("""### Стриминг: время до первого звука

В последовательной схеме пользователь ждёт, пока Qwen допишет **весь** ответ. Если читать токены
через `TextIteratorStreamer` и отправлять в Silero каждое готовое предложение, первый звук появляется
после первого предложения. Ниже это реализовано и замерено: **TTFA** (time to first audio) против
полного времени."""),
code(r'''from threading import Thread
from transformers import TextIteratorStreamer

def run_omni_streaming(image, audio_query, speaker=SILERO_SPEAKER):
    t0 = time.perf_counter()
    asr = asr_transcribe(audio_query)
    t_asr = time.perf_counter() - t0
    question = asr["text"] if asr["reliable"] else UNCLEAR_QUESTION
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    worker = Thread(target=ask_vlm, args=(image, question), kwargs={"streamer": streamer})
    worker.start()
    buf, pieces, ttfa, n_sent = "", [], None, 0
    def speak(sentence):
        nonlocal ttfa
        sentence = for_speech(sentence)
        if not sentence:
            return
        clean = normalize_for_tts(sentence)
        if clean:
            with torch.inference_mode():
                pieces.append(tts_model.apply_tts(text=clean, speaker=speaker, sample_rate=TTS_SR).cpu().numpy())
            ttfa = ttfa or time.perf_counter() - t0
    for token_text in streamer:
        buf += token_text
        while (m := re.search(r"[.!?…](\s+)", buf)) and n_sent < 3:
            speak(buf[:m.end()]); buf = buf[m.end():]; n_sent += 1
    if n_sent < 3:
        speak(buf)
    worker.join()
    total = time.perf_counter() - t0
    path = ANSWERS / f"stream_{uuid.uuid4().hex[:6]}.wav"
    sf.write(str(path), np.concatenate(pieces) if pieces else np.zeros(TTS_SR // 2), TTS_SR)
    return {"asr_s": t_asr, "ttfa_s": ttfa, "total_s": total, "audio": str(path)}

stream_rows = []
for p in PAIRS:
    r = run_omni_streaming(DATA / p["image"], DATA / p["audio"])
    stream_rows.append({"пара": p["id"], "первый звук (стриминг), с": r["ttfa_s"],
                        "полный ответ (стриминг), с": r["total_s"],
                        "последовательно, с": per_pair.loc[p["id"], "total_s"]})
stream = pd.DataFrame(stream_rows).set_index("пара")
stream.loc["среднее"] = stream.mean()
stream.round(2)'''),

md("""## 8. Анализ «глухоты»: что будет, если в аудио сильный шум

Эксперимент: к каждому вопросу подмешивается шум с заданным отношением сигнал/шум (SNR):
* **белый шум**: шипение, равномерный по частотам;
* **«кафе»**: 6 посторонних фраз, наложенных друг на друга (самый неприятный для ASR шум, потому что
  это тоже речь).

SNR 20 дБ — тихий офис, 10 — улица, 5 — шумное кафе, 0 — шум громкий как речь, −5 — шум громче
речи. Для каждого уровня Whisper запускается с шумоподавлением и без, считается WER."""),
code(r'''rng = np.random.default_rng(0)
speech = {p["id"]: load_audio(DATA / p["audio"]) for p in PAIRS}

def make_babble(seconds=30):
    voices = [load_audio(f) for f in sorted((DATA / "audio").glob("babble_*.wav"))]
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    for v in voices:                      # каждая фраза повторяется со своим сдвигом
        start = int(rng.integers(0, SR * 2))
        while start < len(out):
            seg = v[: len(out) - start]
            out[start:start + len(seg)] += seg
            start += len(v) + int(rng.integers(0, SR))
    return out

NOISES = {"белый": rng.normal(0, 1, SR * 30).astype(np.float32), "кафе": make_babble()}

def mix_at_snr(clean, noise, snr_db):
    start = int(rng.integers(0, len(noise) - len(clean)))
    n = noise[start:start + len(clean)]
    n = n * np.sqrt(np.mean(clean ** 2) / (np.mean(n ** 2) * 10 ** (snr_db / 10)))
    mix = clean + n
    return (mix / max(1.0, np.abs(mix).max() / 0.99)).astype(np.float32)

SNRS = [20, 5, -5] if SMOKE else [20, 10, 5, 0, -5]
noise_pairs = PAIRS[:2] if SMOKE else PAIRS
rows = []
for p in noise_pairs:
    for noise_name, noise in NOISES.items():
        for snr in SNRS:
            noisy = mix_at_snr(speech[p["id"]], noise, snr)
            if snr == 0 and noise_name == "кафе":
                sf.write(str(OUT / f"noisy_{p['id']}_cafe_0db.wav"), noisy, SR)
            for dn in (False, True):
                r = asr_transcribe(noisy, denoise_audio=dn)
                rows.append({"пара": p["id"], "шум": noise_name, "SNR, дБ": snr, "шумоподавление": dn,
                             "текст": r["text"], "WER": wer(p["question"], r["text"]), "надёжно": r["reliable"]})
asr_noise = pd.DataFrame(rows)
asr_noise.to_csv(OUT / "asr_noise.csv", index=False)
wer_table = asr_noise.pivot_table(index=["шум", "шумоподавление"], columns="SNR, дБ", values="WER",
                                  aggfunc="mean").sort_index(axis=1, ascending=False)
wer_table.round(2)'''),
code(r'''fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, noise_name in zip(axes, NOISES):
    for dn, style in [(False, "--o"), (True, "-o")]:
        s = wer_table.loc[(noise_name, dn)]
        ax.plot(s.index, s.values, style, label="с шумоподавлением" if dn else "без")
    ax.set_title(f"шум: {noise_name}"); ax.set_xlabel("SNR, дБ"); ax.invert_xaxis(); ax.grid(alpha=.3)
axes[0].set_ylabel("WER"); axes[0].legend(); plt.tight_layout(); plt.savefig(OUT / "asr_noise.png", dpi=120); plt.show()

example = asr_noise[(asr_noise["шумоподавление"]) & (asr_noise["шум"] == "кафе")][["пара", "SNR, дБ", "текст", "WER"]]
example.round(2)'''),
md("""### Догадается ли ассистент по картинке, о чём его спросили?

Для каждой пары берём «кафе» при 0 и −5 дБ и прогоняем весь пайплайн. Отдельно два крайних случая:
в аудио **только** шум (белый или чужая речь) и вопроса нет совсем. Колонка «по делу» показывает,
есть ли в ответе ключевые слова правильного ответа."""),
code(r'''rows = []
cases = [(snr, f"кафе {snr} дБ") for snr in ([-5] if SMOKE else [0, -5])]
for p in (PAIRS[:2] if SMOKE else PAIRS):
    for snr, name in cases:
        noisy = mix_at_snr(speech[p["id"]], NOISES["кафе"], snr)
        path = ANSWERS / f"noisy_in_{p['id']}_{snr}.wav"; sf.write(str(path), noisy, SR)
        ans, _, tr = run_omni_assistant(DATA / p["image"], path, return_trace=True,
                                        out_path=ANSWERS / f"noisy_out_{p['id']}_{snr}.wav")
        rows.append({"пара": p["id"], "условие": name, "Whisper": tr["transcript"],
                     "WER": round(wer(p["question"], tr["transcript"]), 2), "надёжно": tr["asr_reliable"],
                     "ответ": ans, "по делу": any(k in ans.lower() for k in p["keywords"])})
for noise_name in NOISES:                     # вопроса нет, только шум
    path = ANSWERS / f"only_noise_{noise_name}.wav"
    sf.write(str(path), (NOISES[noise_name][: SR * 4] / np.abs(NOISES[noise_name][: SR * 4]).max() * .5).astype(np.float32), SR)
    for guard in (False, True):
        ans, _, tr = run_omni_assistant(DATA / "images/landmark.jpg", path, use_guard=guard, return_trace=True)
        rows.append({"пара": "landmark", "условие": f"только шум «{noise_name}», защита {'вкл' if guard else 'выкл'}",
                     "Whisper": tr["transcript"], "WER": None, "надёжно": tr["asr_reliable"], "ответ": ans,
                     "по делу": None})
deaf = pd.DataFrame(rows)
deaf.to_csv(OUT / "noise_pipeline.csv", index=False)
deaf'''),

md("""## 9. Анализ «слепоты»: насколько точно Qwen читает мелкие детали

На синтетической этикетке 9 строк, шрифт от 64 до 7 px, эталон известен. Просим модель
переписать весь текст дословно (отдельный системный промпт, без ограничения на 3 предложения) и
меняем **бюджет визуальных токенов**: картинка масштабируется так, чтобы занять ≈N токенов по
32×32 px. Исходная этикетка 900×560 ≈ 490 токенов, бюджеты 1024 и 2048 её увеличивают.

Точность строки = 1 − CER по лучшему совпадению среди строк ответа."""),
code(r'''OCR_SYSTEM = "Ты — система распознавания текста. Переписывай текст с изображения дословно, ничего не добавляя."
OCR_PROMPT = "Перепиши весь текст на этикетке дословно, строка за строкой."
LABEL = DATA / "images/label.png"
LABEL_LINES = META["label_lines"]

def norm_ocr(s):
    s = s.lower().replace("ё", "е").replace("«", '"').replace("»", '"')
    return " ".join(re.sub(r"[*#`|]", " ", s).split())

def line_accuracy(truth, output):
    cands = [norm_ocr(l) for l in output.splitlines() if l.strip()] or [""]
    t = norm_ocr(truth)
    return max(max(0.0, 1 - jiwer.cer(t, c)) if c else 0.0 for c in cands)

BUDGETS = [256, 1024] if SMOKE else [128, 256, 512, 1024, 2048]
ocr_rows, ocr_raw = [], {}
for tokens in BUDGETS:
    px = tokens * 32 * 32
    t0 = time.perf_counter()
    out, st = ask_vlm(LABEL, OCR_PROMPT, system=OCR_SYSTEM, max_new_tokens=400, max_pixels=px, min_pixels=px)
    sec = time.perf_counter() - t0
    ocr_raw[tokens] = out
    for line in LABEL_LINES:
        ocr_rows.append({"бюджет токенов": tokens, "размер картинки": f"{st['image_size'][0]}×{st['image_size'][1]}",
                         "шрифт, px": line["font_px"], "строка": line["text"],
                         "точность": line_accuracy(line["text"], out), "сек": sec})
ocr = pd.DataFrame(ocr_rows)
ocr.to_csv(OUT / "ocr_detail.csv", index=False)
ocr_table = ocr.pivot_table(index=["шрифт, px", "строка"], columns="бюджет токенов", values="точность").sort_index(ascending=False)
display((ocr_table * 100).round(0).astype(int).astype(str) + "%")
display(ocr.groupby("бюджет токенов")[["размер картинки", "сек"]].first())'''),
code(r'''fig, ax = plt.subplots(figsize=(8, 4.2))
im = ax.imshow(ocr_table.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ocr_table.columns)), [str(c) for c in ocr_table.columns])
ax.set_yticks(range(len(ocr_table)), [f"{px} px" for px, _ in ocr_table.index])
for i in range(ocr_table.shape[0]):
    for j in range(ocr_table.shape[1]):
        ax.text(j, i, f"{ocr_table.values[i, j]:.0%}", ha="center", va="center", fontsize=8)
ax.set_xlabel("бюджет визуальных токенов"); ax.set_ylabel("размер шрифта")
ax.set_title("Точность чтения строк этикетки"); plt.colorbar(im, ax=ax); plt.tight_layout()
plt.savefig(OUT / "ocr_detail.png", dpi=120); plt.show()

for tokens in (BUDGETS[0], BUDGETS[-1]):
    display(Markdown(f"**Ответ при {tokens} токенах:**\n```\n{ocr_raw[tokens]}\n```"))'''),
code(r'''# Мелкие цифры на реальном фото: жёлтые ценники на рынке
for tokens in (256, 2048) if not SMOKE else (256,):
    px = tokens * 32 * 32
    out, st = ask_vlm(DATA / "images/market.jpg", "Перечисли все цены на жёлтых ценниках и к каким товарам они относятся.",
                      system=OCR_SYSTEM, max_new_tokens=300, max_pixels=px, min_pixels=px)
    display(Markdown(f"**{tokens} токенов ({st['image_size'][0]}×{st['image_size'][1]}):**\n\n{out}"))'''),

md("""## 10. Итоги"""),
code(r'''label_pair = next(r for r in demo_rows if r["пара"] == "label")
default_budget = min(BUDGETS, key=lambda b: abs(b - IMAGE_MAX_PIXELS / 1024))
readable = ocr[(ocr["бюджет токенов"] == default_budget) & (ocr["точность"] >= 0.9)]["шрифт, px"]
cafe = wer_table.loc[("кафе", True)]
summary = {
    "vlm": VLM_ID, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE,
    "mean_total_s": round(float(mean.total_s), 2), "mean_asr_s": round(float(mean.asr_s), 2),
    "mean_vlm_s": round(float(mean.vlm_s), 2), "mean_tts_s": round(float(mean.tts_s), 2),
    "slowest_stage": share.idxmax(), "vlm_tokens_per_s": round(float(mean.vlm_tok_per_s), 1),
    "cold_start_s": round(COLD_START_S, 1),
    "stream_ttfa_s": round(float(stream.loc["среднее", "первый звук (стриминг), с"]), 2),
    "clean_wer": round(float(asr_clean["WER"].mean()), 3),
    "cafe_wer_by_snr": {int(k): round(float(v), 2) for k, v in cafe.items()},
    "answers_on_topic": f"{sum(r['по делу'] for r in demo_rows)}/{len(demo_rows)}",
    "ocr_default_budget": default_budget,
    "ocr_min_font_px_90pct": int(readable.min()) if len(readable) else None,
    "label_answer": label_pair["answer"],
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
summary'''),
md("""### Выводы

**1. Латентность.** Таблица и график §7: самый долгий этап — генерация Qwen3-VL.
Whisper small на T4 расшифровывает 3–5-секундный вопрос за доли секунды, Silero синтезирует
2–3 предложения быстрее реального времени. 8B-модель в NF4 декодирует медленно: на каждом токене
веса распаковываются из 4 бит, а T4 — старая карта. Время ответа растёт линейно с его длиной, поэтому
ограничение «2–3 предложения» в промпте полезно не только для удобства слушателя, но и для скорости.

**2. «Глухота»** (§8). До SNR ≈ 5–10 дБ Whisper почти не ошибается. Ниже ломается, и сильнее всего
на шуме «кафе»: посторонняя речь для модели такая же речь, её нельзя отфильтровать по спектру,
и Whisper начинает расшифровывать чужие фразы. Спектральное шумоподавление помогает на стационарном
белом шуме и почти бесполезно (иногда вредно) на чужой речи. Может ли ассистент «догадаться» по
картинке: частично. Если в расшифровке осталось хотя бы 1–2 ключевых слова («где», «порода»), Qwen
восстанавливает вопрос по контексту изображения. Если ASR выдал связную, но чужую фразу (из шума
кафе) или галлюцинацию на шуме, модель честно отвечает на неправильный вопрос. Поэтому нужна
проверка уверенности ASR (`no_speech_prob`, `avg_logprob`, словарь галлюцинаций): с ней ассистент
описывает картинку и просит повторить, вместо уверенной ерунды.

**3. «Слепота»** (§9). Крупный и средний текст Qwen3-VL читает почти без ошибок, качество падает
вместе с числом пикселей на символ. Решает не размер шрифта сам по себе, а бюджет визуальных
токенов: при увеличении картинки строки 8–10 px снова читаются, но растёт prefill и с ним время.
Самые опасные ошибки — правдоподобные: модель может «дочитать» неразборчивую цифру по смыслу
(срок годности, номер ТУ). Для этикеток и чеков нужен отдельный режим «чтения» с высоким
разрешением или кроп нужной области.

**4. Готова ли архитектура к реальному использованию?** Как прототип да: всё работает на одной T4,
отвечает по делу, по-русски и голосом. Для продукта нет: несколько секунд тишины после вопроса —
это много для голосового интерфейса (у живого собеседника пауза 0,3–0,5 с), и нет защиты от шума
и чужой речи. Что менять, по убыванию эффекта:

* **стриминг на всех этапах**: Qwen отдаёт токены через `TextIteratorStreamer`, Silero озвучивает
  по предложению. Уже в этом ноутбуке это сокращает время до первого звука (§7, таблица стриминга);
  следующий шаг — потоковый ASR с VAD (Silero VAD, распознавание по ходу речи), чтобы ASR
  заканчивался вместе с фразой пользователя;
* **сервинг вместо `generate`**: vLLM / SGLang / TensorRT-LLM с AWQ или FP8 вместо bitsandbytes
  (bnb экономит память, но медленный на декодинге), continuous batching, KV-cache, прогретые модели;
* **легче модели**: Qwen3-VL-4B или 2B для простых вопросов (маршрутизация по сложности),
  faster-whisper / distil-whisper, а для русского — GigaAM;
* **меньше визуальных токенов** по умолчанию и кроп/высокое разрешение только когда нужно читать текст;
* **end-to-end omni-модель** (Qwen3-Omni и аналоги): принимает аудио напрямую и генерирует речь,
  убирает два стыка «аудио → текст → аудио» и теряет меньше информации (интонация, шум);
* **железо**: L4/A10/A100 вместо T4 даёт bf16 и в разы более быстрый декодинг."""),
code(r'''if IN_COLAB:
    import shutil
    from google.colab import files
    shutil.make_archive("/content/llm8_outputs", "zip", OUT)
    files.download("/content/llm8_outputs.zip")'''),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
out = ROOT / "notebooks" / "omni_assistant.ipynb"
nbf.write(nb, out)
print("записан", out.relative_to(ROOT), "—", len(cells), "ячеек")
