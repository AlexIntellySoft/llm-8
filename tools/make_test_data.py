"""Готовит тестовые данные в data/. Запускается один раз на macOS (нужны `say` и `afconvert`),
результат лежит в git, поэтому в Colab этот скрипт запускать не нужно:

    .venv/bin/python -m tools.make_test_data

* data/images/label.png  — синтетическая этикетка йогурта: текст от 64 до 7 px, эталон известен
  построчно, поэтому на ней можно измерить, с какого размера шрифта Qwen перестаёт читать;
* data/audio/q_*.wav     — вопросы голосом (голос Milena, 16 кГц, моно);
* data/audio/babble_*.wav — посторонние фразы, из них в ноутбуке собирается «шум кафе»;
* data/pairs.json        — пары «изображение + аудио», эталонный текст вопроса и ключевые слова
  правильного ответа.

Фотографии (fridge, landmark, dog, market) скачаны с Wikimedia Commons, лицензии в data/images/LICENSES.md.
"""
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
VOICE = "Milena"

# Строки этикетки: (текст, размер шрифта в px, жирный). Эталон для теста «мелких деталей».
LABEL_LINES = [
    ("ЙОГУРТ ПИТЬЕВОЙ", 64, True),
    ("Клубника и банан", 40, False),
    ("Массовая доля жира 2,5%", 28, False),
    ("Белки 2,8 г, жиры 2,5 г, углеводы 12,4 г, в т.ч. сахар 11,0 г", 18, False),
    ("Энергетическая ценность 84 ккал / 352 кДж", 14, False),
    ("Годен до: 14.11.2026", 12, False),
    ("Изготовитель: ООО «Молочный двор», г. Тверь, ул. Заводская, 7", 10, False),
    ("Хранить при температуре от +2 до +6 °C", 8, False),
    ("ТУ 10.51.52-001-12345678-2026, партия 0417", 7, False),
]

PAIRS = [
    {
        "id": "fridge",
        "image": "images/fridge.jpg",
        "audio": "audio/q_fridge.wav",
        "question": "Что мне приготовить на ужин из того, что лежит в холодильнике?",
        "keywords": ["помидор", "томат", "сыр", "апельсин", "морков", "лимон", "виноград"],
    },
    {
        "id": "landmark",
        "image": "images/landmark.jpg",
        "audio": "audio/q_landmark.wav",
        "question": "Где это находится и как называется это здание?",
        "keywords": ["москв", "василия блаженного", "покрова", "красн"],
    },
    {
        "id": "dog",
        "image": "images/dog.jpg",
        "audio": "audio/q_dog.wav",
        "question": "Какой породы эта собака и подойдёт ли она для жизни в квартире?",
        "keywords": ["ретривер"],
    },
    {
        "id": "market",
        "image": "images/market.jpg",
        "audio": "audio/q_market.wav",
        "question": "Что здесь продаётся и в какой валюте указаны цены?",
        "keywords": ["фунт", "£", "стерлинг"],
    },
    {
        "id": "label",
        "image": "images/label.png",
        "audio": "audio/q_label.wav",
        "question": "Какой срок годности у этого йогурта и сколько в нём сахара?",
        "keywords": ["14", "ноябр", "11"],
    },
]

# Посторонняя речь для «шума кафе»: несколько фраз накладываются друг на друга со сдвигом.
BABBLE = [
    "Мне, пожалуйста, большой капучино и круассан с миндалём.",
    "Вчера мы весь вечер обсуждали отпуск, но так и не решили, куда поедем.",
    "Посмотри, автобус опять опаздывает уже на двадцать минут.",
    "Кажется, сегодня обещали дождь, а я забыл зонт дома.",
    "Давай встретимся после работы у входа в торговый центр.",
    "Счёт можно разделить на троих, мы платим картой.",
]


def make_label(path: Path) -> None:
    img = Image.new("RGB", (900, 560), (252, 248, 236))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 12, 887, 547], outline=(200, 40, 60), width=6)
    y = 36
    for text, size, bold in LABEL_LINES:
        font = ImageFont.truetype(FONT_BOLD if bold else FONT, size)
        color = (200, 40, 60) if size >= 40 else (30, 30, 30)
        draw.text((40, y), text, font=font, fill=color)
        y += int(size * 1.45) + 14
    img.save(path)


def say(text: str, path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "x.aiff"
        subprocess.run(["say", "-v", VOICE, "-r", "175", "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)], check=True)


def main() -> None:
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    (DATA / "audio").mkdir(parents=True, exist_ok=True)
    make_label(DATA / "images" / "label.png")
    for pair in PAIRS:
        say(pair["question"], DATA / pair["audio"])
    for i, text in enumerate(BABBLE):
        say(text, DATA / "audio" / f"babble_{i}.wav")
    label_truth = [{"text": t, "font_px": s} for t, s, _ in LABEL_LINES]
    (DATA / "pairs.json").write_text(
        json.dumps({"pairs": PAIRS, "label_lines": label_truth}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(PAIRS)} пар, {len(BABBLE)} фраз для шума, этикетка: {len(LABEL_LINES)} строк")


if __name__ == "__main__":
    main()
