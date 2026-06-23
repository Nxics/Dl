# Проект: Генериране на описание за изображение

В този проект разработих модел, който генерира кратко текстово описание на
подадено изображение. Използвах Flickr8k, предварително обучен VGG16 encoder
и LSTM decoder.

## Стъпка 0: Flickr8k dataset

Използвам [Flickr8k от Kaggle](https://www.kaggle.com/datasets/nunenuh/flickr8k).
След разархивиране подреждам файловете в следната структура:

```text
DATA/flickr8k/
  Images/
  captions.txt
```

Сваляне и разархивиране от директорията на проекта:

```console
mkdir -p DATA/flickr8k
curl --fail --location \
  --output DATA/flickr8k/flickr8k.zip \
  https://www.kaggle.com/api/v1/datasets/download/nunenuh/flickr8k
unzip DATA/flickr8k/flickr8k.zip -d DATA/flickr8k
mv DATA/flickr8k/images DATA/flickr8k/Images
```

## Настройване на средата

Поддържат се Python 3.12–3.14. От директорията на проекта:

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user \
  --name flickr8k-captioning \
  --display-name "Python (.venv) — Flickr8k Captioning"
```

За Windows PowerShell активацията е:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user `
  --name flickr8k-captioning `
  --display-name "Python (.venv) — Flickr8k Captioning"
```

Във VS Code отварям notebook-а и избирам kernel:
`Python (.venv) — Flickr8k Captioning`.

Проверка на средата:

```console
python -m pytest
```

Пускане на приложението:

```console
python -m streamlit run streamlit_app/app.py
```

Приложението очаква обучен модел в `checkpoints/best_model.pt`. Този файл
не се създава от настройването на Python средата. Трябва да бъде получен от
training скрипт или да бъде зададен друг валиден checkpoint път в интерфейса.
Ако има само `checkpoints/smoke_test_model.pt`, приложението го избира
автоматично и показва предупреждение, че резултатите са само техническа проба.

## Структура на проекта

- `projects/image_captioning/vocab.py` — tokenization и vocabulary;
- `projects/image_captioning/data.py` — четене на captions и PyTorch datasets;
- `projects/image_captioning/transforms.py` — preprocessing на изображенията;
- `projects/image_captioning/model.py` — VGG16 encoder и LSTM decoder;
- `projects/image_captioning/features.py` — кеширане на VGG16 features;
- `projects/image_captioning/training.py` — training и validation логика;
- `projects/image_captioning/evaluation.py` — BLEU и ROUGE-L оценяване;
- `projects/image_captioning/inference.py` — генериране на captions;
- `projects/image_captioning/flickr8k_eda_step_by_step.ipynb` — EDA анализ;
- `projects/image_captioning/flickr8k_modeling_step_by_step.ipynb` — подготовка, обучение и оценяване;
- `streamlit_app/app.py` — интерфейс за качване на изображение;
- `tests/` — автоматични тестове.

## Очаквана структура на Flickr8k

Пример:

```text
DATA/flickr8k/
  Images/
    1000268201_693b08cb0e.jpg
    ...
  captions.txt
```

или:

```text
DATA/flickr8k/
  Images/
  captions.csv
```

Файлът с captions трябва да има колони `image,caption` или legacy формат `image.jpg#0\tcaption`.

## Пускане на EDA

```console
python scripts/run_flickr8k_eda.py \
  --images-dir DATA/flickr8k/Images \
  --captions-path DATA/flickr8k/captions.txt \
  --output-dir reports/eda
```

## Пускане на тестовете за проекта

```console
python -m pytest tests/test_image_captioning_vocab.py tests/test_image_captioning_data.py
```

## Текущ резултат

Обучих модела за 5 епохи с предварително кеширани VGG16 features.
Най-добрият checkpoint е от epoch 4:

```text
validation loss: 2.8200
test loss:       2.7989
BLEU-1:          0.5349
BLEU-4:          0.1279
ROUGE-L:         0.3772
```

Артефакти:

- `checkpoints/best_model.pt` — най-добрият обучен модел;
- `DATA/flickr8k/features/vgg16_pool7/` — кеширани features за 8091 изображения;
- `reports/model/generated_captions.csv` — генерирани test captions;
- `reports/model/evaluation_metrics.csv` — test метрики.

## Възможни бъдещи подобрения

1. early stopping и learning-rate scheduler;
2. beam search вместо само greedy decoding;
3. attention decoder;
4. Excel/PDF model report;
5. deployment на Streamlit приложението.
