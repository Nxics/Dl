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

Диагностика на dataset/checkpoint/preprocessing:

```console
python scripts/diagnose_captioning_setup.py
```

Пускане на приложението:

```console
python -m streamlit run streamlit_app/app.py
```

Приложението има три режима:

- `Project VGG retrieval` — използва VGG features и намира визуално най-близки
  Flickr8k train изображения; използвам го като по-стабилен проектен режим при
  снимки, за които LSTM decoder-ът дава твърде общи описания;
- `Project baseline VGG16 + LSTM` — използва обучения в проекта модел от
  `checkpoints/best_model.pt`;
- `Pretrained BLIP` — използва предварително обучен BLIP модел за сравнение с
  по-съвременен vision-language подход.

Project baseline режимът очаква обучен модел в `checkpoints/best_model.pt`.
Този файл не се създава от настройването на Python средата. Трябва да бъде
получен от training скрипт или да бъде зададен друг валиден checkpoint път в
интерфейса. Ако има само `checkpoints/smoke_test_model.pt`, приложението го
избира автоматично и показва предупреждение, че резултатите са само техническа
проба.

Генериране на PowerPoint презентация:

```console
python scripts/build_presentation.py
```

Командата използва `PRESENTATION.md` и записва `.pptx` файл в
`presentation/`.

## Структура на проекта

- `projects/image_captioning/vocab.py` — tokenization и vocabulary;
- `projects/image_captioning/data.py` — четене на captions и PyTorch datasets;
- `projects/image_captioning/transforms.py` — preprocessing на изображенията;
- `projects/image_captioning/model.py` — VGG16 encoder и LSTM decoder;
- `projects/image_captioning/features.py` — кеширане на VGG16 features;
- `projects/image_captioning/training.py` — training и validation логика;
- `projects/image_captioning/evaluation.py` — BLEU и ROUGE-L оценяване;
- `projects/image_captioning/inference.py` — генериране на captions;
- `projects/image_captioning/retrieval.py` — VGG nearest-neighbor retrieval вариант;
- `projects/image_captioning/pretrained_captioning.py` — BLIP captioning вариант за сравнение;
- `projects/image_captioning/experiments.py` — дефиниции на експериментални конфигурации;
- `projects/image_captioning/flickr8k_eda_step_by_step.ipynb` — EDA анализ;
- `projects/image_captioning/flickr8k_modeling_step_by_step.ipynb` — подготовка, обучение и оценяване;
- `streamlit_app/app.py` — интерфейс за качване на изображение;
- `scripts/diagnose_captioning_setup.py` — проверка на dataset, checkpoint и preprocessing;
- `tests/` — автоматични тестове;
- `EXPERIMENTS.md` — обобщение на preprocessing, embedding и architecture експериментите;
- `PRESENTATION.md` — структура на презентацията.

## 1. Разучаване на техники за image captioning

Image captioning комбинира две области: computer vision и natural language
processing. Идеята е изображението първо да се представи като числов вектор, а
след това language модел да генерира последователност от думи.

Разгледах няколко основни подхода:

- encoder-decoder архитектура — CNN извлича image features, а RNN/LSTM
  генерира текста дума по дума;
- attention механизъм — decoder-ът не използва само един общ image vector, а
  може да “гледа” различни части от изображението при генериране на всяка дума;
- transformer-based модели — по-съвременен подход, при който attention е
  основният механизъм както за изображението, така и за текста;
- retrieval/baseline подходи — генериране или избиране на caption чрез прилика
  с вече съществуващи captions.

За този проект избрах по-класическия CNN + LSTM вариант, защото е достатъчно
ясен за проследяване и позволява да се видят всички основни етапи: preprocessing,
feature extraction, vocabulary, sequence modeling, evaluation и Streamlit
интерфейс.

Основни източници:

- Vinyals et al., “Show and Tell: A Neural Image Caption Generator”
  (`https://arxiv.org/abs/1411.4555`);
- Xu et al., “Show, Attend and Tell: Neural Image Caption Generation with
  Visual Attention” (`https://arxiv.org/abs/1502.03044`);
- Stefanini et al., “From Show to Tell: A Survey on Deep Learning-based Image
  Captioning” (`https://arxiv.org/abs/2107.06912`);
- Papineni et al., “BLEU: a method for automatic evaluation of machine
  translation”;
- Lin, “ROUGE: A Package for Automatic Evaluation of Summaries”.

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

Първо обучих модела за 4 епохи, след което продължих training-а с по-нисък
learning rate, scheduler и early stopping. Най-добрият checkpoint след resume
training е от epoch 5:

```text
validation loss: 2.7857
test loss:       2.7643
BLEU-1:          0.5386
BLEU-4:          0.1392
ROUGE-L:         0.3918
```

Артефакти:

- `checkpoints/best_model.pt` — най-добрият обучен модел;
- `DATA/flickr8k/features/vgg16_pool7/` — кеширани features за 8091 изображения;
- `reports/model/generated_captions.csv` — генерирани test captions;
- `reports/model/evaluation_metrics.csv` — test метрики.

## Ограничения на модела

Моделът е работещ baseline, но не е универсален image understanding модел.
Flickr8k е сравнително малък dataset и съдържа ограничен тип сцени — често хора,
кучета, деца, спорт и outdoor ситуации. Затова при произволни лични снимки
описанието може да бъде неточно или твърде generic.

Основните ограничения са:

- dataset-ът е малък за image captioning задача;
- VGG16 + LSTM е по-класическа архитектура и няма attention;
- decoder-ът често използва най-често срещани Flickr8k фрази;
- моделът не е fine-tune-нат върху по-голям dataset като MS COCO;
- greedy decoding може да избира локално най-вероятната дума, без да намира
  по-добро цялостно изречение.

За генериране на captions използвам и beam search в Streamlit приложението. Beam
search държи няколко възможни caption кандидата едновременно и избира
най-добрия според общия log-probability score. Това може да направи описанията
по-стабилни, но не премахва ограниченията на dataset-а и архитектурата.

За сравнение използвам и `Pretrained BLIP` режим. Той не е основният модел,
който обучавам в проекта, а служи като ориентир как се държи по-силен
предварително обучен vision-language модел върху произволни изображения.

След ръчна проверка на генерираните captions използвам и `Project VGG retrieval`
режим. Той използва
същите VGG features и Flickr8k train captions, но вместо да генерира изречение
с LSTM, намира визуално най-близко training изображение и връща негов caption.
Този режим по-рядко дава напълно несвързано описание, защото
caption-ът е директно вързан към визуално подобен пример.

## Покритие на задължителните стъпки

| Стъпка | Статус | Как е покрита |
|---|---:|---|
| 1. Разучаване на научни статии и техники | покрито | Направих кратък обзор на encoder-decoder, attention и transformer подходи. |
| 2. Разглеждане на данните | покрито | EDA notebook-ът анализира captions, images, липсващи стойности, дубликати, дължини и визуализации. |
| 3. Preprocessing и augmentation | частично покрито | Използвам resize, normalization, random horizontal flip и random rotation. Описани са и възможни следващи експерименти. |
| 4. Word embeddings | покрито | Основният модел използва trainable `Embedding` слой. Направих TF-IDF caption experiment и описах Word2Vec/GloVe/FastText/BERT-like варианти. |
| 5. VGG и LSTM варианти | покрито | Реализирана е VGG16 + LSTM архитектура и са изпълнени architecture experiments за hidden size, брой LSTM layers и dropout. |
| 6. Streamlit и тестове | покрито | Има Streamlit приложение и автоматични тестове. Последната проверка е `31 passed`. |
| 7. Презентация | подготвено | Добавен е файл `PRESENTATION.md` със структура на слайдовете. |

## Експерименти и варианти

Основният обучен експеримент е:

- visual encoder: предварително обучен VGG16;
- language decoder: LSTM;
- word representation: обучаем `Embedding` слой;
- image preprocessing: resize до `224x224`, ImageNet normalization;
- train augmentation: horizontal flip и малка rotation;
- split: train / validation / test по уникални изображения;
- evaluation: BLEU-1, BLEU-4, ROUGE-L и test loss.

Конфигурациите са описани и в `projects/image_captioning/experiments.py`, за
да могат да се използват повторно в notebook, доклад или бъдещо обучение.

Реални architecture experiments върху кеширани VGG features:

| Вариант | Hidden size | LSTM layers | Dropout | Validation loss | Sample BLEU-4 |
|---|---:|---:|---:|---:|---:|
| `small_lstm_256` | 256 | 1 | 0.0 | 3.1199 | 0.0921 |
| `baseline_lstm_512` | 512 | 1 | 0.0 | 2.9511 | 0.1042 |
| `two_layer_lstm_dropout` | 512 | 2 | 0.3 | 3.0947 | 0.0892 |

Изводът от този кратък architecture experiment е, че `baseline_lstm_512` е
най-добрият вариант от трите при еднакъв кратък training budget. По-малкият
decoder няма достатъчно капацитет, а двуслойният decoder с dropout изисква
повече обучение, за да покаже предимство.

TF-IDF експериментът върху train captions показва най-характерните думи в
dataset-а след премахване на stop words:

```text
dog, man, two, black, white, boy, woman, girl, wearing, people, water
```

Това подкрепя EDA извода, че Flickr8k е силно ориентиран към хора, кучета,
движение, цветове и outdoor сцени.

След проверката върху грешни captions оформих и по-ясна training
инфраструктура:

- `TrainingConfig` за epochs, learning rate, gradient clipping и scheduler;
- `EarlyStopping` за спиране при липса на подобрение във validation loss;
- `ReduceLROnPlateau` за намаляване на learning rate;
- preprocessing metadata чрез `describe_image_preprocessing`;
- diagnostic script за проверка дали dataset, checkpoint и normalization са
  подредени правилно.

Допълнителни варианти, които са подходящи за сравнение:

| Група | Вариант | Очакван ефект |
|---|---|---|
| Preprocessing | само resize + normalization | по-стабилен baseline, но по-малко разнообразие в train данните |
| Preprocessing | flip + rotation | по-добра устойчивост към малки промени в изображенията |
| Embeddings | trainable `Embedding` | най-прост и директен вариант за end-to-end обучение |
| Embeddings | TF-IDF baseline | полезен за анализ на текста, но не е естествен избор за sequence generation |
| Embeddings | Word2Vec/GloVe/FastText | старт с предварителни езикови знания |
| Embeddings | BERT/RoBERTa/DistilBERT | по-силен езиков encoder, но по-тежка и различна архитектура |
| Architecture | VGG16 frozen | по-бързо обучение и по-малък риск от overfitting |
| Architecture | VGG16 partially trainable | потенциално по-добра адаптация към Flickr8k, но по-бавно обучение |
| Architecture | LSTM hidden size 256/512 | контрол на капацитета на decoder-а |
| Architecture | 1 vs 2 LSTM layers + dropout | по-силен decoder, но с риск от overfitting |
| Retrieval | VGG nearest-neighbor caption | по-малко несвързани описания, защото caption-ът идва от визуално подобен train пример |

## Възможни бъдещи подобрения

1. early stopping и learning-rate scheduler;
2. beam search вместо само greedy decoding;
3. attention decoder;
4. Excel/PDF model report;
5. deployment на Streamlit приложението.
