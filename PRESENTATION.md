# Презентация: Генериране на описание за изображение

## Слайд 1: Тема и цел

- Тема: генериране на описание за изображение.
- Цел: уеб приложение, което приема снимка и връща кратък caption.
- Dataset: Flickr8k.

## Слайд 2: Какво представлява задачата

- Image captioning комбинира computer vision и NLP.
- Моделът трябва да разпознае визуалното съдържание и да го превърне в текст.
- Изходът е последователност от думи, не единичен class label.

## Слайд 3: Разгледани техники

- CNN + RNN/LSTM encoder-decoder.
- Attention-based image captioning.
- Transformer-based vision-language модели.
- BLEU и ROUGE-L за автоматична оценка.

## Слайд 4: Данни

- Dataset: Flickr8k.
- 8091 изображения.
- 40455 captions.
- Средно около 5 captions на изображение.
- Проверени са липсващи изображения, празни captions и дубликати.

## Слайд 5: EDA

- Анализирах дължина на captions.
- Проверих най-чести думи.
- Проверих размери и aspect ratio на изображенията.
- Използвах визуализации за текстовите и image статистиките.

## Слайд 6: Preprocessing

- Resize до `224x224`.
- ImageNet normalization.
- Train augmentation: random horizontal flip и random rotation.
- Validation/test preprocessing е детерминиран.

## Слайд 7: Модел

- Encoder: предварително обучен VGG16.
- Decoder: LSTM.
- Word representation: обучаем `Embedding` слой.
- Кеширах VGG features, за да ускоря обучението.
- Използвах и VGG retrieval вариант за по-заземени captions.

## Слайд 8: Обучение и оценка

- Split: train / validation / test по уникални изображения.
- Loss: cross-entropy върху следващата дума.
- Метрики: BLEU-1, BLEU-4, ROUGE-L.
- Най-добрият checkpoint е избран по validation loss.

## Слайд 9: Резултати

- Validation loss: `2.7857`.
- Test loss: `2.7643`.
- BLEU-1: `0.5386`.
- BLEU-4: `0.1392`.
- ROUGE-L: `0.3918`.
- Генерираните captions са запазени в `reports/model/generated_captions.csv`.
- Architecture experiments показаха, че hidden size 512 е най-добър от тестваните кратки варианти.

## Слайд 10: Streamlit приложение

- Потребителят качва изображение.
- Моделът генерира caption.
- Има три режима: VGG retrieval, VGG16 + LSTM и pretrained BLIP.
- Project baseline зарежда checkpoint от `checkpoints/best_model.pt`.
- Използвам greedy и beam search режим за VGG16 + LSTM модела.

## Слайд 11: Тестове

- Написани са тестове за vocabulary, dataset, model, training и evaluation.
- Има тестове за retrieval, TF-IDF, presentation builder и pretrained captioning wrapper.
- Последна проверка: `31 passed`.

## Слайд 12: Ограничения

- Flickr8k е малък dataset и не покрива всички ежедневни сцени.
- VGG16 + LSTM е baseline модел без attention.
- При произволни снимки captions могат да бъдат generic или неточни.
- Beam search подобрява decoding-а, но не решава напълно ограниченията на модела.
- VGG retrieval намалява несвързаните captions чрез визуално подобни train примери.
- Pretrained BLIP режимът служи като сравнение с по-силен модел.

## Слайд 13: Изводи и бъдещи подобрения

- VGG16 + LSTM дава работещ baseline.
- Attention или transformer модел вероятно би подобрил качеството.
- Beam search може да подобри inference-а.
- Могат да се добавят повече architecture и embedding експерименти.
