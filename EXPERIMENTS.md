# Експерименти

Този файл обобщава експериментите и вариантите, които използвам за проекта.
Целта е да е ясно какво е реално изпълнено и какви посоки за сравнение са
подходящи при разширяване на модела.

## 1. Dataset split

Разделям данните по уникални изображения, а не по отделни captions. Това е
важно, защото всяко изображение има няколко captions. Ако captions за едно и
също изображение попаднат едновременно в train и test, оценката ще стане
изкуствено по-лесна.

Използваният split е:

- train: 6473 изображения;
- validation: 809 изображения;
- test: 809 изображения.

## 2. Preprocessing и augmentation

### Baseline preprocessing

За validation и test използвам детерминиран preprocessing:

- resize до `224x224`;
- преобразуване до tensor;
- ImageNet normalization.

Това е подходящо за VGG16, защото предварително обученият модел очаква image
statistics, близки до ImageNet.

### Train augmentation

За train split-а добавям:

- random horizontal flip;
- random rotation.

Идеята е моделът да не запомня изображенията прекалено буквално и да стане
по-устойчив към малки визуални промени.

## 3. Word embedding варианти

### Реализиран вариант: trainable Embedding

Основният модел използва `nn.Embedding`, който се обучава заедно с decoder-а.
Това е най-директният вариант за sequence generation, защото всяка дума от
vocabulary-то получава собствен learnable vector.

### TF-IDF baseline

TF-IDF е полезен за текстов анализ и retrieval baseline, но не е най-естественият
избор за LSTM decoder, защото caption generation изисква последователно
предсказване на следващата дума. В проекта го разглеждам като подходящ baseline
за анализ на captions, не като основен decoder embedding.

### Word2Vec / GloVe / FastText

Тези методи могат да се използват за инициализация на embedding слоя. Предимството
им е, че думите започват от предварително научено езиково пространство. Това може
да помогне при по-малък dataset като Flickr8k.

### BERT / RoBERTa / DistilBERT

Transformer embeddings са по-силни, но променят значително архитектурата. Те са
по-подходящи за по-сложен encoder-decoder или transformer captioning модел,
отколкото за прост LSTM baseline.

## 4. Архитектурни варианти

### Реализиран вариант: VGG16 + LSTM

Използвам предварително обучен VGG16 като image encoder и LSTM като language
decoder. VGG16 backbone-ът е замразен, а projection слой и decoder-ът се
обучават.

### Вариант: различен LSTM hidden size

По-малък hidden size намалява броя параметри и риска от overfitting. По-голям
hidden size дава повече капацитет, но може да изисква повече обучение.

### Вариант: повече LSTM layers и dropout

Два LSTM слоя могат да моделират по-сложни зависимости в текста, но dropout е
важен, защото Flickr8k е сравнително малък dataset.

### Вариант: частично размразен VGG16

Ако се размразят последните convolutional блокове на VGG16, encoder-ът може да
се адаптира по-добре към Flickr8k. Недостатъкът е по-бавно обучение и по-висок
риск от overfitting.

## 5. Основен резултат

Основният обучен модел е:

- encoder: frozen VGG16;
- decoder: LSTM;
- word representation: trainable `Embedding`;
- training: resume training до epoch 10, с най-добър checkpoint на epoch 5;
- selection: най-добър validation loss.

Резултати върху test split:

| Метрика | Стойност |
|---|---:|
| validation loss | 2.7857 |
| test loss | 2.7643 |
| BLEU-1 | 0.5386 |
| BLEU-4 | 0.1392 |
| ROUGE-L | 0.3918 |

## 6. Конфигурации за сравнение

В `projects/image_captioning/experiments.py` държа
експерименталните конфигурации в структуриран вид. Това позволява вариантите да
се показват в notebook или доклад без да се пишат на ръка всеки път.

| Конфигурация | Preprocessing | Embedding | Encoder | Decoder |
|---|---|---|---|---|
| `baseline_vgg16_lstm` | resize + ImageNet normalization | trainable embedding | frozen VGG16 | one-layer LSTM |
| `augmented_vgg16_lstm` | resize + ImageNet normalization + augmentation | trainable embedding | frozen VGG16 | one-layer LSTM |
| `larger_lstm_decoder` | resize + ImageNet normalization + augmentation | trainable embedding | frozen VGG16 | larger hidden size LSTM |
| `dropout_two_layer_lstm` | resize + ImageNet normalization + augmentation | trainable embedding | frozen VGG16 | two-layer LSTM with dropout |
| `pretrained_word_vectors` | resize + ImageNet normalization + augmentation | Word2Vec/GloVe/FastText initialization | frozen VGG16 | one-layer LSTM |

## 7. Реални architecture experiments

Изпълних кратки architecture experiments върху кеширани VGG features. Целта не е
да се получи нов финален модел, а да се сравнят варианти при еднакъв training
budget.

| Вариант | Trainable parameters | Validation loss | Sample BLEU-1 | Sample BLEU-4 | Sample ROUGE-L |
|---|---:|---:|---:|---:|---:|
| `small_lstm_256` | 8,302,414 | 3.1199 | 0.4252 | 0.0921 | 0.3538 |
| `baseline_lstm_512` | 10,028,366 | 2.9511 | 0.4654 | 0.1042 | 0.3543 |
| `two_layer_lstm_dropout` | 12,129,614 | 3.0947 | 0.3942 | 0.0892 | 0.3403 |

Извод: при кратко обучение най-добър е `baseline_lstm_512`. По-малкият LSTM
губи капацитет, а двуслойният LSTM с dropout има повече параметри и вероятно
има нужда от повече epochs, за да стане конкурентен.

## 8. TF-IDF word experiment

TF-IDF експериментът показва кои думи са най-характерни за train captions след
премахване на stop words.

Най-силни думи:

```text
dog, man, two, black, white, boy, woman, girl, wearing, people, water, red
```

Това потвърждава, че Flickr8k е доминиран от captions за хора, кучета, цветове,
движение и outdoor сцени. Този анализ е полезен и за обяснение защо моделът
често генерира общи captions от типа “man/woman/dog”.

## 9. VGG retrieval baseline

Тъй като LSTM decoder-ът понякога генерира граматически правилни, но визуално
грешни captions, използвам и VGG nearest-neighbor retrieval вариант.

Този режим:

- използва кешираните VGG features;
- сравнява входното изображение с train изображенията;
- намира най-близките визуални примери;
- връща caption от най-близкото training изображение.

Това не е толкова гъвкаво като generative модел, но по-рядко дава напълно
несвързани описания. Например при dog сцени retrieval режимът търси визуално подобни
dog изображения, вместо LSTM decoder-ът да извади често срещано изречение като
`a woman is sitting on a bench`.

## 10. Qualitative comparison

В `reports/model/qualitative_comparison.csv` сравнявам:

- reference captions;
- VGG16 + LSTM caption;
- VGG retrieval caption;
- най-близкото retrieval изображение.

Този отчет е важен, защото автоматичните метрики не винаги показват дали
caption-ът е визуално смислен. Qualitative проверката показва кога LSTM decoder
дава несвързан caption и кога retrieval вариантът е по-заземен.

## 11. Training reliability подобрения

След ръчна проверка с реални изображения оформих няколко неща, които
правят обучението и анализа по-надеждни:

- early stopping по validation loss;
- `ReduceLROnPlateau` scheduler;
- централизирани ImageNet normalization стойности;
- `describe_image_preprocessing`, за да се вижда дали train/eval preprocessing
  са консистентни;
- diagnostic script за dataset, checkpoint и preprocessing metadata.

Тези промени не правят LSTM baseline модела автоматично “умен”, но помагат да се
разграничи проблем в setup-а от реално ограничение на dataset-а/архитектурата.

## 12. Извод

VGG16 + LSTM е добър baseline за проекта, защото е разбираем, работещ и
сравнително лек за обучение. Най-логичното следващо подобрение е attention
decoder или beam search при генериране на caption.

## 13. Ограничения при произволни изображения

При снимки извън стила на Flickr8k моделът често генерира неточни или твърде
общи captions. Това е очаквано, защото моделът е обучен върху малък dataset и
няма допълнително знание за всички възможни обекти и сцени.

За inference използвам beam search като алтернатива на greedy decoding. Beam
search не променя самия модел, но подобрява начина, по който се избира крайното
изречение. Ако трябва значително по-добро качество върху произволни снимки,
следващата стъпка е attention decoder или pretrained vision-language модел.

В Streamlit включих и `Pretrained BLIP` режим за сравнение. Той е отделен от
основния VGG16 + LSTM модел и показва как изглеждат captions от по-силен
предварително обучен модел върху произволни изображения.
