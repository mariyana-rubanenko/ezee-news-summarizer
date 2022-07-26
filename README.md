# ezee-news-summarizer

![Build Status](https://github.com/dmmiller612/bert-extractive-summarizer/actions/workflows/test.yml/badge.svg)

Библиотека для извлечения основного смысла (суммаризации) новостного текста.
Берутся эмбеддинги предложений Bert/SBert моделей, на них запускается алгоритм кластеризации, 
который находит центроиды кластеров и ищет близкие к центроидам предлложения.
Для разрешения кореференций используется neuralcoref библиотека.


## Установка библиотеки

```bash
pip install ezee-news-summarizer
```

## Примеры использования библиотеки

Простой пример.

```python
from summarizer import Summarizer

body = 'Здесь Ваш текст1'
body2 = 'Здесь Ваш текст2'
model = Summarizer()
model(body)
model(body2)
```

Пример с количеством заданных предложений суммаризации.

```python
from summarizer import Summarizer
body = 'Здесь Ваш текст'
model = Summarizer()
result = model(body, ratio=0.2)  # Specified with ratio
result = model(body, num_sentences=3)  # Will return 3 sentences 
```

Использование множества последних скрытых слоев:

```python
from summarizer import Summarizer
body = 'Здесь Ваш текст'
model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
result = model(body, num_sentences=3)
```

Пример с использованием SBert.

```
pip install -U sentence-transformers
```

```python
from summarizer.sbert import SBertSummarizer

body = 'Здесь Ваш текст'
model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
result = model(body, num_sentences=3)
```

Поиск эмбеддингов суммаризации:

```python
from summarizer import Summarizer
body = 'Text body that you want to summarize with BERT'
model = Summarizer()
result = model.run_embeddings(body, ratio=0.2)  # Specified with ratio. 
result = model.run_embeddings(body, num_sentences=3)  # Will return (3, N) embedding numpy matrix.
result = model.run_embeddings(body, num_sentences=3, aggregate='mean')  # Will return Mean aggregate over embeddings. 
```

Для кореференций сначала инсталлируем необходимые пакеты:

```bash
pip install spacy
pip install transformers # > 4.0.0
pip install neuralcoref

python -m spacy download en_core_web_md
```

Для использования кореференций:

```python
from summarizer import Summarizer
from summarizer.text_processors.coreference_handler import CoreferenceHandler

handler = CoreferenceHandler(greedyness=.4)
# Как работает кореференция:
# >>>handler.process('''My sister has a dog. She loves him.''', min_length=2)
# ['My sister has a dog.', 'My sister loves a dog.']

body = 'Здесь Ваш текст1'
body2 = 'Здесь Ваш текст2'
model = Summarizer(sentence_handler=handler)
model(body)
model(body2)
```
Кастомная модель:

```python
from transformers import *

# Load model, model config and tokenizer via Transformers
custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)

from summarizer import Summarizer

body = 'Здесь Ваш текст1'
body2 = 'Здесь Ваш текст2'
model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
model(body)
model(body2)
```

Пример большого текста:

```python
from summarizer import Summarizer

body = 'Здесь Ваш огромный по размеру текст'

model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
```

Вычисление elbow кластеризации.

```python
from summarizer import Summarizer

body = 'Здесь Ваш текст'
model = Summarizer()
res = model.calculate_elbow(body, k_max=10)
print(res)
```

Нахождение оптимального числа предложений для суммаризации, используя пороговое значение количества кластеров:

```python
from summarizer import Summarizer

body = 'Здесь Ваш текст'
model = Summarizer()
res = model.calculate_optimal_k(body, k_max=10)
print(res)
```

## Описание параметров

```
model = Summarizer(
    model: Модель трансформера с hugginface.
    custom_model: Кастомная предобученная модель.
    custom_tokenizer: Кастомный токенизатор.
    hidden: Отрицательное число, характеризующее сколько последних скрытых слоев требуется учитывать.
    reduce_option: Стратегия пулинга.
    sentence_handler: Хендлер предложений.
)

model(
    body: str # Исходный текст.
    ratio: float # Отношение числа предложений суммаризационного текста к числу предложений исходного текста.
    min_length: int # Минимальная длина предложения.
    max_length: int # Максимальная длина предложения.
    num_sentences: Количество предложений суммаризации.
)
```

## Запуск сервиса

Чтобы запустить flask сервис библиотеки ezee-news-summarizer, требуется выполнить следующие команды:

```
make docker-service-build
make docker-service-run
```

Ниже будет использоваться bert-large-uncased модель:

```
docker build -t summary-service -f Dockerfile.service ./
docker run --rm -it -p 5000:5000 summary-service:latest -model bert-large-uncased
```

Список возможных аргументов.

* -greediness: Флоат параметр для neurocolef библиотеки.
* -reduce: Пулинг эмбеддинг слоя (mean, median, max).
* -hidden: Определяет скрытый слой для эмбеддинга. (дефолтное значение=-2)
* -port: Используемый порт.
* -host: Используемый хост.

Эндпоинт запущенного сервиса `http://localhost:5000/summarize`. 
Этот эндпоинт принимает text/plain входные значения исходного текста для суммаризации. Также
можно передать следующие параметры в эндпоинт:

* ratio: Отношение числа предложений суммаризационного текста к числу предложений исходного текста. (дефолтное значение=0.2)
* min_length: Минимальная длина предложения. (дефолтное значение=25)
* max_length: Максимальная длина предложения. (дефолтное значение=500)

Пример запроса:

```
POST http://localhost:5000/summarize?ratio=0.1

Content-type: text/plain

Body:
Здесь Ваш текст
