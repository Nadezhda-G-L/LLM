# Разговорный ИИ и большие языковые модели
# Лабораторная работа №2

**Выполнила:** Гусева Надежда Леонидовна  
**Группа:** М4221

## Задание
Обучить 2 диалоговые модели (decoder-only и encoder-decoder) предсказывать ответ системы на запрос пользователя. 

1. Выбрать подходящую модель https://huggingface.co/models.
2. Выбрать подходящий датасет https://huggingface.co/datasets.

Выполнить шаги 3-8 для decoder-only и для encoder-decoder модели.

3. Проверить работу модели в сценарии близком к выбранному датасету в zero-shot режиме.
4. Посчитать метрики bleu и meteor исходной модели на валидационном сабсете выбранного датасета.
5. Дообучить модель на обучающем сабсете выбранного датасета.
6. Проверить работу модели в сценарии близком к выбранному датасету.
7. Посчитать метрики bleu и meteor обученной модели на валидационном сабсете выбранного датасета.
8. Сравнить метрики обученной и исходной моделей.


## 1. Вводные данные

| Параметр | Значение |
| --- | --- |
| **Decoder-only модель** | `facebook/opt-125m` |
| **Encoder-decoder модель** | `google/flan-t5-small` |
| **Датасет** | `HuggingFaceH4/no_robots` |
| **Размер обучающей выборки** | 4 255 пар (input/target) |
| **Размер валидационной выборки** | 670 пар |
| **Размер тестовой выборки** | 284 пары |


## 2. Описание датасета HuggingFaceH4/no_robots

Датасет представляет собой коллекцию диалогов, очищенную от «роботизированных» (robotic) и шаблонных ответов. Он создан для обучения ассистентов, которые звучат естественно и по-человечески.

Структура: Каждый пример содержит messages — список реплик с явными ролями User и Assistant.
Тематика: Широкий спектр задач: от фактических вопросов (география, история) и математических рассуждений до творческих запросов (стихи, генерация идей, бизнес-нейминг) и ролевых игр.
Особенность: В отличие от многих веб-датасетов, здесь ответы ассистента не являются простым извлечением фрагмента из контекста. Они перефразированы, обобщены или содержат личное мнение, что делает задачу более сложной для моделей.

Примеры пар из датасета (до обработки):

Творчество: Input: "Write a haiku about puppies..." $\rightarrow$ Target: "Cuddles and kisses Furry fluff from head to toe..."  
Фактология: Input: "What is Queen Street's alternate name?" $\rightarrow$ Target: "Lot Street is Queen Street's alternate name..."  
Мозговой штурм: Input: "Brainstorm a list of 3 names for a new recycling business..." $\rightarrow$ Target: "1. Better World Recycling. We recycle to make a better world for our future!..."  

## 3. Подготовка данных

В ходе подготовки данных были применены строгие критерии фильтрации для обеспечения чистоты эксперимента:

1. **Разделение на уровне диалогов:** Датасет был разделен на `train` / `validation` / `test` на уровне целых диалогов (8075 / 1425 / 500 диалогов соответственно), и только затем развернут в пары. Это полностью исключает утечку данных (data leakage) между сплитами.

2. **Фильтрация контекста:** Использовалась история из 6 последних реплик с ограничением в 240 слов.

3. **Очистка разметки:** Введены пустые строки между репликами в контексте для decoder-only модели и уникальный маркер `[ASSISTANT]` для обозначения начала генерации.

4. **Дедупликация:** Удалены точные дубликаты пар и пары из валидации/теста, которые уже встречались в обучении.

**Статистика после фильтрации:** Из исходных 9500 диалогов до стадии обучения дошло 2851 (около 30%), так как остальные не содержали валидных пар `User -> Assistant` или состояли из системных логов.

## 4. Обучение и метрики

### 4.1. Модель: facebook/opt-125m (Decoder-only)

**Параметры обучения:**

- Learning Rate: `5e-5`
- Оптимизатор: `adamw_torch`
- Стратегия генерации (Zero-shot): Top-p sampling (`top_p=0.9`, `temperature=0.7`)
- Стратегия генерации (Fine-tuned): Greedy search

| Стадия | BLEU | METEOR | Avg. длина ответа (слова) |
| --- | ---: | ---: | ---: |
| **Zero-shot** | 0.39 | 9.93 | 45.79 |
| **Fine-tuned (Validation)** | 1.83 | 11.81 | 25.13 |
| **Fine-tuned (Test)** | 1.49 | 12.45 | 22.45 |

**Анализ:** Несмотря на низкие показатели BLEU, дообучение дало качественный скачок. В режиме zero-shot модель страдала от тяжелых галлюцинаций (выдумывала несуществующие факты, путала роли, повторяла случайные токены). После обучения модель научилась давать короткие, релевантные и грамматически верные ответы. Снижение средней длины ответа с 45 до 25 слов свидетельствует об устранении проблемы бесконечного повторения (rambling).

### 4.2. Модель: google/flan-t5-small (Encoder-decoder)

**Параметры обучения:**

- Learning Rate: `2e-4`
- Оптимизатор: `adamw_torch`
- Стратегия генерации (Zero-shot): Greedy
- Стратегия генерации (Fine-tuned): Beam search (size=4) + Length penalty + Early stopping

| Стадия | BLEU | METEOR | Avg. длина ответа (слова) |
| --- | ---: | ---: | ---: |
| **Zero-shot** | 1.66 | 9.67 | 10.12 |
| **Fine-tuned (Validation)** | 6.32 | 16.95 | 20.26 |
| **Fine-tuned (Test)** | 4.81 | 15.49 | 17.45 |

**Анализ:** Flan-T5 показал себя значительно превосходящим OPT как в исходном состоянии, так и после дообучения. Благодаря архитектуре seq2seq и предварительному обучению на инструкциях (Flan-T5), модель изначально лучше понимала контекст диалога. Дообучение позволило кратно увеличить метрики (METEOR вырос почти в 1.7 раза). Модель стала давать более развернутые и семантически близкие к референсу ответы.

## 5. Сравнение качества генерации

### 5.1. Состояние до дообучения (Zero-shot)

В этом режиме модели используют только свои предтренированные веса без адаптации к конкретному датасету.

**Кейс 1: Творческий запрос (Акростих)**

**Контекст:**

> **User:** Please create an acrostic poem using the word "MOTHER." Format the poem like {letter in the word} - {line of poetry}. Put a space between each line of the poem.  
> **Reference:** M - My friend, O - Offering love without end. T - Thank you, Mom, for all you do...

| Модель | Ответ (Zero-shot) | Оценка |
| --- | --- | --- |
| **OPT-125m** | The following is a summary of all the dialogue dialogue: (1) The following dialog is to be used in your poem: | **Критическая.** Полный провал. Модель не поняла задачу, восприняла инструкцию как часть истории диалога и попыталась сгенерировать метаданные. |
| **Flan-T5-small** | You're welcome. I'll be in touch with you shortly. | **Низкая.** Модель выдала стандартную вежливую отписку, характерную для сервисных ботов. Она проигнорировала творческую часть запроса. |

**Кейс 2: Фактический вопрос (История)**

**Контекст:**

> **User:** Who was the first President of the United States?  
> **Reference:** George Washington was the first President of the United States.

| Модель | Ответ (Zero-shot) | Оценка |
| --- | --- | --- |
| **OPT-125m** | Presy McPresface was the first President of the United States. | **Критическая.** Тяжелая галлюцинация. Модель выдумала несуществующее имя, что является частой проблемой малых decoder-only моделей при работе с фактурой. |
| **Flan-T5-small** | George Washington was the first President of the United States. | **Высокая.** Flan-T5 справился с задачей идеально. Благодаря обученности на инструкциях (Instruct tuning), он распознал вопрос и извлек верный факт из памяти. |

### 5.2. Состояние после дообучения (Fine-tuned)

### Кейс 1: Ультрамарафон

**Контекст:**

> **User:** I just completed my ultra marathon. What do you think?  
> **Assistant:** Congratulations, James! That's an amazing accomplishment. It takes a lot of hard work and dedication to complete an ultra marathon. Keep up the great work and keep pushing yourself to reach your goals!  
> **User:** What can I do to celebrate my marathon?  
> **Reference:** Celebrating is a great way to acknowledge your success! Some ideas for celebrating could be getting a massage, treating yourself to a special treat, or gathering with friends and family to share the occasion.

| Модель | Ответ | Оценка |
| --- | --- | --- |
| **OPT-125m (Fine-tuned)** | You can take a photo of your running record and post it on Instagram, or you can post a picture of the finish line with your caption, "I'm so proud of myself!" | **Высокая.** Предложен конкретный, современный и выполнимый план действий. Ответ грамматически идеален и семантически плотный. |
| **Flan-T5-small (Fine-tuned)** | It's a great way to celebrate a marathon. You can do it with a friend or family member to celebrate your marathon. | **Средняя.** Модель уловила тему, но не предложила ни одного конкретного действия. Ответ семантически пустой и содержит тавтологию («celebrate a marathon... to celebrate your marathon»). |

**Вывод по сравнению:** В данном конкретном примере меньшая по метрикам модель (OPT) сгенерировала более полезный и креативный ответ, чем Flan-T5. Это демонстрирует, что BLEU/METEOR не всегда отражают реальную полезность (фактическую информативность) ответа для пользователя.

### Кейс 2: Физика и здравый смысл 

**Контекст:**

> **User:** Does a heavier object fall faster than a lighter one in a vacuum?  
> **Assistant:** No, in a vacuum all objects fall at the same rate regardless of their mass. This is because gravity accelerates all objects equally.  
> **User:** So if I drop a hammer and a feather on the Moon, what happens?  
> **Reference:** They will fall at the same rate and hit the ground at the same time. This was famously demonstrated by astronaut David Scott during the Apollo 15 mission.

| Модель | Ответ | Оценка |
| --- | --- | --- |
| **OPT-125m (Fine-tuned)** | They will fall at the same time. This is because there is no air resistance on the Moon to slow down the feather. | **Высокая.** Ответ абсолютно верен с научной точки зрения. Модель правильно указала причину (отсутствие сопротивления воздуха), что демонстрирует понимание физического принципа, а не просто заученную фразу. |
| **Flan-T5-small (Fine-tuned)** | The hammer and the feather will fall at the same time. | **Базовая.** Ответ верен, но минималистичен. Модель упустила важное уточнение про отсутствие воздуха и не упомянула исторический факт (Аполлон-15), который есть в референсе и важен для полноты ответа в образовательном диалоге. |

**Вывод по сравнению:** В первом кейсе OPT выиграл за счет креативности, во втором — за счет глубины объяснения. Flan-T5 в обоих случаях дает «безопасные», но поверхностные ответы, что характерно для малых seq2seq моделей, склонных к генерации наиболее вероятного (и часто самого общего) продолжения.


## 6. Итоговая таблица метрик

| model | stage | BLEU | METEOR |
| --- | --- | ---: | ---: |
| decoder_only_opt125m | zero_shot | 0.39 | 9.93 |
| decoder_only_opt125m | fine_tuned | 1.83 | 11.81 |
| decoder_only_opt125m | test | 1.49 | 12.45 |
| encoder_decoder_flan_t5_small | zero_shot | 1.66 | 9.67 |
| encoder_decoder_flan_t5_small | fine_tuned | 6.32 | 16.95 |
| encoder_decoder_flan_t5_small | test | 4.81 | 15.49 |

## 7. Заключение

1. **Архитектура:** Для задачи диалогового ответа в формате QA encoder-decoder архитектура (Flan-T5) является более предпочтительной. Она эффективнее усваивает паттерны «инструкция -> ответ» на малых объемах данных.

2. **Дообучение:** Обе модели успешно адаптировались к датасету. У OPT-125m эффект выразился в качественном переходе от бреда к релевантности, у Flan-T5 — в количественном росте метрик и следовании структуре референса.

