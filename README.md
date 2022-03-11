# gpm_test  тестовое задание
## структура каталогов

```
├── data
│   ├── original
│   │   ├── images                                                      каталог с изображениями
│   │   └── images_labelling.csv                                        классы
│   └── preprocessed                                                    дополнительные данные
│       └── team_membership.json                                        соответсвие игрока и команды
├── Dockerfile     
├── model_weights                                                       каталог с весами
│   └── team_clf.pth
├── README.md
├── requirements.txt
├── scripts каталог                                                     со скриптами 
│   ├── build_docker_image.sh  
│   ├── download_dataset.sh
│   ├── pipeline.sh     
│   ├── start_docker.sh
│   └── train.sh
├── src                                                                 исходники 
│   ├── lib
│   │   ├── augmentation_transform.py
│   │   ├── common_utils.py
│   │   ├── football_players_dataset.py
│   │   ├── __init__.py
│   │   ├── team_classifier_model.py
│   │   └── train_test_utils.py
│   ├── predict_team_cli.py                                             код для предсказания через командную строку
│   ├── predict_team_rest_server.py                                     код для запуска серсива с rest api
│   └── train_clf.py                                                    код для тренировки модели
└── задание.txt
```

# установка 
```
git clone https://github.com/SokoloiD/gpm_test
cd gpm_test
pip install -r requirements.txt
```


# запуск одной командой
```commandline
./scripts/pipeline.sh 
```
- скачивается датасет
- производится обучение модели
- создается докер образ
- запускается докер контейнер

проверка работоспособности REST_API
```commandline
curl --request POST -F "file=@./data/original/images/203.png"   localhost:8000/predict

```

# запуск отдельными командами
##  скачать датасет
```commandline
./scripts/download_dataset.sh

```
##  обучить модель
```commandline
./scripts/train.sh

```
## создать докер образ
```commandline
./scripts/build_docker_image.sh

```
## запустить  докер контейнер
```commandline
./scripts/start_docker.sh
```
## проверить работу
```commandline
curl --request POST -F "file=@./data/original/images/203.png"   localhost:8000/predict
```





# Отчет

В задании выполнено предсказание типа (класса) команды по изображению игрока. Детекция игрока не  сделана по следующим соображениям
- в исходном датасете большинство изображений игроков являются соседними кадрами одного видеоряда. т.е классы в датасете недостаточно предстваленны. Формально подойдя  к заданию можно получит предсказания по изображению для конкретного игрока
но по факту это будет переобученная модель, коротая на других данных работать не будет
- разрешение большинства изображений игроков недостаточно для детекции
- на мой взгляд для детекции игроков следует применять  подход, включающий в себя object tracking.
Например, с использованием DEEP SORT, и далее брать уже несколько изображений с трека для детекции

## общее описание

На основе представленных данных сделана дополнительная разметка (файл data/preprocessed/team_membership.json), содержащией перечень
команд и словарь для определения имени команды по имени класса игрока.

Датасет (src/lib/football_players_dataset.py) возвращает изображение игрока и имя класса. 
В датасете дополнительно реализоана функция, возвращающая список индексов для тренировочного и тестового 
набора такой, что все классы в нем представленны в заданной пропорции. Все преобразования (имя классв в номер команды и тд)
реализованиы через transform - список последовательных трансформаций, которые передабтся в датасет и применяются внутри 
__get_item__.

Для классификации используется относительно легкая сеть resnet18 из стандартного репозитория pytorch. 
В качестве лосс-функции - corossentropyloss, оптимизатор -Adam, шедулер -ReduceLROnPlateau. GPU используется при наличии. 


Параметры, достигнутые при обучении (классификация на 5 команд "blue","white", "red", "other", "green","yellow"):
```commandline
confusion matrix
[[300   0   0   0   0   0]
 [  0 299   0   0   1   0]
 [  0   0  60   0   0   0]
 [  1   3   0  26   0   0]
 [  0   0   0   0  30   0]
 [  0   0   0   0   0  30]]


classification report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       300
           1       0.99      1.00      0.99       300
           2       1.00      1.00      1.00        60
           3       1.00      0.87      0.93        30
           4       0.97      1.00      0.98        30
           5       1.00      1.00      1.00        30

    accuracy                           0.99       750
   macro avg       0.99      0.98      0.98       750
weighted avg       0.99      0.99      0.99       750

```

Ввиду малого количества уникальных изображений для каждого класса  выборка разбивалась только на обучающую и тестовую 
без валидационного набора. 
В результате обучения веса для наилучшей модели сохраняются в model_weights/team_clf.pth


Для предсказания имени команды по изображению игрока используюся две утилиты
- из командной строки src/predict_team_cli.py
- REST API  src/predict_team_rest_server.py

пример использования:

```commandline
predict_team_cli.py -l model_weights/team_clf.pth -t data/preprocessed/team_membership.json -i data/original/images/203.png
```

```commandline

python src/predict_team_rest_server.py \
    -l model_weights/team_clf.pth \
    -t data/preprocessed/team_membership.json
    
curl --request POST -F "file=@./data/original/images/203.png" localhost:8000/predict

```
# TODO

1. В confusion matrix содержатся ненулевые элеменды в клетках не на главной диагонали. Нужно разобраться где классификатор ошибается
фвв


