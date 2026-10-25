# Пояснение к структуре ML_QuantTunnel

Код проекта. Общее описание и физическая постановка — в корневом `README.md`.

## Структура
```
ML_QuantTunnel/
├── checkpoints/ веса обученных моделей (не в git, см. .gitignore)
├── configs/     генератор физических конфигураций
├── docs/        документация и пояснения
├── model/       PINN-модели на DeepXDE
├── physics/     физический модуль
└── scripts/     обучение и проверки
```


## Что где лежит

**`physics/`** — чистые функции физики, без нейросетей:
- `constants.py` — система единиц (ℏ = 1, m = 1)
- `potentials.py` — потенциалы барьера и ямы
- `initial_condition.py` — гауссов пакет и его нормировка
- `boundary_conditions.py` — граничные условия ψ(±L) = 0
- `analytics.py` — формулы для коэффициента прохождения D

**`docs/`** — документация:
- `validation_plan.md` — план тестов для валидации

**`configs/`** — генератор конфигураций:
- `generator.py` — генерация наборов (W, d, ε, b), разбиение train/val/test

**`model/`** — PINN на DeepXDE (бэкенд PyTorch, `DDE_BACKEND=pytorch`):
- `pinn_free.py` — Тест 1: PDE/IC/BC для свободного движения (V=0)

**`scripts/`** — обучение и проверки:
- `check_physics.py` — единый скрипт проверки физического модуля
- `train_free_particle.py` — обучение (или дообучение из чекпоинта) модели
  из `model/pinn_free.py` и проверка по критериям Теста 1 из
  `docs/validation_plan.md`

## Запуск проверки физики

```bash
python scripts/check_physics.py
```

## Обучение PINN (свободное движение)

```bash
# обучение с нуля
DDE_BACKEND=pytorch python scripts/train_free_particle.py

# дообучение из чекпоинта + доводка L-BFGS
DDE_BACKEND=pytorch python scripts/train_free_particle.py \
    --resume checkpoints/pinn_free-8000.pt --iterations 0 --lbfgs_maxiter 3000
