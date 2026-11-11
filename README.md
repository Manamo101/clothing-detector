# Clothing Detector POC

Projekt jest demonstracyjnym klasyfikatorem obrazów zbudowanym w PyTorch.
Model rozpoznaje pięć klas:

- `Shirts`
- `Jeans`
- `Cups`
- `Backpacks`
- `Watches`

## Założenia

- Model jest trenowany od zera — nie używa transfer learningu ani gotowych wag.
- Wejściowy obraz jest skalowany do rozmiaru szerokość `192` x wysokość `256`
  pikseli. W kodzie torchvision jest to zapisane jako `(256, 192)`, ponieważ
  używa kolejności `(wysokość, szerokość)`.
- Sieć jest lekką, własną architekturą CNN, opartą na blokach konwolucyjnych
  i `MaxPool2d`.
- Model zawiera tylko jedną warstwę liniową (`Linear`) na końcu.
- Każde zdjęcie treningowe powinno przedstawiać pojedynczy obiekt.
- Dane są podzielone na `train`, `val` i `test` w proporcji 80/10/10.
- `sliding_window.py` jest rozwiązaniem demonstracyjnym. Nie zastępuje
  pełnego detektora obiektów, takiego jak YOLO.

## Struktura danych

Skrypt `podziel_zbior.py` tworzy strukturę zgodną z `torchvision.datasets.ImageFolder`:

```text
dataset/
├── train/
│   ├── Backpacks/
│   ├── Cups/
│   ├── Jeans/
│   ├── Shirts/
│   └── Watches/
├── val/
│   └── ...
└── test/
    └── ...
```

Każdy folder klas powinien zawierać zdjęcia danej klasy. Puste foldery klas
nie wystarczą do treningu — każdy z podzbiorów musi zawierać co najmniej jedno
zdjęcie.

## Instalacja

Wybierz wariant zależnie od systemu i sprzętu. Najpierw sprawdź, czy
sterownik NVIDIA działa:

```powershell
nvidia-smi
```

Jeśli używasz CPU albo instalacja CUDA nie jest dostępna:

```powershell
python -m pip install -r .\requirements-cpu.txt
```


Aby pobrać wersję z CUDA, należy dobrać do aktualnego sterownika i systemu na stronie
[PyTorch - Start Locally](https://pytorch.org/get-started/locally/).
Pakiet PyTorch zawiera potrzebne biblioteki uruchomieniowe CUDA — nie trzeba
instalować pełnego CUDA Toolkit tylko po to, aby uruchomić trening.

Jeżeli wcześniej była zainstalowana inna wersja PyTorch, usuń ją w tym samym
środowisku Python przed zmianą wariantu:

```powershell
python -m pip uninstall torch torchvision torchaudio -y
```

Sprawdź, czy PyTorch widzi GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Oczekiwany wynik dla działającego GPU zawiera `True` oraz nazwę karty.
`False` oznacza, że należy używać `--device cpu` albo sprawdzić sterownik,
wersję PyTorch i wybrane środowisko Python.

W Google Colab wybierz runtime z GPU, a następnie zainstaluj wariant zgodny
z obrazem Colab:

```python
!pip install -r requirements.txt
!pip install torch torchvision
```

Nie instaluj jednocześnie wariantu CPU i CUDA w jednym środowisku.

## Przygotowanie danych

Jeżeli zdjęcia i plik CSV nie zostały jeszcze podzielone:

```powershell
python .\podziel_zbior.py `
  --images "C:\sciezka\do\zdjec" `
  --csv "C:\sciezka\do\plik.csv" `
  --output ".\dataset" `
  --seed 42
```

Przy zdjęciach o dużej rozdzielczości można podczas tworzenia zbioru zmniejszyć
maksymalny bok każdego zdjęcia, zachowując jego proporcje:

```powershell
python .\podziel_zbior.py `
  --images "C:\sciezka\do\zdjec" `
  --csv "C:\sciezka\do\plik.csv" `
  --output ".\dataset" `
  --resize 1024
```

Opcja `--resize` nie powiększa zdjęć, których dłuższy bok jest już mniejszy od
podanej wartości. Bez tej opcji zdjęcia są kopiowane bez zmian.

CSV musi zawierać:

- nazwę pliku lub identyfikator zdjęcia w kolumnie 1 (`id`),
- nazwę klasy w kolumnie 5 (`articleType`).

## Trenowanie

Uruchomienie z automatycznym wyborem urządzenia:

```powershell
python .\train.py `
  --dataset ".\dataset" `
  --output ".\runs\clothing_cnn" `
  --epochs 30 `
  --batch-size 64 `
  --image-size 192 256 `
  --device auto
```

Przy wymuszeniu GPU:

```powershell
python .\train.py `
  --dataset ".\dataset" `
  --output ".\runs\clothing_cnn" `
  --device cuda
```

Najważniejsze opcje:

| Opcja | Domyślna wartość | Znaczenie |
|---|---:|---|
| `--epochs` | `30` | Maksymalna liczba epok |
| `--batch-size` | `64` | Liczba zdjęć w batchu |
| `--learning-rate` | `0.001` | Współczynnik uczenia |
| `--patience` | `7` | Liczba epok bez poprawy przed early stopping |
| `--workers` | `2` | Liczba procesów ładowania danych |
| `--seed` | `42` | Powtarzalność treningu |
| `--image-size` | `192 256` | Szerokość i wysokość obrazu wejściowego w pikselach |
| `--device` | `auto` | `auto`, `cpu` albo `cuda` |

W folderze wynikowym zostaną zapisane:

- `best_model.pt` — checkpoint z najlepszą dokładnością walidacyjną,
- `history.json` — historia treningu oraz wynik na zbiorze testowym.

## Sliding window

Po wytrenowaniu modelu można przeskanować większe zdjęcie:

```powershell
python .\sliding_window.py `
  --model ".\runs\clothing_cnn\best_model.pt" `
  --image ".\duze_zdjecie.jpg" `
  --output ".\wynik.jpg" `
  --window-width 256 `
  --window-height 256 `
  --stride 64 `
  --threshold 0.85
```

Parametry:

- `--window-width`, `--window-height` — rozmiar przesuwanego okna,
- `--stride` — przesunięcie okna w pikselach,
- `--threshold` — minimalna pewność predykcji,
- `--nms-iou` — próg usuwania nakładających się detekcji.

Wynik zostanie zapisany jako obraz z prostokątami i etykietami klas.
Mniejsze `stride` daje dokładniejsze skanowanie, ale zwiększa czas działania.

## Pliki projektu

- [model.py](./model.py) — definicja `ClothingCNN`,
- [train.py](./train.py) — trening, walidacja i test,
- [sliding_window.py](./sliding_window.py) — demonstracyjna detekcja,
- [podziel_zbior.py](./podziel_zbior.py) — przygotowanie zbioru,
- [wymagania_projektu_sieci_cnn.md](./wymagania_projektu_sieci_cnn.md) — wymagania projektu.
