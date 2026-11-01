# Podsumowanie Projektu: Detekcja Ubrań (PyTorch)

## 1. Cel Główny
Celem projektu jest stworzenie i wytrenowanie od zera sztucznej sieci neuronowej, która będzie w stanie **rozpoznawać 5 klas ubrań**. Docelowo model ma być wykorzystywany do **detekcji wielu ubrań na jednym, większym zdjęciu**.

## 2. Wymagania dotyczące Architektury Sieci
Zamiast korzystać z ciężkich, gotowych architektur (takich jak pełne VGG16, które jest zbyt duże i wolne do trenowania od zera), sieć zostanie zbudowana jako **Custom CNN** (własna konwolucyjna sieć neuronowa) spełniająca warunki:
* **Framework:** PyTorch.
* **Wejście:** Obrazy o szerokości **60** i wysokości **80 pikseli**.
* **Warstwy Liniowe (Fully Connected):** Ograniczone do minimum – sieć ma posiadać **tylko jedną warstwę liniową** na samym końcu (odpowiadającą za klasyfikację).
* **Ekstrakcja cech:** Oparta na blokach konwolucyjnych i warstwach *Max Pooling* (wzorowanych na architekturze VGG), które drastycznie zmniejszą wymiary przestrzenne obrazu przed trafieniem do jedynej warstwy liniowej.
* **Szybkość:** Zastosowane odchudzenie architektury ma zagwarantować szybki czas trenowania (od kilkunastu minut do maksymalnie kilku godzin) w darmowym środowisku chmurowym.

## 3. Strategia Danych i Uczenia
* **Trening od zera:** Model nie będzie korzystał z uczenia transferowego (Transfer Learning) ani wstępnie wytrenowanych wag. Uczy się zupełnie od podstaw.
* **Zbiór danych:** Własny zestaw zdjęć, w którym **pojedyncze zdjęcie zawiera tylko jedno ubranie** (np. na czystym tle). Klasy ubrania znajdują się w pliku CSV, który zawiera w 1. kolumnie (id) nazwę pliku, a w 5. kolumnie (articleType) interesującą klasę ubrania.
* **Podział danych:** Zastosowany zostanie klasyczny podział na zbiór treningowy (do nauki wag) i walidacyjny (do monitorowania procesu i zapobiegania przeuczeniu).

## 4. Mechanizm Detekcji na Docelowym Obrazie (Sliding Window)
Z uwagi na to, że model będzie trenowany jako klasyfikator na pojedynczych obiektach, a cel wymaga detekcji wielu obiektów na dużym zdjęciu/wideo, zastosowana zostanie metoda **Przesuwanego Okna (Sliding Window)**:
1. Program nałoży wirtualną ramkę (np. 256x256) na duże zdjęcie docelowe.
2. Wytnie fragment i prześle go do wytrenowanego klasyfikatora.
3. Jeśli model z wysoką pewnością rozpozna jedną z klas, program narysuje w tym miejscu prostokąt i podpisze go nazwą ubrania.
4. Ramka będzie przesuwana po całym zdjęciu, skanując je fragment po fragmencie, potencjalnie w różnych skalach, aby wykryć obiekty o różnych rozmiarach.

Należy sprawdzić czy istnieją już gotowe implementacje Sliding Window, które można wykorzystać

## 5. Środowisko
* **Platforma:** Google Colab.
* **Akceleracja:** Karta graficzna GPU (np. T4) w celu zrównoleglenia obliczeń operacji konwolucyjnych.

## 6. Przygotowanie folderów z danymi

Skrypt `podziel_zbior.py` filtruje klasy `Shirts`, `Jeans`, `Cups`, `Backpacks`
i `Watches`, a następnie kopiuje zdjęcia do folderów `train`, `val` i `test`
w proporcjach 80/10/10. Podział jest stratyfikowany, czyli każda klasa jest
dzielona osobno.

Przykład uruchomienia w PowerShell:

```powershell
python .\podziel_zbior.py `
  --images "C:\sciezka\do\zdjec" `
  --csv "C:\sciezka\do\plik.csv" `
  --output ".\dataset" `
  --seed 42
```

CSV powinien zawierać nazwę zdjęcia w kolumnie 1 (`id`) i etykietę w kolumnie
5 (`articleType`). Zdjęcia bez wybranej etykiety są pomijane. Folder wynikowy
powinien być pusty, aby przypadkowo nie połączyć dwóch różnych podziałów.

## 7. Trenowanie klasyfikatora

Zależności można zainstalować lokalnie lub w Google Colab:

```powershell
python -m pip install -r requirements.txt
```

Trening od zera uruchamia plik `train.py`:

```powershell
python .\train.py `
  --dataset ".\dataset" `
  --output ".\runs\clothing_cnn" `
  --epochs 30 `
  --batch-size 64 `
  --device auto
```

Model używa wejścia o szerokości 60 i wysokości 80 pikseli, bloków konwolucyjnych z Max Pooling
i dokładnie jednej warstwy liniowej na wyjściu. Najlepsze wagi są zapisywane
w `runs/clothing_cnn/best_model.pt`, a wyniki epok w `history.json`.
Argument `--device cuda` wymusza GPU; `--device auto` wybiera GPU, jeśli jest
dostępne, w przeciwnym razie CPU.

## 8. Sliding window

Po treningu można przeskanować większe zdjęcie:

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

Skrypt stosuje klasyfikator do kolejnych fragmentów i usuwa nakładające się
wyniki. Jest to rozwiązanie demonstracyjne: klasyfikator trenowany na jednym
obiekcie nie zastępuje pełnego detektora obiektów i może wymagać dostrojenia
rozmiaru okna, kroku oraz progu pewności.