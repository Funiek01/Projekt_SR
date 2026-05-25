# Klasyfikacja sygnałów EEG ruchu wyobrażonego z wykorzystaniem logiki rozmytej i metod uczenia maszynowego

Projekt został wykonany w ramach przedmiotu **Systemy Rozmyte**.  
Celem projektu jest klasyfikacja sygnałów EEG związanych z ruchem wyobrażonym lewej i prawej dłoni oraz ocena wiarygodności decyzji klasyfikatora za pomocą systemu logiki rozmytej.

## Temat projektu

**Klasyfikacja sygnałów EEG ruchu wyobrażonego z wykorzystaniem logiki rozmytej i metod uczenia maszynowego**

Projekt skupia się na klasyfikacji dwóch klas:

- `LEFT_HAND` - wyobrażenie ruchu lewej dłoni,
- `RIGHT_HAND` - wyobrażenie ruchu prawej dłoni.

## Wykorzystane technologie

W projekcie wykorzystano:

- Python,
- NumPy,
- MNE,
- scikit-learn,
- scikit-fuzzy,
- Tkinter.

## Dataset

W projekcie wykorzystano dataset:

**PhysioNet EEG Motor Movement/Imagery**

Dane zawierają sygnały EEG zarejestrowane podczas rzeczywistego oraz wyobrażonego ruchu kończyn.  
W projekcie użyto przebiegów:

```text
4, 8, 12
```

Odpowiadają one zadaniu wyobrażenia ruchu lewej oraz prawej dłoni.

## Ogólny schemat działania programu

Pipeline projektu wygląda następująco:

```text
Wczytanie danych EEG
↓
Preprocessing sygnału
↓
Ekstrakcja cech
↓
Klasyfikacja ML
↓
Ocena wiarygodności przez system rozmyty
↓
Prezentacja wyników w GUI
```

## Preprocessing EEG

Dane EEG są wczytywane przy użyciu biblioteki **MNE**.  
Sygnał jest filtrowany w paśmie:

```text
8–45 Hz
```

Zakres ten pozwala wykorzystać pasma istotne dla ruchu wyobrażonego oraz wyznaczyć prosty wskaźnik zaszumienia sygnału.

## Ekstrakcja cech

Z każdej epoki EEG wyznaczane są cechy w postaci mocy pasmowej dla pasm:

```text
mu:   8-13 Hz
beta: 13-30 Hz
```

Wybrane pasma są istotne dla zadań typu motor imagery, ponieważ są związane z aktywnością kory sensomotorycznej.

Do ekstrakcji cech wykorzystano kanały EEG związane z obszarem ruchowym, m.in.:

```text
C3, C4, Cz, FC3, FC4, CP3, CP4, C1, C2, C5, C6
```

Ostatecznie każda próbka reprezentowana jest przez:

```text
22 cechy
```

czyli:

```text
11 kanałów × 2 pasma
```

## Klasyfikatory

W aplikacji dodano możliwość wyboru klasyfikatora z poziomu GUI.

Dostępne klasyfikatory:

- SVM RBF,
- Random Forest,
- LDA.

Najlepszy wynik uzyskano dla klasyfikatora:

```text
LDA
```

## System logiki rozmytej

Najważniejszym elementem związanym z przedmiotem **Systemy Rozmyte** jest moduł logiki rozmytej.

Klasyfikator ML zwraca decyzję:

```text
LEFT_HAND albo RIGHT_HAND
```

Natomiast system rozmyty ocenia, jak bardzo można ufać tej decyzji.

System rozmyty wykorzystuje trzy wejścia:

```text
confidence - największe prawdopodobieństwo klasyfikatora,
margin     - różnica między prawdopodobieństwami klas,
noise      - uproszczony wskaźnik zaszumienia sygnału EEG.
```

Na podstawie tych wartości system zwraca ocenę wiarygodności decyzji w skali:

```text
0-100
```

oraz etykietę lingwistyczną:

```text
PEWNA
UMIARKOWANA
NIEPEWNA
```

Przykładowa interpretacja reguł rozmytych:

```text
Jeżeli confidence jest wysokie,
margin jest duży,
a noise jest niski,
to decyzja jest PEWNA.

Jeżeli confidence jest średnie,
to decyzja jest UMIARKOWANA.

Jeżeli confidence jest niskie,
margin jest mały
lub noise jest wysoki,
to decyzja jest NIEPEWNA.
```

## Graficzny interfejs użytkownika

Projekt posiada prosty graficzny interfejs użytkownika wykonany z użyciem biblioteki **Tkinter**.  
Jest to standardowa biblioteka Pythona do tworzenia aplikacji okienkowych, dlatego nie wymaga instalowania dodatkowego frameworka GUI.

Interfejs umożliwia wygodne uruchamianie eksperymentów bez konieczności ręcznej edycji kodu. Użytkownik może z poziomu okna aplikacji:

- wpisać numery badanych,
- wpisać numery przebiegów,
- wybrać klasyfikator,
- uruchomić trening modelu,
- zobaczyć wyniki klasyfikacji,
- zobaczyć wyniki modułu logiki rozmytej,
- zapisać wytrenowany model do pliku.

Po uruchomieniu programu otwiera się okno aplikacji.

W polu **Numery badanych** należy wpisać numery badanych oddzielone przecinkami, np.:

```text
7
```

lub:

```text
4,5,7
```

W polu **Runs** należy wpisać numery przebiegów.  
Dla klasyfikacji lewej i prawej dłoni używana jest konfiguracja:

```text
4,8,12
```

Następnie z listy rozwijanej **Klasyfikator** można wybrać jeden z dostępnych modeli:

```text
SVM RBF
Random Forest
LDA
```

Po kliknięciu przycisku **Trenuj model** aplikacja wykonuje cały pipeline:

```text
wczytanie danych EEG → ekstrakcja cech → trening klasyfikatora → ocena fuzzy → pokazanie wyników
```

Wyniki pojawiają się w dużym polu tekstowym w oknie aplikacji. Program wyświetla między innymi:

- liczbę epok EEG,
- liczbę cech,
- użyte kanały,
- wybrany klasyfikator,
- accuracy,
- classification report,
- macierz pomyłek,
- średnią wiarygodność decyzji fuzzy,
- liczbę decyzji `PEWNA`, `UMIARKOWANA`, `NIEPEWNA`,
- przykładowe decyzje testowe wraz z prawdopodobieństwami klas.

Po zakończeniu treningu można użyć przycisku **Zapisz model**, aby zapisać wytrenowany model do pliku `.joblib`.

## Najlepsze uzyskane wyniki

Najlepsze wyniki uzyskano dla konfiguracji:

```text
Badany: 7
Runs: 4, 8, 12
Klasyfikator: LDA
Liczba epok: 45
Liczba cech: 22
```

Wyniki klasyfikacji:

```text
Accuracy: 0.833
```

Macierz pomyłek:

```text
[[5 1]
 [1 5]]
```

Oznacza to, że model poprawnie sklasyfikował:

```text
10 z 12 próbek testowych
```

Dla obu klas uzyskano zbalansowane wyniki:

```text
LEFT_HAND:  precision = 0.83, recall = 0.83, f1-score = 0.83
RIGHT_HAND: precision = 0.83, recall = 0.83, f1-score = 0.83
```

## Wyniki systemu rozmytego

Dla najlepszego wariantu system rozmyty uzyskał:

```text
Średnia wiarygodność decyzji: 76.7/100
PEWNA: 8 próbek
UMIARKOWANA: 4 próbki
NIEPEWNA: 0 próbek
```

Oznacza to, że większość decyzji została oceniona jako wiarygodna.


## Podsumowanie

W projekcie udało się przygotować działający hybrydowy system klasyfikacji EEG, który łączy klasyczne metody uczenia maszynowego z modułem wnioskowania rozmytego.

Najlepszy uzyskany wynik klasyfikacji wyniósł:

```text
83.3%
```

Najlepszą konfiguracją okazało się połączenie cech `mu + beta` z klasyfikatorem `LDA`, które dla badanego nr `7` osiągnęło `Accuracy = 0.833` oraz średnią wiarygodność fuzzy `76.7/100`.

Najważniejszy element projektu stanowi system rozmyty, który pozwala nie tylko otrzymać decyzję klasyfikatora, ale również ocenić jej wiarygodność. Dzięki temu projekt nie jest wyłącznie klasycznym modelem uczenia maszynowego, lecz systemem hybrydowym łączącym ML oraz logikę rozmytą.