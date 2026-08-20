# Czy post-training zmienia odporność na błędne prefiksy rozumowania?

Data przeglądu: 2026-08-12

## Pytanie i werdykt

Pytanie: czy istnieje już praca, która porównuje model bazowy z jego bezpośrednio post-trenowanym lub destylowanym potomkiem na tych samych poprawnych, subtelnie zepsutych i poprawionych prefiksach rozumowania, mierzy poprawność kontynuacji / odzyskanie po błędzie / propagację błędu i wiąże sparowaną zmianę zachowania ze zlokalizowaną zmianą aktywacji?

**Werdykt:** nie znalazłem pracy realizującej cały ten projekt naraz, ale jego niesparowana wersja jest już bardzo blisko zajęta. Najbliższa praca, anonimowy preprint TMLR *From Decorative to Load-Bearing*, wstrzykuje pojedynczy błąd do kroku CoT, ucina ślad, wymusza kontynuację, klasyfikuje bypass / self-correction / error propagation, sonduje stany ukryte i osobno pokazuje, że `DeepSeek-R1-Distill-Qwen-7B` częściej sam się poprawia. Nie porównuje go jednak z jego bezpośrednim checkpointem źródłowym `Qwen2.5-Math-7B` na tych samych przykładach ([OpenReview PDF](https://openreview.net/pdf?id=TiZQnKDIHq)).

Dlatego obronna teza nowości brzmi węziej:

> **Jak destylacja reasoningowa zmienia przyczynową podatność dokładnie tego samego modelu na pierwszy rzeczywisty błąd w cudzym toku rozumowania — i czy ta sparowana zmiana funkcjonalna jest związana ze zmianą obliczenia wewnętrznego przy granicy błędu?**

To jest nowość **kompozycyjna i kontrolna**, nie nowość każdego składnika. Samo „prefix utility”, samo wstrzykiwanie błędów, samo badanie self-correction, samo porównanie base/post-trained ani sama lokalizacja aktywacji nie są już nowe.

## Najbliższe prace i dokładna granica nowości

| Praca | Co już robi | Czego nie robi względem proponowanego eksperymentu |
|---|---|---|
| *From Decorative to Load-Bearing: Task Difficulty Shapes Chain-of-Thought Faithfulness* (TMLR, under review, zmodyfikowana 2026-06-26) | Pojedyncza perturbacja kroku, ucięcie, kontynuacja od błędnego prefiksu; trzy tryby zachowania; 28 584 kontynuacje; hidden-state probes i steering; osobny wynik dla `DeepSeek-R1-Distill-Qwen-7B`, gdzie trening reasoningowy przesuwa zachowanie ku self-correction ([PDF](https://openreview.net/pdf?id=TiZQnKDIHq), [status](https://openreview.net/submissions?page=29&venue=TMLR)). | Nie ma sparowanego rodzica `Qwen2.5-Math-7B`, nie izoluje więc efektu destylacji od rodziny/modelu i trudności. Perturbacje są sztuczne; nie używa ekspercko oznaczonego pierwszego naturalnego błędu. Nie ma poprawionego kontrfaktycznego wariantu tego samego kroku ani między-checkpointowego displacementu. |
| *DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes* | Trenuje modele przez RL, aby odzyskiwały poprawne rozwiązanie po prefiksach błędnych trajektorii słabszych modeli ([arXiv](https://arxiv.org/abs/2605.28421), [kod](https://github.com/ALEX-nlp/DenoiseRL)). | To metoda treningowa, nie kontrolowana diagnoza tego, co zmieniła konkretna destylacja; brak matched parent/descendant i lokalizacji aktywacji. |
| *From Correctness to Utility: Gain-Based Prefix Evaluation for LLM Reasoning* (PUM) | Definiuje dokładnie funkcjonalną wartość prefiksu jako zmianę prawdopodobieństwa poprawnego rozwiązania „z prefiksem minus bez prefiksu”, mierzoną kontynuacjami wielu studentów; rozdziela utility wspólne i zależne od polityki ([arXiv](https://arxiv.org/abs/2606.07190), [pełny tekst](https://arxiv.org/html/2606.07190v1)). | Ogólna idea „czy prefiks pomaga kontynuacji” nie jest już nowa. PUM nie bada kontrolowanej korupcji/popravy, nie porównuje bezpośredniego potomka z rodzicem i nie analizuje aktywacji obu checkpointów. |
| *The Potential of CoT for Reasoning* | Mierzy, jak części CoT zmieniają szansę poprawnej kontynuacji, i pokazuje transfer prefiksów z silniejszego do słabszego modelu ([arXiv](https://arxiv.org/abs/2602.14903)). | Brak błędów kontrfaktycznych, matched training pair i analizy aktywacji. |
| *Reasoning that Travels* | Dostawca–odbiorca, rosnące prefiksy, force-answer i free-generation; pokazuje, że częściowe CoT mogą scaffoldować dalsze rozumowanie ([arXiv](https://arxiv.org/abs/2605.28913)). | Brak korupcji/correction, matched parent/descendant i aktywacji. |
| *Reliable Chain-of-Thought via Prefix Consistency* | Ucina własny CoT i regeneruje resztę; stabilność odpowiedzi służy jako sygnał poprawności ([arXiv](https://arxiv.org/abs/2605.07654)). | Nie wprowadza oznaczonego błędu ani nie bada różnicy wywołanej post-trainingiem. |
| *Self-Correction Bench* (COLM 2026) | Wstrzykuje identyczny błąd jako „własny” lub „zewnętrzny”, identyfikuje self-correction blind spot i mechanistycznie znajduje przyczynowy kierunek roli konwersacyjnej ([arXiv v3](https://arxiv.org/abs/2507.02778)). | Nie jest eksperymentem kontynuacji po matematycznej granicy pierwszego błędu i nie porównuje bezpośredniego parent/descendant. Odbiera za to nowość ogólnemu hasłu „post-training włącza korektę błędów i ma wewnętrzny mechanizm”. |
| *Hidden Error Awareness in Chain-of-Thought Reasoning* | Proby hidden states przewidują poprawność, także dla `DeepSeek-R1-Distill-Qwen-7B`; sygnał jest zlokalizowany głównie w górnych warstwach, lecz patching/steering nie daje wiarygodnej korekty ([arXiv](https://arxiv.org/abs/2605.09502)). | Brak matched parent i tego samego kontrolowanego prefiksu; etykieta dotyczy poprawności trajektorii, nie różnicy odporności na pierwszy błąd. |
| *Rethinking RL for LLM Reasoning: It’s Sparse Policy Selection* | Porównuje modele base i RL, lokalizuje nieliczne różnice tokenowe przy wysokiej entropii i robi interwencje, które odzyskują dużą część zysku RL ([arXiv](https://arxiv.org/abs/2605.06241)). | Silnie zajmuje ogólną tezę „post-training zmienia decyzje w lokalnych punktach rozumowania”, ale nie bada reakcji obu checkpointów na identyczny cudzy poprawny/błędny/poprawiony prefiks ani granicy błędu z ProcessBench. |
| *Localizing Reasoning Training-Induced Changes in LLMs* | Porównuje 19 Qwen-derived reasoning models z modelami bazowymi; CKA wskazuje środkowe warstwy, SFT zmienia więcej niż RL ([OpenReview](https://openreview.net/forum?id=WDlhBhceGZ), [kod](https://github.com/mklabunde/localizing-reasoning)). | Zabiera nowość samej mapie CKA/RSA/displacement. Nie łączy zmiany reprezentacji z parowanym efektem błędnego prefiksu. |
| *How Post-Training Reshapes LLMs* (COLM 2025) | Mechanistycznie porównuje base/post-trained w wiedzy, truthfulness, refusal i confidence; bada podobieństwo oraz transfer kierunków aktywacji ([arXiv](https://arxiv.org/abs/2504.02904)). | Nie dotyczy kontynuacji reasoningowych ani recovery/propagation. |
| *LLM Reasoning as Trajectories* (ACL 2026) | Pokazuje uporządkowaną geometrię kroków, różnice correct/wrong i efekt reasoning trainingu na szybkość konwergencji; zawiera trajectory steering ([arXiv](https://arxiv.org/abs/2604.05655)). | Nie ma kontrolowanego wspólnego prefiksu z pierwszym błędem i sparowanego parent→descendant efektu zachowania. |
| *PRISM* | Pokazuje, że RL może zachować geometrię reprezentacji z CKA > 0.998 mimo zmian wydajności ([arXiv](https://arxiv.org/abs/2603.17074)). | To ważne ostrzeżenie: sam CKA/displacement nie wystarczy. Nie bada błędnych prefiksów. |

### Konsekwencja dla claimu

Nie należy pisać:

- „pierwszy test, czy błędny prefiks propaguje się w LLM” — robi to *From Decorative to Load-Bearing*;
- „pierwsza funkcjonalna miara wartości prefiksu” — robią to PUM i wcześniejszy CoT potential;
- „pierwsze porównanie wewnętrzne base i reasoning model” — robią to Klabunde–Lemmerich, Du et al. i inne prace;
- „pierwsze wykrycie wewnętrznego sygnału błędu” — robią to *Hidden Error Awareness* i prace o self-verification;
- „pierwszy związek post-trainingu z lokalnymi punktami decyzji” — bardzo blisko robi to *Sparse Policy Selection*.

Można natomiast, po ponownym przeglądzie tuż przed submission, bronić:

- pierwszego **within-lineage**, example-matched oszacowania wpływu destylacji na przyczynową load-bearingness rzeczywistych błędów rozumowania;
- pierwszego kontrastu parent→distilled wokół **tej samej ekspercko oznaczonej granicy pierwszego błędu**;
- pierwszego połączenia sparowanej zmiany recovery/propagation z aktywacją mierzoną na tej samej treści i tej samej pozycji semantycznej.

## Pięć składników literatury

### 1. Corrupted/faithful CoT i propagacja błędu

Lanham et al. interweniują na CoT przez dodawanie błędów, parafrazy i skracanie oraz pokazują, że stopień polegania na CoT zależy od zadania i modelu ([arXiv](https://arxiv.org/abs/2307.13702)). Najnowsze *From Decorative to Load-Bearing* robi już najbardziej bezpośrednią wersję planowanego testu: perturbuje jeden krok, ucina i wymusza dalszą generację, a następnie rozdziela bypass, self-correction i propagation ([OpenReview PDF](https://openreview.net/pdf?id=TiZQnKDIHq)).

Tyen et al. pokazują inną ważną granicę: modele słabo znajdują błąd, ale znacznie lepiej korygują, gdy podana jest jego lokalizacja ([arXiv](https://arxiv.org/abs/2311.08516)). To uzasadnia rozdzielenie „wykrył błąd” od „potrafił odzyskać poprawne rozwiązanie”. DenoiseRL dalej pokazuje, że recovery z błędnych prefiksów może być bezpośrednim celem treningowym ([arXiv](https://arxiv.org/abs/2605.28421)).

### 2. Kontynuacja prefiksu i korekta

PUM formalizuje prefix gain jako `P(correct | problem, prefix) - P(correct | problem)` ([pełny tekst](https://arxiv.org/html/2606.07190v1)). *Reasoning that Travels* i *The Potential of CoT* używają prefiksów do transferu lub scaffoldingu między modelami ([arXiv 2605.28913](https://arxiv.org/abs/2605.28913), [arXiv 2602.14903](https://arxiv.org/abs/2602.14903)). Prefix continuation jako taki jest więc zajęty.

Otwarte pozostaje **difference-in-differences**:

`(distilled_corrupt - distilled_clean) - (parent_corrupt - parent_clean)`,

oraz analogiczny kontrast dla prefiksu naprawionego. To izoluje zmianę podatności spowodowaną przejściem parent→distilled, zamiast pytać tylko, czy pojedynczy model wykorzystuje prefiks.

### 3. Matched base vs post-trained/distilled

Oficjalne materiały DeepSeek wskazują `Qwen2.5-Math-7B` jako base model dla `DeepSeek-R1-Distill-Qwen-7B`; destylowane checkpointy powstały przez fine-tuning na 800 tys. próbek wygenerowanych przez DeepSeek-R1 ([repozytorium DeepSeek, tabela modeli i licencja](https://github.com/deepseek-ai/DeepSeek-R1#3-model-downloads)). To czyni tę parę znacznie czystszą niż porównanie z `Qwen2.5-7B-Instruct`.

Jednocześnie DeepSeek explicite ostrzega, że zmienił konfiguracje i tokenizery ([repozytorium, linia pod tabelą modeli](https://github.com/deepseek-ai/DeepSeek-R1#3-model-downloads)). Dlatego „byte-identical” może dotyczyć tylko wstawianej treści problemu i prefiksu. Nie wolno twierdzić, że modele widzą identyczne token IDs lub identycznie serializowany prompt. Należy raportować oba tokenizery, token counts, template i pozycję względną.

### 4. Lokalizacja wewnętrzna: asocjacyjna kontra przyczynowa

Klabunde–Lemmerich zajmują opisową lokalizację zmian treningowych CKA ([OpenReview](https://openreview.net/forum?id=WDlhBhceGZ)); *Hidden Error Awareness* pokazuje, że sygnał diagnostyczny może nie być dźwignią przyczynową ([arXiv](https://arxiv.org/abs/2605.09502)); PRISM pokazuje, że globalne podobieństwo CKA może pozostać niemal idealne mimo funkcjonalnej poprawy ([arXiv](https://arxiv.org/abs/2603.17074)).

Wobec tego związek `activation displacement -> behavioral delta` należy nazywać **asocjacyjnym**, dopóki nie ma interwencji. Minimalna wersja pracy może pozostać asocjacyjna, ale claim musi brzmieć „predicts/marks”, nie „causes”. Jeśli potrzebny jest claim przyczynowy, najtańszym dodatkiem jest jeden predeclared patching test w warstwie/paśmie wybranym na development split i ocenionym raz na holdoucie; negatywny wynik nadal byłby informacyjny wobec *Hidden Error Awareness*.

### 5. Benchmarki process supervision i verification

`ProcessBench` ma 3 400 trudnych przypadków, wiele generatorów oraz pierwszy błędny krok oznaczony przez wielu ekspertów; poprawne przypadki mają etykietę `-1` ([arXiv/ACL 2025](https://arxiv.org/abs/2412.06559)). To obecnie najlepszy gotowy anchor do naturalnej granicy pierwszego błędu. Benchmark został jednak zbudowany do **identyfikacji** błędu, nie do continuation, więc proposed task byłby nowym użyciem danych.

`PRM800K` zawiera około 800 tys. ludzkich etykiet poprawności kroków dla rozwiązań MATH ([oficjalne repozytorium OpenAI](https://github.com/openai/prm800k), [paper](https://arxiv.org/abs/2305.20050)). `Math-Shepherd` automatycznie etykietuje krok przez wielokrotne dokończenia i sprawdzenie odpowiedzi ([arXiv](https://arxiv.org/abs/2312.08935)). Te zasoby potwierdzają, że „czy z tego prefiksu da się dojść do poprawnej odpowiedzi” jest utrwaloną ideą w process supervision; nowość musi pochodzić z matched training transition i kontrfaktycznego błędu.

## Zalecany, minimalny projekt

### Rdzeń, który nadal warto zrobić

1. **Modele:** tylko `Qwen/Qwen2.5-Math-7B` i `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` w pierwszym sprincie. Ta para jest oficjalnie parent→distilled ([DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1#3-model-downloads)).
2. **Dane:** losowy, zamrożony podzbiór błędnych rozwiązań ProcessBench z jednoznaczną granicą pierwszego błędu. Start od małego pilotu 50–100, potem dopiero moc/statystyka. Nie zaczynać od MATH-500 z arbitralnym syntetycznym błędem, bo ten design jest już pokryty przez najbliższą pracę.
3. **Trzy warianty na przypadek:**
   - `PRE`: prefiks kończy się tuż przed pierwszym błędnym krokiem;
   - `ERR`: ten sam prefiks plus ekspercko oznaczony pierwszy błędny krok;
   - `FIX`: `ERR`, ale tylko pierwszy błędny krok zastąpiony minimalną poprawką zatwierdzoną ręcznie.
4. **Wyniki behawioralne:** final-answer correctness oraz mutually exclusive `bypass / explicit self-correction / propagation`. Osobno raportować wykrycie/werbalizację błędu, bo znalezienie błędu i zdolność korekty są różnymi zdolnościami ([Tyen et al.](https://arxiv.org/abs/2311.08516)).
5. **Główny estimand:** sparowana zmiana efektu `ERR` względem `PRE` między potomkiem a rodzicem; `FIX` jest sprawdzeniem, czy destylacja pomaga używać naprawionego stanu, a nie tylko uruchamia frazę „wait”. Klaster/bootstrap na problemie, nie na trzech wariantach osobno.
6. **Aktywacje:** last-prefix-token dla wszystkich warstw i jawne porównanie przy granicy semantycznej. Predeclare pasmo na development split; testować, czy różnica aktywacji przewiduje paired behavioral delta poza długością, token count, position, residual norm i confidence. Nie przedstawiać samego displacementu/CKA jako wkładu.
7. **Holdout:** wybór warstw, agregacji i reguł klasyfikacji tylko na dev; jednorazowy test na zamrożonym holdoucie. Najbliższa praca ma już duży benchmark, więc przewaga tej pracy musi pochodzić z czystości identyfikacji, nie tylko skali.

### Najmniejszy pilot zabijający pomysł szybko

Na 50 przypadkach uruchomić oba checkpointy dla `PRE` i `ERR`, deterministycznie i dodatkowo kilkoma próbkami dla solve-rate. Kontynuować tylko wtedy, gdy:

- istnieje niezerowy, stabilny paired difference-in-differences w recovery/propagation;
- wynik nie jest wyłącznie skutkiem tego, że distilled model ma wyższą bazową accuracy;
- klasyfikacja zachowania daje wysoką zgodność człowieka z regułami/judge;
- efekt nie znika po zrównaniu długości i jawnej kontroli różnic tokenizacji.

`FIX` i aktywacje można dodać po przejściu tego gate. To najtańszy sposób uniknięcia pełnego eksperymentu, jeśli teza funkcjonalna nie istnieje.

## Ryzyka dla nowości i interpretacji

1. **Najbliższy konkurent może się zaktualizować.** *From Decorative to Load-Bearing* jest nadal w recenzji TMLR i wersja już została rozszerzona o DeepSeek-R1-Distill. Trzeba sprawdzić jego najnowszą rewizję bezpośrednio przed submission ([status OpenReview](https://openreview.net/submissions?page=29&venue=TMLR)).
2. **Lepsza ogólna accuracy nie jest odpornością.** Główny wynik musi być interakcją model × condition, nie różnicą surowej poprawności.
3. **„Same text” nie oznacza same tokens.** DeepSeek deklaruje zmiany tokenizera/configu; treść musi być zachowana bajtowo, lecz różnice tokenizacji raportowane i kontrolowane ([DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1#3-model-downloads)).
4. **ProcessBench nie dostarcza gotowych poprawek.** Oznacza pierwszy błąd, lecz wariant `FIX` trzeba stworzyć i zatwierdzić. Minimalna poprawka jednego kroku jest lepsza niż pełne przepisywanie rozwiązania.
5. **Probe nie daje mechanizmu.** Poprawna narracja bez patchingu: „training-induced activation marker of changed recovery”. Narracja „mechanism” wymaga interwencji i kontroli losowej/warstwowej.
6. **Środkowe warstwy nie mogą być odkryciem.** Ten wynik ma bezpośredni prior art; warstwy trzeba prerejestrować lub wybrać na dev i testować na holdoucie ([Klabunde–Lemmerich](https://openreview.net/forum?id=WDlhBhceGZ)).

## Fit do warsztatów NeurIPS 2026

- **NeurReps 2026:** istnieje publiczny oficjalny CFP. Warsztat obejmuje geometrię/topologię reprezentacji i dopuszcza standardowe preprinty bez limitu stron; deadline Findings to 30 sierpnia 2026 podaje OpenReview ([CFP](https://neurreps.org/), [OpenReview Findings](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FNeurReps_Findings)). Fit jest dobry tylko wtedy, gdy wkład wewnętrzny jest centralny i wykracza poza wykres CKA.
- **Transitioning from Pre-Training to Post-Training:** tytuł z oficjalnej listy wygląda tematycznie najlepiej, ale na dzień przeglądu nie znalazłem indeksowanego publicznego CFP/strony warsztatu, więc nie da się jeszcze źródłowo potwierdzić zakresu, formatu ani polityki dual submission. Nie należy opierać harmonogramu na domysłach.
- **Interpretability as a Science:** analogicznie nie znalazłem publicznego CFP. Fit koncepcyjny jest mocny, jeśli eksperyment ma prerejestrację, matched control, holdout i ostrożnie rozróżnia predykcję od przyczynowości, ale trzeba go potwierdzić po publikacji strony.
- NeurIPS sugeruje 29 sierpnia 2026 jako termin contributions i wymaga decyzji do 29 września; daty konkretnego warsztatu mogą się różnić ([oficjalny Call for Workshops 2026](https://neurips.cc/Conferences/2026/CallForWorkshops)).

## Ostateczna rekomendacja

Nie wdrażać szerokiego planu „correct/corrupt/corrected prefixes + activations” jako rzekomo nowego benchmarku. Najbliższa praca już wykonuje większość tej kombinacji.

Wdrożyć jeden mały, rozstrzygający pilot:

> **ProcessBench first-error boundary × (`PRE`, `ERR`) × (`Qwen2.5-Math-7B`, `R1-Distill-Qwen-7B`)**, analizowany jako paired difference-in-differences.

Jeśli interakcja przejdzie gate, dodać ręcznie walidowany `FIX`, aktywacje i zamrożony holdout. Wtedy wkład jest czytelny: nie „LLM-y propagują błędy”, lecz „reasoning distillation zmienia odporność tego samego odziedziczonego modelu na ten sam rzeczywisty błąd, a zmiana ma mierzalny marker wewnętrzny”.
