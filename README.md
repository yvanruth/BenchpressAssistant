# Bench Press Tracker

Welkom bij de Bench Press Tracker! Deze applicatie helpt je bij het analyseren van je bankdrukprestaties door reps te tellen, bewegingen te volgen en diverse statistieken per herhaling te berekenen.

## Inhoudsopgave

*   [Functionaliteiten](#functionaliteiten)
*   [Installatie & Opstarten](#installatie--opstarten)
*   [Gebruik van de Applicatie](#gebruik-van-de-applicatie)
    *   [Interface Overzicht](#interface-overzicht)
    *   [Modi: Live & Debug](#modi-live--debug)
    *   [ROI (Region of Interest) Instellen](#roi-region-of-interest-instellen)
    *   [Kalibratie voor Vermogensmeting](#kalibratie-voor-vermogensmeting)
    *   [Bedieningstoetsen](#bedieningstoetsen)
*   [Statistieken Uitleg (`session_stats.json`)](#statistieken-uitleg-session_statsjson)
*   [Technische Details](#technische-details)
*   [Toekomstige Verbeteringen](#toekomstige-verbeteringen)

## Functionaliteiten

*   **Real-time Rep Tellen**: Detecteert automatisch voltooide herhalingen.
*   **Bewegingsanalyse**: Volgt de verticale beweging van de halterstang.
*   **Snelheidsmeting**: Berekent gemiddelde en maximale snelheden voor de neergaande (excentrische) en opwaartse (concentrische) fase van de lift.
*   **Vermogensschatting**: Schat het gemiddelde en piekvermogen (in Watt) tijdens de opwaartse fase.
*   **Hellingdetectie**: Analyseert de hoek van de stang om scheef gaan te detecteren.
*   **Statistieken Opslag**: Slaat gedetailleerde statistieken per rep op in een `session_stats.json` bestand voor latere analyse.
*   **Visuele Feedback**: Toont de status van de rep, beweging, en real-time piekvermogen op het scherm.
*   **Video Afspeel Controls (Debug Mode)**: Pauzeren, versnellen/vertragen, en navigeren in opgenomen video's.
*   **Aanpasbaar Gewicht**: Stel het totale gewicht op de stang in.

## Installatie & Opstarten

1.  **Vereisten**:
    *   Python 3.x
    *   OpenCV (`cv2`)
    *   NumPy
    *   Pygame (voor geluidseffecten)
    *   SciPy (voor het schrijven van .wav bestanden, indien `beep.wav` of `countdown.wav` niet bestaan)
    *   Een webcam (voor live modus) of een videobestand (voor debug modus).

    Installeer de benodigde Python packages via pip:
   ```bash
    pip install opencv-python numpy pygame scipy
    ```

2.  **Bestanden**:
    *   `bench_press_tracker.py`: Het hoofdbestand van de applicatie.
    *   `beep.wav`, `countdown.wav`: Geluidsbestanden (worden automatisch aangemaakt indien niet aanwezig).
    *   `roi_settings.json`: Slaat de ROI en kalibratie-instellingen op (wordt automatisch aangemaakt/bijgewerkt).
    *   `session_stats.json`: Hierin worden de statistieken van je trainingssessie opgeslagen (wordt aangemaakt/bijgewerkt na elke sessie).
    *   Optioneel: een `movies` map met daarin je testvideo (bijv. `test_bench.mp4` zoals in de code gespecificeerd voor debug mode).

3.  **Opstarten**:
    Voer het script uit vanuit je terminal:
   ```bash
   python bench_press_tracker.py
   ```
    De applicatie start standaard in "debug mode" met een testvideo (indien correct pad in de code) of in "live mode" met je webcam.

## Gebruik van de Applicatie

### Interface Overzicht

*   **Hoofdvenster ("Main")**: Toont het videobeeld van de camera of het afgespeelde videobestand. Hierop wordt de meeste visuele feedback getoond.
*   **Modus Indicator (linksboven)**: Geeft aan of je in `LIVE MODE` of `DEBUG MODE` bent.
*   **Rep Teller (linksboven)**: Toont het huidige aantal getelde reps.
*   **Timer (middenboven, alleen debug mode)**: Toont de verstreken tijd in de video.
*   **Gewichtsdisplay (rechtsboven)**: Toont het huidige ingestelde gewicht en de toetsen om dit aan te passen.
*   **ROI Rechthoek (oranje)**: Indien ingesteld, toont dit het gebied waarbinnen de stang gedetecteerd wordt.
*   **Gedetecteerde Stang (groene lijn)**: Visualiseert de gedetecteerde positie en hoek van de stang binnen de ROI.
*   **Hoek Indicator (middenonder)**: Toont de huidige gestabiliseerde hoek van de stang.
*   **Rep State & Movement Info (midden-boven op video)**: Toont de huidige `Movement State` (UPWARD, DOWNWARD, STABLE) en `Rep State` (READY, DESCENDING, BOTTOM, ASCENDING, COMPLETED).
*   **Rep Statistieken Paneel (rechtsonder)**: Toont live en voltooide statistieken voor de huidige rep, inclusief real-time piekvermogen.
*   **Optioneel: Bar Detection Venster**: Een apart venster dat de interne stappen van de stangdetectie (HSV masker) laat zien (aan/uit te zetten met 'd').
*   **Optioneel: HSV Controls Venster**: Een apart venster met schuifbalken om de HSV drempelwaarden voor stangdetectie aan te passen (aan/uit te zetten met 'h').

### Modi: Live & Debug

*   **Live Mode**: Gebruikt je standaard webcam als videobron.
*   **Debug Mode**: Speelt een videobestand af (standaard `movies/test_bench.mp4`). In deze modus heb je extra controls zoals pauzeren, snelheid aanpassen, en vooruit/achteruit springen.
*   Wisselen van modus: Druk op **'m'**.

### ROI (Region of Interest) Instellen

De ROI is het gebied waarbinnen de applicatie zoekt naar de halterstang. Dit is essentieel voor accurate detectie.

1.  **ROI Selecteren**:
    *   Als er geen ROI is ingesteld, zie je een instructie op het scherm.
    *   Klik met je linkermuisknop op een hoek van het gewenste gebied en sleep de muis naar de tegenoverliggende hoek. Laat de muisknop los.
    *   De ROI wordt nu als een oranje rechthoek getoond.
2.  **Opslaan ROI**: De geselecteerde ROI wordt automatisch opgeslagen in `roi_settings.json` wanneer je een geldige ROI hebt getekend. Bij een volgende start van de applicatie wordt deze ROI automatisch geladen.
3.  **Resetten ROI**: Druk op **'r'** (kleine letter) om de huidige ROI te verwijderen (ook uit het opslagbestand). Je kunt dan een nieuwe tekenen.

**Tips voor een goede ROI**:
*   Zorg dat de volledige bewegingsbaan van de stang (van hoogste tot laagste punt) binnen de ROI valt.
*   Maak de ROI niet onnodig groot; focus op het gebied waar de stang beweegt.
*   Zorg voor voldoende contrast tussen de stang en de achtergrond binnen de ROI.

### Kalibratie voor Vermogensmeting

Om het vermogen (Wattage) correct te kunnen schatten, moet de applicatie weten hoeveel pixels in het videobeeld overeenkomen met een werkelijke afstand (bijv. een meter). Dit gebeurt door de lengte van de halterstang te kalibreren.

**Kalibratie Stappen Flow:**

1.  **Activeer Meetmodus**: Druk op de **'b'** toets. De instructies voor het meten verschijnen.
2.  **Meet de Stang**:
    *   Klik met je linkermuisknop op één uiteinde van de zichtbare halterstang in het videobeeld.
    *   Houd de muisknop ingedrukt en sleep de lijn naar het andere uiteinde van de halterstang.
    *   Laat de muisknop los.
3.  **Resultaat**:
    *   In de console zie je nu de gemeten lengte in pixels en de berekende `pixels_per_meter` waarde. Bijvoorbeeld:
        ```
        Measured bar length: 350.5 pixels
        Calibration: 159.32 pixels per meter (bar length: 2.2m)
        ```
    *   De `pixels_per_meter` waarde wordt nu gebruikt voor vermogensberekeningen.
    *   Als er al een ROI was gedefinieerd, wordt deze kalibratiewaarde (samen met de ROI) **automatisch opgeslagen** in `roi_settings.json`. Bij een volgende start wordt deze kalibratie dan geladen.
4.  **Verlaat Meetmodus**: Druk nogmaals op **'b'** om de meetmodus te verlaten.

**Hoe de applicatie de kalibratie gebruikt:**
*   De applicatie kent de werkelijke lengte van een standaard Olympische halterstang (`ACTUAL_BAR_LENGTH_METERS = 2.2` meter, dit is instelbaar in de code).
*   Door de lengte van deze bekende stang in pixels op te meten, berekent het `pixels_per_meter = gemeten_pixels / werkelijke_lengte_meters`.
*   Deze factor wordt gebruikt om snelheden van pixels/seconde om te rekenen naar meters/seconde.
*   Vermogen (Watt) = Kracht (Newton) \* Snelheid (m/s)
*   Kracht = Totaal Gewicht (kg) \* `GRAVITY` (9.81 m/s²)

**Belangrijk**:
*   Voer de kalibratie uit *voordat* je reps doet waarvan je het vermogen wilt meten.
*   Doe de kalibratie opnieuw als de camera-afstand of -hoek significant verandert.
*   Als je een stang gebruikt die geen 2.2 meter lang is, pas dan de `self.ACTUAL_BAR_LENGTH_METERS` waarde aan in de `__init__` methode van het script voor accurate resultaten.

### Bedieningstoetsen

*   **'q'**: Applicatie afsluiten.
*   **'m'**: Wisselen tussen Live en Debug modus.
*   **'s'**: Start 3-seconden countdown (alleen in Live modus, als tracking nog niet actief is).
*   **'+'/ '='**: Gewicht verhogen met 10kg.
*   **'-'/ '\_'**: Gewicht verlagen met 10kg.
*   **'p'**: Gewicht verhogen met 2.5kg.
*   **'o'**: Gewicht verlagen met 2.5kg.
*   **Spatiebalk (in Debug Mode)**: Video pauzeren/hervatten.
*   **'[' (linker vierkante haak, in Debug Mode)**: Video vertragen.
*   **']' (rechter vierkante haak, in Debug Mode)**: Video versnellen.
*   **Linker Pijl (in Debug Mode)**: 0.5 seconde terugspoelen.
*   **Rechter Pijl (in Debug Mode)**: 0.5 seconde vooruitspoelen.
*   **'r' (kleine letter)**: Huidige ROI resetten (wordt ook uit `roi_settings.json` verwijderd).
*   **'R' (hoofdletter)**: Volledige reset van alle parameters (rep teller, state machine, etc.), handig voor een schone start.
*   **'b'**: Wisselen naar/van "bar measurement mode" voor kalibratie.
*   **'h'**: Tonen/verbergen van het HSV Controls venster.
*   **'d'**: Tonen/verbergen van het Bar Detection (debug masker) venster.
*   **Cijfers 1-6 & Shift+Cijfers (!-^)**: HSV drempelwaarden fijnafstellen (als HSV Controls venster actief is).

## Statistieken Uitleg (`session_stats.json`)

Na elke sessie (wanneer je de applicatie met 'q' afsluit) wordt de `rep_history` opgeslagen in `session_stats.json`. Dit bestand bevat een lijst, waarbij elk item in de lijst een dictionary is met de statistieken van één succesvol getelde herhaling.

Hier is een uitleg van de belangrijkste velden per rep:

*   `"start_time"`: Tijdstip (YYYY-MM-DD HH:MM:SS.mmm) waarop de neergaande beweging begon.
*   `"bottom_time"`: Tijdstip waarop de stang het laagste punt (bodem) bereikte.
*   `"end_time"`: Tijdstip waarop de opwaartse beweging voltooid was.
*   `"pause_duration"`: Duur (in seconden) dat de stang op het laagste punt (borst) pauzeerde.
*   `"descent_distance"`: Totale verticale afstand (in pixels) van de neergaande beweging.
*   `"ascent_distance"`: Totale verticale afstand (in pixels) van de opwaartse beweging.
*   `"max_descent_speed"`: Maximale snelheid (in pixels/seconde) tijdens de neergaande beweging.
*   `"max_ascent_speed"`: Maximale snelheid (in pixels/seconde) tijdens de opwaartse beweging.
*   `"avg_descent_speed"`: Gemiddelde snelheid (in pixels/seconde) tijdens de neergaande beweging.
*   `"avg_ascent_speed"`: Gemiddelde snelheid (in pixels/seconde) tijdens de opwaartse beweging.
*   `"start_y"`: Verticale Y-coördinaat (in pixels) van de stang bij de start van de rep.
*   `"bottom_y"`: Verticale Y-coördinaat (in pixels) van de stang op het laagste punt.
*   `"end_y"`: Verticale Y-coördinaat (in pixels) van de stang aan het einde van de rep.
*   `"_rep_counted_this_cycle"`: Een interne vlag (meestal `true`) die aangeeft dat deze rep geteld is.
*   `"peak_power_ascent"`: Geschat piekvermogen (in Watt) geleverd tijdens de opwaartse fase. *Vereist correcte kalibratie en gewichtsinstelling.*
*   `"avg_power_ascent"`: Geschat gemiddeld vermogen (in Watt) geleverd tijdens de opwaartse fase. *Vereist correcte kalibratie en gewichtsinstelling.*
*   `"avg_angle_descent"`: Gemiddelde hoek (in graden) van de stang tijdens de neergaande fase. (0 = horizontaal, positief = rechterkant lager, negatief = linkerkant lager).
*   `"std_angle_descent"`: Standaarddeviatie van de hoek tijdens de neergaande fase (een maat voor hoe stabiel de hoek was).
*   `"max_abs_angle_descent"`: Maximale absolute afwijking van een perfect horizontale stang (0 graden) tijdens de neergaande fase.
*   `"avg_angle_ascent"`: Gemiddelde hoek (in graden) van de stang tijdens de opwaartse fase.
*   `"std_angle_ascent"`: Standaarddeviatie van de hoek tijdens de opwaartse fase.
*   `"max_abs_angle_ascent"`: Maximale absolute afwijking van een perfect horizontale stang (0 graden) tijdens de opwaartse fase.

**Let op**: `pixels_per_meter` zelf wordt opgeslagen in `roi_settings.json`, niet in elke rep van `session_stats.json`. De vermogenswaarden in `session_stats.json` zijn berekend *met de kalibratie die actief was op het moment van die rep*.

## Technische Details

*   **Stangdetectie**: Gebruikt OpenCV's HSV kleurbereik-filtering, gevolgd door morfologische operaties (openen, dilateren) en contour-analyse om de halterstang te isoleren.
*   **Hoekberekening**: `cv2.minAreaRect` of momenten worden gebruikt om de oriëntatie van de gedetecteerde stangcontour te bepalen.
*   **State Machine**: Een state machine (`READY`, `DESCENDING`, `BOTTOM`, `ASCENDING`, `COMPLETED`) beheert de logica voor het detecteren en valideren van een rep.
*   **Snelheidsberekening**: Gebaseerd op de verandering in Y-positie over tijd (aantal frames / FPS). Een klein buffer wordt gebruikt voor het middelen van de snelheid.

## Toekomstige Verbeteringen

*   Grafische weergave van statistieken (bijv. snelheidscurves, vermogensgrafieken).
*   Meer geavanceerde bar path tracking (horizontale afwijking).
*   Gebruikersinterface voor het instellen van `ACTUAL_BAR_LENGTH_METERS`.
*   Detectie en feedback op "grinding" reps (zeer langzame opwaartse fase).
*   Automatische set detectie.