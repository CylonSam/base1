# Informações sobre as bases de dados

Bases utilizadas:

- [St. Vincent's University Hospital / University College Dublin Sleep Apnea Database](https://physionet.org/content/ucddb/1.0.0/)
- [You Snooze You Win: The PhysioNet/Computing in Cardiology Challenge 2018](https://physionet.org/content/challenge-2018/1.0.0/)
- [MIT-BIH Polysomnographic Database](https://physionet.org/content/slpdb/1.0.0/)

## DB #1 - St. Vincent's University Hospital / University College Dublin Sleep Apnea Database

Ano: 2011

Quantidade de pacientes: 25 (21H e 4M)

Sinais:

- **EEG (C3-A2)**
- **EEG (C4-A1)**
- Left EOG
- Right EOG
- Submental EMG
- ECG (modified lead V2)
- Oro-nasal airflow (thermistor)
- Ribcage movements
- Abdomen movements (uncalibrated strain gauges)
- Oxygen saturation (finger pulse oximeter)
- Snoring (tracheal microphone)
- Body position

Anotações:

- Hypopnea (Central, Mixed e Obstructive)
- Apnea (Central, Mixed e Obstructive)

## DB #2 - You Snooze You Win: The PhysioNet/Computing in Cardiology Challenge 2018

Ano: 2018

Quantidade de pacientes: 1983 (1290H e 693M)

Sinais:

- **EEG (C3-M2)**
- **EEG (C4-M1)**
- **EEG (F3-M2)**
- **EEG (F4-M1)**
- **EEG (O1-M2)**
- **EEG (O2-M1)**
- Left EOG (E1-M2)
- EMG (Chin1-Chin2)
- ABD
- CHEST
- AIRFLOW
- SaO2
- ECG

Anotações:

- Hypopnea (Central, Mixed e Obstructive)
- Apnea (Central, Mixed e Obstructive)
- Outros

## DB #3 - MIT-BIH Polysomnographic Database

Ano: 1999

Quantidade de pacientes: 16 (16H)

Sinais:
- AHI
- ECG
- BP
- **EEG** (sinal diferente para diferentes pacientes)
- Resp
- Resp
- EOG
- EMG
- SV
- SO2

Anotações:
- H	Hypopnea
-	HA	Hypopnea with arousal
-	OA	Obstructive apnea
-	X	Obstructive apnea with arousal
-	CA	Central apnea
-	CAA	Central apnea with arousal
-	L	Leg movements
-	LA	Leg movements with arousal
-	A	Unspecified arousal
-	MT	Movement time

Each annotation in the `.st' files applies to the thirty seconds of the record that follow the annotation.