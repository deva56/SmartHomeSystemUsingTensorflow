# SmartHomeSystemUsingTensorflow

Sustav pametne kuće koja izvršava naredbe nakon što joj je dana govorna komanda.

Ovaj projekt je izrađen u svrhu Završnog rada na preddiplomskom studiju Računarstva.

Demonstraciju projekta moguće je vidjeti na linku: https://drive.google.com/file/d/1TMhVR1GNAVIwwaFT1dUjOdXsVW-geQO8/view?usp=sharing

Za pokretanje programa potrebno je imati Raspberry Pi uređaj sa dovoljno memorije i GPIO pinovima.
(Projekt je rađen na Rasspberry Pi 4 Model B uređaju.)

Kada su LED diode spojene na brojeve pinova koji se mogu vidjeti u Program.py datoteci u deklaraciji varijabli pri vrhu datoteke skripta 
se može pokrenuti te će automatski učitati potrebne datoteke iz root foldera po potrebi (ConvModel.tflite, Labels.txt i povratne poruke iz 
foldera PovratnePoruke).

Za slušanje glasovnih povratnih poruka potrebno je također imati spojeni zvučnik preko AUX kabla na Raspberry Pi te imati eksternu 
zvučnu karticu za mikrofon pošto Raspberry Pi nema funkcionalnu zvučnu karticu za primanje audio signala.

Kada je program pokrenut na terminalu bi se trebao vidjeti ispis podataka ovisno o tome da li je naredba prepoznata i koja je prepoznata.

Za dodatno istraživanje projekta u root folderu se nalazi Tensorflow model koji je korišten za izradu projekta  pod imenom ConvModel.py

Numpy datoteke koje su korištene za treniranje i validaciju modela se nalaze na sljedećem linku: https://drive.google.com/file/d/1UjloAoWmGPBpu1SCc37SiPCplwN_CiYB/view?usp=sharing


