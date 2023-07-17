# SmartHomeSystemUsingTensorflow

(CROATIAN)
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

Rad je moguće vidjeti na adresi na sljedećem linku:
https://zir.nsk.hr/islandora/object/mev%3A1256
*********************************************************************************************************************************************
(ENGLISH)
A smart home system that executes commands after being given a voice command.

This project was created for the purpose of the Final Thesis for the undergraduate study of Computer Science.

The demonstration of the project can be seen at the link: https://drive.google.com/file/d/1TMhVR1GNAVIwwaFT1dUjOdXsVW-geQO8/view?usp=sharing

To run the program, you need a Raspberry Pi device with enough memory and GPIO pins. (The project was done on a Raspberry Pi 4 Model B device.)

When the LEDs are connected to the pin numbers that can be seen in the Program.py file in the variable declaration at the top of the file the script can be run and will automatically load the necessary files from the root folder as needed (ConvModel.tflite, Labels.txt and return messages from folder PovratnePoruke).

To listen to voice feedback messages, it is also necessary to have a speaker connected via an AUX cable to the Raspberry Pi and to have an external sound card for the microphone, since the Raspberry Pi does not have a functional sound card for receiving audio signals.

When the program is run on the terminal you should see data printed depending on whether the command is recognized and which one is recognized.

For additional project research, the root folder contains the Tensorflow model that was used to create the project under the name ConvModel.py

Numpy files that were used for training and validating the model can be found at the following link: https://drive.google.com/file/d/1UjloAoWmGPBpu1SCc37SiPCplwN_CiYB/view?usp=sharing

The work can be seen at the following link:
https://zir.nsk.hr/islandora/object/mev%3A1256
