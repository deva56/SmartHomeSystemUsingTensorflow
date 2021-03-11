"""Program za prepoznavanje govornih komandi koji simulira pametnu kuću ili bilo koji sustav koji vrši nekakve
komande ovisno o govoru. Princip je sličan kao Google-ov sustav na mobitelima gdje korisnik nakon naredbe "Hey google"
može narediti nekakvu radnju te ju sustav izvrši. Ovaj demonstrativni sustav ima 7 naredbi koje prepoznaje te sadrži jednu
govornu komandu koja budi sustav poput Google-ovog "Hey google". Također sadrži još jednu kategoriju koju prepoznaje, a ona sadrži
sve pozadinske šumove i buku koju sustav drukčije ne može kvalificirati."""

"Ovaj projekt je izrađen u svrhu Završnog rada na preddiplomskom studiju Računarstva."

import queue
import threading
import timeit

import librosa as lb
import numpy as np
import python_speech_features as python_speech_features
import scipy.signal
import sounddevice as sd
import soundfile as sf
from gpiozero import LED
from tflite_runtime.interpreter import Interpreter

word_threshold = 0.5
rec_duration = 1
sample_rate = 48000
resample_rate = 8000
num_channels = 1
wakeWordOn = False
stopInference = False

# Definicije LED diodi
napraviKavuLED = LED(14)
prozorLED = LED(18)
vrataLED = LED(4)
tvLED = LED(25)

# Buffer u koji ćemo poslije spremati snimku te iz nje vršiti prepoznavanje
window = np.zeros(int(rec_duration * resample_rate) * 2)


# Funkcija za izvlačenje korisnih karakteristika iz snimke
def extract_features(file, result):
    rez = lb.util.normalize(file, axis=0)
    y = python_speech_features.base.mfcc(rez,
                                         samplerate=resample_rate,
                                         winlen=0.256,
                                         winstep=0.050,
                                         numcep=20,
                                         nfilt=26,
                                         nfft=2048,
                                         preemph=0.0,
                                         ceplifter=22,
                                         appendEnergy=False,
                                         winfunc=np.hanning)
    y = y.transpose()
    mfccs = np.pad(y, ((0, 0), (0, 36 - y.shape[1])), mode='constant')
    result.put(mfccs)


# Funkcija za čitanje oznaka iz Labels.txt datoteke
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


# Podešavanje tensora u modelu
def set_input_tensor(interpreter, wav):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = wav


# Funkcija koja se vrši na modelu i provjerava da li se ulazna snimka
# podudara sa kojom korisnom istreniranom snimkom i vraća rezultat
def classify_wav(interpreter, wav, result, top_k=1):
    """Sortiran niz klasifikacijskih rezultata"""
    set_input_tensor(interpreter, wav)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # Ako je model kvantiziran (tflite - optimizacija)  (uint8 data), onda dekvantiziraj
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    result.put([(i, output[i]) for i in ordered[:top_k]])


def decimate(signal, old_fs, new_fs):
    # Provjera da stvarno smanjujemo sampling rate
    if new_fs > old_fs:
        print("Error: novi sr je viši od starog!")
        return signal, old_fs

    # Zbog metode downsampling možemo to raditi samo po cijelom broju, (int) faktoru
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: downsampling nije int faktor")
        return signal, old_fs

    # vršenje downsamplinga
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal


# Timeout funkcija koja nakon 3 sekunde gasi slušanje naredbe
def timeout():
    global wakeWordOn
    global stopInference
    wakeWordOn = False
    stopInference = False


# Gasi LED diodu za pravljenje kave
def ugasiNapraviKavu():
    napraviKavuLED.off()


# Pušta na zvučnik glasovnu naredbu ovisno koja je naredba prepoznata u govoru
def povratnaPoruka(indeks):
    filename = ''

    if (indeks == 2):
        filename = 'PovratnePoruke/pravimKavu.wav'
    elif (indeks == 3):
        filename = 'PovratnePoruke/otvaramProzor.wav'
    elif (indeks == 4):
        filename = 'PovratnePoruke/otvaramVrata.wav'
    elif (indeks == 5):
        filename = 'PovratnePoruke/gasimTelevizor.wav'
    elif (indeks == 6):
        filename = 'PovratnePoruke/palimTelevizor.wav'
    elif (indeks == 7):
        filename = 'PovratnePoruke/zatvaramProzor.wav'
    elif (indeks == 8):
        filename = 'PovratnePoruke/zatvaramVrata.wav'

    data, fs = sf.read(filename, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()


def inference(rec):
    global wakeWordOn
    global stopInference
    # Provjerava se da li je varijabla stopInference false te ako je znači da se trenutno ne vrši nikakva inferencija
    # i da ju mi možemo vršiti, ako je true izlazimo iz petlje i čekamo na novu iteraciju
    if (stopInference == False):
        # wakeWordOn služi kao mehanizam da sustav zna šta treba u ovome trenutku slušati, ako je wake word true znači
        # da smo u prijašnjoj iteraciji prepoznali da je govornik rekao "hej kućo" i sada narednih 3 sekunde slušamo za druge naredbe
        # ako je wake word false znači da trenutno ne očekujemo nikakvu naredbu te da samo slušamo i čekamo prvo na wake word "hej kućo"
        if (wakeWordOn):
            # Promjena oblika snimke i downsample
            rec = rec.transpose()
            rec = np.squeeze(rec)
            rec = decimate(rec, sample_rate, resample_rate)

            # Prozor za provjeru snimke u bufferu
            window[:len(window) // 2] = window[len(window) // 2:]
            window[len(window) // 2:] = rec

            # Dretva za izvlačenje mfcc podataka za vršenje inference-a
            y_thread_result = queue.Queue()
            y = threading.Thread(target=extract_features, args=(window, y_thread_result))
            y.start()
            y.join()
            inferenceWav1 = y_thread_result.get()
            inferenceWav1 = np.reshape(inferenceWav1, (1, inferenceWav1.shape[0], inferenceWav1.shape[1], 1))

            # Putanja modela i labela
            model = 'ConvModel.tflite'
            labels = 'Labels.txt'

            # Dretva i ostali dio koda za vršenje inference-a
            labels = load_labels(labels)
            interpreter = Interpreter(model)
            interpreter.allocate_tensors()
            z_thread_result = queue.Queue()
            z = threading.Thread(target=classify_wav, args=(interpreter, inferenceWav1, z_thread_result))
            start = timeit.default_timer()
            z.start()
            z.join()
            results = z_thread_result.get()
            elapsed_ms = (timeit.default_timer() - start)
            label_id, prob = results[0]
            print(str(labels[label_id]) + ";%.2f" % (prob) + '%' + ";%.2f" % (elapsed_ms) + 'ms')
            print("\n")
            # Prepoznavanje naredbe nakon što je uključena ključna riječ,
            # ako se komanda prepozna vrši se neka od mapiranih funkcija preko GPIO
            # Vjerojatnost mora biti 60% ili veća da bi se rezultat okarakterizirao kao uspješan
            # Ako nije jednostavno se izlazi iz petlje i čeka na novu snimku
            if (prob >= 0.6):
                # Provjerava se ako prepoznati zvuk nije wake word "hej kućo" ili pozadinski šum
                # U slučaju da jesu u ovome dijelu se oni ignoriraju
                if (label_id != 1 and label_id != 0):
                    if (label_id == 2):
                        stopInference = True
                        napraviKavuLED.on()
                        t = threading.Timer(2, ugasiNapraviKavu)
                        t.start()
                        pp = threading.Thread(target=povratnaPoruka, args=([label_id]))
                        pp.start()
                        pp.join()
                    if (label_id == 3):
                        stopInference = True
                        prozorLED.on()
                        pp = threading.Thread(target=povratnaPoruka, args=([label_id]))
                        pp.start()
                        pp.join()
                    if (label_id == 4):
                        stopInference = True
                        vrataLED.on()
                        pp = threading.Thread(target=povratnaPoruka, args=([label_id]))
                        pp.start()
                        pp.join()
                    if (label_id == 5):
                        stopInference = True
                        tvLED.off()
                        pp = threading.Thread(target=povratnaPoruka, args=([label_id]))
                        pp.start()
                        pp.join()
                    if (label_id == 6):
                        stopInference = True
                        tvLED.on()
                        pp = threading.Thread(target=povratnaPoruka, args=([label_id]))
                        pp.start()
                        pp.join()
                    if (label_id == 7):
                        stopInference = True
                        prozorLED.off()
                        pp = threading.Thread(target=povratnaPoruka, args=([label_id]))
                        pp.start()
                        pp.join()
                    if (label_id == 8):
                        stopInference = True
                        vrataLED.off()
                        pp = threading.Thread(target=povratnaPoruka, args=([label_id]))
                        pp.start()
                        pp.join()

                    print("Prepoznata naredba <--> " + str(labels[label_id]))
                    print("\n")
            else:
                pass
        else:
            # Promjena oblika snimke i downsample
            rec = rec.transpose()
            rec = np.squeeze(rec)
            rec = decimate(rec, sample_rate, resample_rate)

            # Prozor za provjeru snimke u bufferu
            window[:len(window) // 2] = window[len(window) // 2:]
            window[len(window) // 2:] = rec

            # Dretva za izvlačenje mfcc podataka za vršenje inference-a
            y_thread_result = queue.Queue()
            y = threading.Thread(target=extract_features, args=(window, y_thread_result))
            y.start()
            y.join()
            inferenceWav1 = y_thread_result.get()
            inferenceWav1 = np.reshape(inferenceWav1, (1, inferenceWav1.shape[0], inferenceWav1.shape[1], 1))

            # Putanja modela i labela
            model = 'ConvModel.tflite'
            labels = 'Labels.txt'

            # Dretva i ostali dio koda za vršenje inference-a
            labels = load_labels(labels)
            interpreter = Interpreter(model)
            interpreter.allocate_tensors()
            z_thread_result = queue.Queue()
            z = threading.Thread(target=classify_wav, args=(interpreter, inferenceWav1, z_thread_result))
            start = timeit.default_timer()
            z.start()
            z.join()
            results = z_thread_result.get()
            elapsed_ms = (timeit.default_timer() - start)
            label_id, prob = results[0]
            print(str(labels[label_id]) + ";%.2f" % (prob) + '%' + ";%.2f" % (elapsed_ms) + 'ms')
            print("\n")
            # Ako wake word nije bio uključen i sada se prepozna uključuje se timer koji sljedeće
            # 3 sekunde sluša za naredbe i nakon toga se isključuje i opet sluša za wake word
            # Vjerojatnost mora biti bar 60% da se prihvati kao ispravan rezultat prepoznavanja
            if (prob >= 0.6):
                if (label_id == 1):
                    wakeWordOn = True
                    t = threading.Timer(3, timeout)
                    t.start()
            else:
                pass
    else:
        pass


# Funkcija callbacka koja se pozove svaki put kad istekne vrijeme snimanja
def callback(rec, frames, time, status):
    x = threading.Thread(target=inference, args=(rec,))
    x.start()
    x.join()


def main():
    with sd.InputStream(channels=num_channels,
                        samplerate=sample_rate,
                        blocksize=int(sample_rate * rec_duration),
                        callback=callback):
        while True:
            pass


if __name__ == '__main__':
    main()
