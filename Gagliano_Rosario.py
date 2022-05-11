"""
prova finale del modulo Analisi datiITS Steve JOBS Corso 15
"""

"""
COGNOME STUDENTE: Gagliano
NOME STUDENTE: Rosario

Salva questo file come COGNOME_NOME.py, e consegnalo a fine compito al docente
(esempio ROSSI_FABIO.py)
"""

"""
FOGLIO ELETTRONICO

task 0
Importare il file studenti.csv fornito dal docente in un foglio Google e
denominare tale foglio con il proprio COGNOME_NOME (esempio: ROSSI_FABIO)
Il file al termine della elaborazione andrà salvato in locale sul
tuo computer di lavoro  e consegnato al docente, NON VA CONDIVISO. 

Il significato dei valori riportati nel file è come segue:

AGE:    1, sotto i venti anni
        2: tra i venti e i venticinque
        3: sopra i venticinque
Gender: 1: maschio
        2: femmina
HS_Type:    1: liceo
            2: istituto tecnico
            3: altro
Work:   1: inoccupato
        2: occupato
Partner: 1: single
         2: con partner
Mother edu: interi con grado di istruzione della madre
Father edu: interi con grado di istruzione del padre
#_Siblings: numero di fratelli/sorelle
Study_hrs: ore di studio media quotidiana
Grade: media di profitto

task 1

Calcolare minimo, massimo, media e mediana della colonna Grade
per i gruppi che seguono:
    a) i maschi con padre con istruzione minima
    b) i maschi con padre con istruzione massima
    c) tutti i maschi
    d) tutte le femmine

task 2
Calcolare la percentuale di maschi single e maschi con partner su tutti i maschi
Calcolare la percentuale di femmine single e femmine con partner su tutte le femmine

task 3
Produrre un istogramma della distribuzione dei valori nella colonna Grade

NOTA BENE:
    Tutte le risposte richieste devono essere leggibili e calcolate nel 
    foglio elettronico che consegnerai
"""

"""
PYTHON

task 0
Importare il file studenti.csv fornito dal docente in un DataFrame di Pandas

"""
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

RD=pd.read_csv("Gagliano_Rosario.csv")
RD.info()
RD.describe()

"""

task 1
Ottenere un diagramma a torta che mostri la provenienza degli studenti da vari
tipi di scuola.

"""

RD["HS_TYPE"].value_counts().plot.pie()

"""

task 2
In unico riquadro ottenere affiancati i box plot relativi ai Grades ottenuti
dai maschi e dei Grades ottenuti dalle femmine

"""
plt.figure()
plt.subplot(1,2,1)
plt.boxplot(RD["GRADE"][RD["GENDER"]==1])
plt.xlabel("GRADE maschi")
plt.ylim([0,8])
plt.subplot(1,2,2)
plt.boxplot(RD["GRADE"][RD["GENDER"]==2])
plt.xlabel("GRADE femmine")
plt.ylim([0,8])

"""

task 3
Quante sono le studentesse con partner che hanno avuto Grade superiore al Grade medio?

"""
grdM=RD["GRADE"].mean()
std1=RD["GENDER"][(RD["PARTNER"]==2)&(RD["GRADE"]>grdM)]
print("le studentesse con partner che hanno avuto",\
      " Grade superiore al Grade medio",len(std1))

grdM=RD["GRADE"].mean()
std2=RD["GENDER"][(RD["PARTNER"]==1)&(RD["GRADE"]>grdM)]
print("le studentesse con partner che hanno avuto",\
      " Grade superiore al Grade medio",len(std2))


totF=len(RD[RD["GENDER"]==2])
fGS=len(RD["GENDER"][(RD["GRADE"]>=grdM)&(RD["GENDER"]==2)])
print("Frequenza studentesse con grado superiore alla media = ",fGS/totF )
print("Percentuale studentesse con grado superiore alla media =",100*fGS/totF)


totM=len(RD[RD["GENDER"]==1])
mGS=len(RD["GENDER"][(RD["GRADE"]>=grdM)&(RD["GENDER"]==1)])
print("Frequenza studenti con grado superiore alla media = ",mGS/totM )
print("Percentuale studenti con grado superiore alla media =",100*mGS/totM)

pd.plotting.scatter_matrix(RD)

correlazioni = RD.corr()
RD["MOTHER_EDU"].corr(RD["FATHER_EDU"])

RD["AGE"].corr(RD["STUDY_HRS"])


"""
Trovare la coppia di colonne che mostra il massimo grado di correlazione positiva
Riferiamoci ad esse come colonne A e B
Trovare la coppia di colonne che mostra il massimo grado di correlazione negativa
Riferiamoci ad esse come colonne C e D


task 5
Produrre in una unica figura (ma in tre subplot separati) il plot che mostri
a) la distribuzione dei valori nelle colonne A e B
b) la distribuzione dei valori nelle colonne C e D
c) la distribuzione dei valori nelle colonne A e B colorando di blu i dati relativi ai maschi 
    e rossi quelli relativi alle femmine
    
"""
plt.figure()
plt.subplot(1,3,1)
plt.plot(RD["MOTHER_EDU"],RD["FATHER_EDU"])

plt.subplot(1,3,2)
plt.plot(RD["AGE"],RD["STUDY_HRS"])

plt.subplot(1,3,3)
plt.plot(RD["MOTHER_EDU"][RD["GENDER"]==1],\
         RD["FATHER_EDU"][RD["GENDER"]==1],"ob",markersize=3)
plt.plot(RD["MOTHER_EDU"][RD["GENDER"]==2],\
             RD["FATHER_EDU"][RD["GENDER"]==2],"or",markersize=3)

"""
    
task 6
Calcolare la relazione di regressione lineare che "preveda" B a partire da A
Produrre plot che illustrino tale relazione

"""
R=LinearRegression()
regressore=RD["MOTHER_EDU"].values.reshape(145,1)
regressa=RD["FATHER_EDU"].values.reshape(145,1)
M=R.fit(regressore, regressa)
stima=M.predict(regressore)
print(M.score(regressore, regressa))

plt.figure()
plt.plot(RD["MOTHER_EDU"],RD["FATHER_EDU"],"ob")
plt.plot(RD["MOTHER_EDU"], stima, color="red")

