## Abstract

Point: MRI je drago, dolgo traja in malo prostora pokrije (ni efektivno!), kar 
lahko izboljšamo z mnj meritvami, toda take slike zigubijo kvaliteto. ML 
pristopi poskušajo iz LR slik ustvarit HR slike preko SR. Obstajajo GANs (generative neural networks), mi naj bi bili sposobni generirati podatke glede na neke informacije, s čimer pridobijo nazaj izgubljene podatke.

## Uvod

MRI so pomembni za diagnoze in sledenje razvijajocim boleznim pacientov. 
Dobljene informacije iz slik mehkih tkiv, organov in kosti pomagajo fiziologom, 
da paciente boljše obravnavajo in zdravijo.

Problem je, ker je težko dobiti MRI slike. Pacienti morajo biti na miru ves čas, 
dolgo traja in drago je. MRI slike morajo biti kalibrirane na pacienta. 
Med slikanjem se ustvari več sto posnetih slik (slojev) iz več smeri,
katere se skupaj sestavi v t.i. volume.

Čas v katerem se ustvarijo te slici se razlikuje glede na tehniko slikanja, 
kljub temu traja predolgo, da bi bilo uporabno za krizne situacije, npr. Ko ima 
nekdo stroke ali pa neko resno poškodbo.

Prav tako je čas potreben za slikanje linearno koreliran s kvaliteto slik. 
To je še posebej vidno v 3D slikah, kjer je potrebno veliko več vložiti v 
zelo majhno izboljšanje kvalitete.

Boljše slike proizvedejo boljše naprave, klinike z malo denarja zato uporabljajo 
slabše naprave, ki ustvarjajo slabše slike.

S konceptom SR, lahko skrajšamo čas slikanja in naknadno iz slabih slik dopolnimo
in konstruiranje MRI slike v volume.

## SR

Super resolution - proces ki lahko rekonstruira sliko iz reduced spatial 
resolution
