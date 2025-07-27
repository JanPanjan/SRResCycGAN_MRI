"""
Pristop 1: `strides=1, padding='valid'` (Implicitno zmanjševanje)

*   **Kako deluje?** Jedro (npr. velikosti 3x3) se premika po sliki s korakom 1. Ker nima oblazinjenja
(`padding='valid'`), se lahko premika samo po območjih, kjer se v celoti prekriva s sliko. Posledično
se izhodna slika "skrči" za `kernel_size - 1` pikslov v vsaki dimenziji.
*   Če imate vhodno sliko 8x8 in jedro 3x3, bo izhodna slika velikosti 6x6.
*   Če imate vhodno sliko 8x8 in jedro 5x5, bo izhodna slika velikosti 4x4.

*   **Kakšen je namen?** Glavni namen tukaj **ni** zmanjšanje resolucije, ampak **ekstrakcija značilnosti
(feature extraction)**. Zmanjšanje je le stranski učinek. Ta pristop je bil pogost v zgodnjih konvolucijskih
mrežah za klasifikacijo, kjer se je dimenzija postopoma zmanjševala skozi več slojev.

*   **Pomanjkljivost:** Zmanjšanje ni "lepo". Ni nujno, da se slika zmanjša za celoštevilski faktor (npr.
polovico, četrtino). Rezultat je odvisen od velikosti jedra, kar ni praktično za arhitekture, kot je U-Net,
kjer potrebujemo simetrijo med kodirnikom in dekodnikom.

---

### Pristop 2: `strides=2, padding='same'` (Eksplicitno zmanjševanje)

*   **Kako deluje?** Jedro se premika po sliki s korakom 2, torej **preskoči vsak drugi piksél**. Parameter
`padding='same'` pa je ključen, ker poskrbi za dodajanje ravno prav ničel na robove, da je izhodna velikost
natančno `vhodna_velikost / korak`.
*   Če imate vhodno sliko 8x8 in `strides=2`, bo izhodna slika velikosti 4x4.
*   Če imate vhodno sliko 320x320 in `strides=2`, bo izhodna slika velikosti 160x160.

*   **Kakšen je namen?** Glavni in edini namen te kombinacije je **kontrolirano in predvidljivo zmanjšanje
resolucije slike (downsampling)** za določen faktor.

*   **Prednost:** To je standarden, čist in učinkovit način za gradnjo kodirnikov (downsamplerjev) v modernih
arhitekturah. Točno vemo, kakšna bo izhodna velikost, kar je nujno za preskočne povezave (skip connections)
in ujemanje dimenzij v dekodirniku.
"""

import tensorflow as tf
from keras.layers import Conv2D

# Vhodni tenzor: 1 slika, velikosti 4x4, z 1 kanalom
# Vrednosti so samo zato, da vidimo rezultat, pomembna je oblika (shape).
input_tensor = tf.constant([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
], dtype=tf.float32)

input_tensor.shape

# Obliko moramo razširiti, da ustreza formatu Kerasa: (batch, visina, sirina, kanali)
input_tensor = tf.reshape(input_tensor, [1, 4, 4, 1])

print(f"Originalna oblika vhoda: {input_tensor.shape}\n")

# --- PRIMER 1: KORAK (STRIDES) = 1 ---

# 1a. Uporaba padding='same'
print("--- Primer 1a: padding='same', strides=1 ---")
conv_same = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')
output_same = conv_same(input_tensor)
print(f"Izhodna oblika: {output_same.shape}")
print("Zaključek: Velikost je ostala ENAKA (4x4 -> 4x4).\n")


# 1b. Uporaba padding='valid'
print("--- Primer 1b: padding='valid', strides=1 ---")
conv_valid = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid')
output_valid = conv_valid(input_tensor)
print(f"Izhodna oblika: {output_valid.shape}")
print("Zaključek: Velikost se je ZMANJŠALA (4x4 -> 2x2).\n")


# --- PRIMER 2: KORAK (STRIDES) = 2 (Downsampling) ---

# 2a. Uporaba padding='same'
print("--- Primer 2a: padding='same', strides=2 ---")
conv_same_stride2 = Conv2D(filters=1, kernel_size=3, strides=2, padding='same')
output_same_stride2 = conv_same_stride2(input_tensor)
print(f"Izhodna oblika: {output_same_stride2.shape}")
print("Zaključek: Velikost se je natančno RAZPOLOVILA (4x4 -> 2x2).\n")

# 2b. Uporaba padding='valid'
print("--- Primer 2b: padding='valid', strides=2 ---")
conv_valid_stride2 = Conv2D(filters=1, kernel_size=3, strides=2, padding='valid')
output_valid_stride2 = conv_valid_stride2(input_tensor)
print(f"Izhodna oblika: {output_valid_stride2.shape}")
print("Zaključek: Velikost se je zmanjšala, ampak na 'čudno' vrednost (4x4 -> 1x1).\n")
