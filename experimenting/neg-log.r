# Določimo zaporedje vrednosti x od zelo majhnega števila do 1.
# Uporabimo majhno število namesto 0, ker log(0) ni definiran.
x_vrednosti <- seq(0.001, 1, by = 0.001)

# Izračunamo vrednosti y po funkciji f(x) = -log(x).
y_vrednosti <- -log(x_vrednosti)

# Odpremo grafično napravo PNG za shranjevanje slike.
# Slika se bo shranila v datoteko "graf_log_funkcije.png".
png("graf_log_funkcije.png", width = 800, height = 600)

# Izrišemo graf.
# 'type = "l"' pomeni, da želimo črtni graf.
# 'col = "blue"' nastavi barvo črte na modro.
# 'lwd = 2' nastavi debelino črte.
# 'main', 'xlab', in 'ylab' so naslovi grafa in osi.
plot(x_vrednosti, y_vrednosti,
  type = "l", col = "blue", lwd = 2,
  main = "Graf funkcije f(x) = -log(x)",
  xlab = "x",
  ylab = "f(x)",
  ylim = c(0, max(y_vrednosti))
) # Nastavimo meje y-osi od 0 do največje vrednosti.

# Zapremo grafično napravo, kar shrani datoteko.
dev.off()

# Izpišemo sporočilo v konzolo, da je bila slika shranjena.
print("Graf je bil uspešno shranjen v datoteko 'graf_log_funkcije.png'")
