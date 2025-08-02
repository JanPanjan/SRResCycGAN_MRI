x_vrednosti <- seq(0.001, 1, by = 0.001)
y_vrednosti <- -log(x_vrednosti)

png("graf_log_funkcije.png", width = 800, height = 600)

plot(x_vrednosti, y_vrednosti,
  type = "l", col = "blue", lwd = 2,
  main = "Graf funkcije f(x) = -log(x)",
  xlab = "x",
  ylab = "f(x)",
  ylim = c(0, max(y_vrednosti))
)

dev.off()

print("Graf je bil uspešno shranjen v datoteko 'graf_log_funkcije.png'")
