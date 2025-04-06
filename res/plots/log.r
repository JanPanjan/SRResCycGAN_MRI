x <- seq(0.01, 1, 0.01)
y <- sapply(x, log)

library(ggplot2)

ggplot() +
  aes(x=x,y=y) +
  geom_rect(
    xmin=1.0,
    xmax=max(x),
    ymin=min(y),
    ymax=max(y),
    fill="orangered",
    alpha=0.002
  ) +
  geom_rect(
    xmin=min(x),
    xmax=1,
    ymin=min(y),
    ymax=max(y),
    fill="seagreen1",
    alpha=0.002
  ) +
  geom_line() +
  geom_line(aes(x=x,y=0), col = "blue") +
  theme_minimal() +
  labs(
    x = "estimated prediction",
    y = "scaled prediction",
    title = "Logarithmic scaling of models estimated prediction"
  ) 

