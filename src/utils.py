from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam

LR_SHAPE = (80, 80, 3)
HR_SHAPE = (320, 320, 3)

def adam_opt(lr=1e-4, b1=0.9, b2=0.999, decay_steps=1e4, decay_rate=0.5):
    lr_schedule = ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True  # da se lr spremeni na vsakih 10k korakov
    )
    return Adam(learning_rate=lr_schedule, beta_1=b1, beta_2=b2, weight_decay=False)