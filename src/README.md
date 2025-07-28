# SRResCycGAN

## Structure

### LR -> HR Generator (G_HR)

- Super Resolution generator
- Input je LR image (y)
- Output je fake HR image (x hat)
- SRResDNet struktura; upscale -> refine pristop
- najprej poveča sliko nato odpravi artefakte in pomanjkljivosti
- ResDNet opravi refinement (residual blocks)

### HR -> LR Generator (G_LR)

- Degradation generator
- Input je HR image
- Output je LR image
- Convolution -> ResNet -> Convolution
- downsampla sliko basically

### HR Discriminator (D_HR)

- Input je HR image
- Output je float prediction (real img / sharp HR img / fake img)
- Standardni CNN (PatchGan, VGG19 perhaps?)

### LR Discriminator (D_LR)

- Input je LR image
- Output je float (real img / naturally degraded img / fake img)
- Podobno kot HR discriminator, le za manjši input

## Training "cycle"

### Forward cycle (LR -> HR; -> LR)

- **CILJ:** naredi dober HR image

1. input LR image -> G_HR -> HR image
2. HR image -> D_HR (proba ugotovit ali je fake ali ne) -> adversarial loss
3. HR image -> G_LR -> LR image
4. LR vs HR image -> cycle-consistency loss

### Backward cycle (HR -> LR; -> HR)

- **CILJ:** learn the degradation features

1. HR image -> G_LR -> LR image
2. LR image -> D_LR -> adversarial loss
3. LR image -> G_HR -> HR image
4. HR vs LR image -> cycle-consistency loss

## Losses

### Perceptual loss (p_loss)

- uporabiš pre-trained image classifier, VGG19
- feedaš fake in real HR sliki
- gledaš aktivacije nekega vmesnega sloja (feature maps primerjaš)
- loss = L2 ( VGG(G_HR(real LR)) - VGG(real HR) )

- NOTE: real HR mora bit ground truth za fake HR image
- [poglej ta primer](https://www.tensorflow.org/tutorials/generative/style_transfer#define_content_and_style_representations)

### Adversarial loss (adv_loss)

- BinaryCrossEntropy ker se obnese boljše kot MSE (preveri paper kaj pravijo o tem)
- **D_HR**: BCE(D_HR(real_hr), 1) + BCE(D_HR(G_HR(real_lr)), 0)
- **G_HR**: BCE(D_HR(G_HR(real_lr)), 1)
- **D_LR**: BCE(D_LR(real_lr), 1) + BCE(D_LR(G_LR(real_hr)), 0)
- **G_LR**: BCE(D_LR(G_LR(real_hr)), 1)

### Content loss

- L1 loss, Mean Absolute Error, namesto L2, saj je L2 podvržen oversmoothing
- L1 ( G_HR(real LR) - real HR )

### Total variation loss (tv_loss)

- _regularizer_, ki pomaga s smoothness slike
- ker so MRI slike podvržene nizkem PSNR (peak signal to noise ratio), je treba slike preprocessat
  in poskrbet za robustno ravnanje modela s šumom
- kaznuje ostre, šumne spremembe med bližnjimi pixli v fake HR sliki
- uporablja vertikalne in horizontalne gradiente, ki primerno zaznajo šum na slikah
- `tf.image.total_variation(<image>).numpy()` = delta (ali to vrne loss, ki je definiran spodaj?)
- delta_h za horizontralne edges in delta_v za vertikalne edges
- loss = L1(delta_h(G_HR(real LR)) - delta_h(real HR)) + L1(delta_v(G_HR(real LR)) - delta_v(real HR))

### Cyclic loss (cyc_loss)

- **IDEJA:** če preslikaš sliko iz A v B domeno in nazaj v A, naj bi dobil enako sliko.
- loss forward = L1( G_LR(G_HR(real LR)) - real LR )
- loss backward = L1( G_HR(G_LR(real HR)) - real HR )
- total loss = loss forward + loss backward

### Identity loss (id_loss)

- optional, ampak priporočeno
- da ohranja barvo in kontrast slike (morda kontrast... nevem če bo razlika)
- generatorju daš sliko ki je že v tarčni domeni

- real HR -> G_HR -> real HR
  - idloss HR = L1 loss (real HR) (G_HR(real HR))

- real LR -> G_LR -> real LR
  - idloss LR = L1 loss (real LR) (G_LR(real LR))

### Total loss

- total_loss_D =
  loss D_HR +
  loss D_LR

- total_loss_G =
  perceptual_loss +
  adv_loss_G +
  tv_loss +
  (content_loss x lambda_content) +
  (cycle_loss x lambda_cyc)

- izgubi cycle in conent sta veliko bolj pomembni in doprineseta več k izgubi zaradi lambda parametrov

## Training

- Adam optimizer, b1=0.9, b2=0.999 (default vrednosti)
- Learning rate najprej 1e-4 (0.0001), zmanjšuje se za faktor 2 vsakih 10k iteracij (pomnoži z 0.5)

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.5,
    staircase=True) # staircase=True zagotovi, da se lr spremeni natančno na vsakih 10k korakov

adam_optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)
```

- NOTE: Trainable projection layer sem izpustil ker je prekompleksno za implementirat
