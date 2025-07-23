# SRResCycGAN
## Structure
### LR -> HR Generator (G_HR)
* Super Resolution generator
* Input je LR image (y)
* Output je fake HR image (x hat)
* SRResDNet struktura; upscale -> refine pristop
* najprej poveča sliko nato odpravi artefakte in pomanjkljivosti
* ResDNet opravi refinement (residual blocks)

### HR -> LR Generator (G_LR)
* Degradation generator
* Input je HR image
* Output je LR image
* Convolution -> ResNet -> Convolution
* downsampla sliko basically

### HR Discriminator (D_HR)
* Input je HR image
* Output je float prediction (real img / sharp HR img / fake img)
* Standardni CNN (PatchGan perhaps?)

### LR Discriminator (D_LR)
* Input je LR image
* Output je float (real img / naturally degraded img / fake img)
* Podobno kot HR discriminator, le za manjši input

## Training "cycle"
### Forward cycle (LR -> HR; -> LR)
* **CILJ:** naredi dober HR image
1. input LR image -> G_HR -> HR image
3. HR image -> D_HR (proba ugotovit ali je fake ali ne) -> adversarial loss
5. HR image -> G_LR -> LR image
6. LR vs HR image -> cycle-consistency loss

### Backward cycle (HR -> LR; -> HR)
* **CILJ:** learn the degradation features
1. HR image -> G_LR -> LR image
2. LR image -> D_LR -> adversarial loss
3. LR image -> G_HR -> HR image
4. HR vs LR image -> cycle-consistency loss

## Losses
### Perceptual loss
* uporabiš pre-trained image classifier, VGG19
* feedaš fake in real HR sliki
* gledaš aktivacije nekega vmesnega sloja (feature maps primerjaš)
* loss = L2 loss med VGG(fake HR) VGG(real HR) outputs al nekaj
* NOTE: real HR mora bit ground truth za fake HR image
* [poglej ta primer](https://www.tensorflow.org/tutorials/generative/style_transfer#define_content_and_style_representations)

### Adversarial loss
* uporabi BCE ker je boljše kot MSE (preveri paper kaj pravijo o tem)
* **D_HR loss:**
  1. D_HR output za real HR, label 1
  2. D_HR output za fake HR, label 0
  3. total loss = average (1) in (2)
* **G_HR loss:**
  * D_HR output za fake HR, label 1

### Cycle-consistency loss
* **IDEJA:** če preslikaš sliko iz A v B domeno in nazaj v A, naj bi dobil enako sliko
* **forward cycle**:
  1. real LR -> G_HR -> fake HR
  2. fake HR -> G_LR -> fake LR
  3. loss = L1 loss (real LR) (fake LR)
    * MAE (L1) > MSE (L2), ker naredi manj blurry results
* **backward cycle**:
  1. real HR -> G_LR -> fake LR
  2. fake LR -> G_HR -> fake HR
  3. loss = L1 loss (real HR) (fake HR)

### Total variation loss
* _regularizer_, ki pomaga s smoothness slike
* ker so MRI slike podvržene nizkem PSNR (peak signal to noise ratio), je treba slike preprocessat in poskrbet za robustno ravnanje modela s šumom
* kaznuje ostre, šumne spremembe med bližnjimi pixli v fake HR sliki
* uporablja vertikalne in horizontalne gradiente, ki primerno zaznajo šum na slikah
* `tf.image.total_variation(<image>).numpy()` je ta loss

### Identity loss
* optional, ampak priporočeno
* da ohranja barvo in kontrast slike (morda kontrast... nevem če bo razlika)
* generatorju daš sliko ki je že v tarčni domeni
* real HR -> G_HR -> real HR
  * idloss HR = L1 loss (real HR) (G_HR(real HR))
* real LR -> G_LR -> real LR
  * idloss LR = L1 loss (real LR) (G_LR(real LR))

### Total loss
* loss D = loss D HR + loss D LR
* loss G = adv_loss G HR + adv_loss G LR + (lambda_cyc * cyc_loss) * (lambda_id * id_loss)
* cyc_loss = cyc loss LR + cyc loss HR
* id_loss = id loss HR + id loss LR
* lambda_cyc, lambda_id sta neka hyperparametra, ponavadi sta 10 in 5
  * to pomeni da sta te izgubi veliko bolj pomembni in doprineseta več k izgubi
* te izgube uporabiš da posodobiš uteži vseh 4ih modelov