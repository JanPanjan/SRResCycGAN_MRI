// bibliography:
// https://github.com/typst/hayagriva/blob/main/docs/file-format.md

#import "@preview/zebraw:0.5.5": *
#show: zebraw
#let cb-lines(body, hdr: none, no-par: false, w: 80%) = {
  if (no-par == false) { par()[] }
  align(
    center,
    box(
      width: w,
      fill: rgb("#F5F5F5"),
      inset: 5pt,
      radius: 2pt,
      zebraw(header: strong(hdr), body),
    ),
  )
  if (no-par == false) { par()[] }
}

#let cb-nolines(body, hdr: none, no-par: false, w: 80%, fs: 9pt, hl-lines: range(1, 1)) = {
  if (no-par == false) { par()[] }
  align(
    center,
    box(
      width: w,
      fill: rgb("#F5F5F5"),
      inset: 5pt,
      radius: 2pt,
      text(
        size: fs,
        zebraw(
          numbering: false,
          header: hdr,
          highlight-lines: hl-lines,
          body,
        ),
      ),
    ),
  )
  if (no-par == false) { par()[] }
}

#let code(body) = {
  box(fill: rgb("#eeeeee"), radius: 2pt, outset: 2pt, body)
}

#let g(body1, body2) = {
  grid(
    columns: 2,
    body1, body2,
  )
}

#let m(ts: 9pt, body) = {
  text(size: ts, body)
}

#set math.equation(numbering: "(1)")
#set page(paper: "a4", margin: (y: 2cm, x: 2.3cm))
#set heading(numbering: "1.", bookmarked: true)
#set terms(tight: true, hanging-indent: 10pt)

#set text(
  lang: "si",
  size: 9pt,
  weight: "light",
  font: "Iosevka NF",
)

#show heading.where(level: 1): h => {
  linebreak()
  align(center, h)
  par()[]
}
#show heading.where(level: 2): h => {
  par()[]
  align(right, h)
  par()[]
}
#show heading.where(level: 3): h => {
  par()[]
  align(left, h)
  par()[]
}
#show link: l => { text(rgb("#4B69C6"), underline(l)) }
#show figure: f => {
  f
  par()[]
}
#show table.cell.where(y: 0): strong

// naslovnica
\ \ \ \
#align(center, text(28pt)[ Super resolucija MRI slik s pomočjo generativnih modelov ])
#par()[]
#par()[]
#align(center, text(12pt)[ Osnove strojnega učenja in podatkovnega rudarjenja ])
\ \ \ \ \ \ \ \ \ \
\ \ \ \ \ \ \ \ \ \
\ \ \ \ \ \ \ \ \ \
#align(center, text(12pt)[ 2024/25 ])
#align(center, text(12pt)[ Jan Panjan ])
#pagebreak()
#outline(depth: 3)
#pagebreak()

// šele tu nastavi številčenje strani, da je naslovnica prazna
#set par(justify: true)
#set page(numbering: "1", number-align: center)
#counter(page).update(1)

= Uvod

Uporaba magnetne resonance za slikanje pacientov je izjemnega pomena za diagnozo in sledenju boleznim, ki motijo zdravje pacientov. Podatki o organih, mehkih tkivih in kosteh - ki jih pridobijo z MRI slikanjem - omogočajo zdravnikom, da bolj učinkovito ocenijo stopnjo bolezni in posledično primerno prilagodijo način zdravljenja. Pacienti so tako deležni boljšega zdravljenja, kar je še predvsem pomembno pri raznih kompleksnih boleznih. Z MRI slikanjem - kljub temu da je izjemno orodje - ni enostavno pridobiti kvalitetnih podatkov. K temu pripomore več različnih faktorjev, tako človeških kot mehanskih.

== Problemi MRI slik

Čas priprave na slikanje je dolg, saj mora biti naprava kalibrirana na pacienta.
Čas slikanja je dolg, saj potrebuje naprava dovolj časa, da zajame toliko informacij, da jih lahko zdravniki natančno ocenijo. Naprava mora narediti mnogo slik (t.i. _slices_) iz različnih smeri. Vse te slike se na koncu združijo v smiselno celoto (t.i. _volume_).
Med slikanjem mora biti pacient na miru. Kakršnokoli premikanje vmesti v slike nezaželen šum in razne artefakte. Čas pridobivanja slike je sorazmeren s končno kvaliteto slik (manjši čas pridobivanja, manjša ločljivost). Med drugim so MRI slikanja tudi zelo draga za zdravstvene klinike kot posledica oskrbe naprave. Zaradi tega so dolžni tudi pacienti plačati več. Tu pripomorejo t.i. _low-field MRI scanners_, ki so cenejši. To omogoča, da je MRI slikanje dostopno vsem, vendar so slike pridobljene s temi napravami, relativno nižje resolucije.

#par()[]

Pri tem problemu lahko pomagajo SR (super resolution) metode, ki so zmožne rekonstruirati slike nizkih ločljivosti (LR, "low resolution") v slike visokih ločljivosti (HR, "high resolution"). Tu so aktualni t.i. GAN ("generative adversarial networks") modeli, ki so z globokim učenjem zmožni rekonstruirati slike z visoko natančnostjo. Razvitih je bilo več GAN modelov katerih namen je super resolucija slik, npr. SRGAN, ESRGAN, Real-ESRGAN,... ter tudi SRResCycGAN oziroma _Super Resolution Residual Cycle-consistent GAN_. @sr-with-mri

== O projektu

Za ta projekt sem si zadal implementirati SRResCycGAN model v jeziku Python z uporabo knjižnice TensorFlow. Programska koda je dostopna na #link("https://github.com/JanPanjan/SRResCycGAN_MRI")[github] in na #link("https://colab.research.google.com/drive/15Wztc85dwmdJwXp0B03hag4qVt5m_LR1?usp=sharing")[Google Colab]. Model je bil treniran na #link("https://fastmri.med.nyu.edu/")[FastMRI] podatkih, specifično na t.i. _singlecoil_ (2D) slikah kolen.

= Predstavitev podatkov

Zaradi velike količine podatkov (\~100GB) sem se omejil na podmnožico #code[singlecoil_train] in #code[singlecoil_val] podatkov. Imena uporabljenih datotek so vidna #link("https://github.com/JanPanjan/SRResCycGAN_MRI/blob/main/data/uporabljene_datoteke")[tu].

Podatki so shranjeni v #code[h5] datotekah. Vsaka datoteka predstavlja eno MRI slikanje. Na voljo so rekonstruirane slike pod ključem #code[reconstruction_rss] in surovi podatki k-prostora pod ključem #code[kspace].

*K-prostor* predstavlja MRI sliko v obliki prostorskih frekvenc. Te frekvence so pridobljene preko MR signalov. Signali, ki so v osnovi kompleksna števila, so pretvorjeni v t.i. _image space_ realnih števil s pomočjo inverzne Fourierjeve preslikave.

Oblike (uporabljenih) k-prostorskih podatkov se gibljejo od (28, 640, 320), kjer je 28 število slik, 640x320 pa velikost polja števil, do (50, 640, 640). Oblike rekonstruiranih slik se razlikujejo samo v številu slik, od 28 do 50 z velikostmi 320x320.

#par()[]

Spodaj je primer podatkov 10. slike shranjene v datoteki #code[file1000001.h5].

#grid(
  columns: 2, [
    #text(size: 7pt,
      figure(
        image("assets/kspace.png", width: 64%),
        caption: "vizualizacija k-prostorskih podatkov"
      )
    )
  ], [
    #text(size: 7pt,
      figure(
        image("assets/reconstructed.png"),
        caption: "rekonstruirana slika pridobljena iz k-prosorskih podatkov"
      )
    )
  ]
)

#align(center)[
  #text(size: 7pt,
    figure(
      image("assets/primerjava.png"),
      caption: "primerjava rekonstruirane (kspace) in očiščene slike (reconstruction_rss)"
    )
  )
]

Za učenje modela je bila uporabljena množica rekonstruiranih slik (#code[reconstruction_rss]) in sicer v obliki *parov (LR, HR)*, torej slike nizke in visoke ločljivosti. LR slike so pridobljene iz HR slik *bikubično interpolacijo* in so velikosti 80x80, HR slike pa 320x320. Oba nabora slik sta shranjena s 3 kanali.

Metoda #code[create_paired_dataset] postopoma na vsaki datoteki pokliče metodo #code[h5_generator], ki naloži, transformira in vrne slike kot par. Podatkovna zbirka #code[tf.data.Dataset] je ustvarjena z #code[BATCH_SIZE=8]. Programska koda je v modulu #link("https://github.com/JanPanjan/SRResCycGAN_MRI/blob/main/src/data_loader.py")[data_loader.py].

= Predstavitev GAN strutkure

GAN modeli so v osnovi sestavljeni iz dveh nevronskih mrež - generatorja in diskriminatorja - ki med sabo tekmujeta. Cilj *generatorja* je, da se nauči porazdelitev podatkov tako, da bo sposoben generirati resnične slike. Ker sam po sebi ne more prepoznati kdaj so njegove slike resnične je tu potreben *diskriminator*. Njegova naloga je, da se nauči razlikovati med generiranimi in resničnimi slikami.

#par()[]

Z drugimi besedami, cilj generatorja je, da za nek vzorec LR slik ustvari neresnične HR slike, ki bodo prepričale (oz. preslepile) diskriminatorja v to, da jih oceni kot resnične. Ocene diskriminatorja o resničnosti slik so potrebne za učenje obeh modelov skozi _backpropagation_, kjer bo diskriminator kaznovan ob napačnih ocenah slike (npr. neresnično oceni kot resnično), generator pa ko bodo njegove slike (pravilno) ocenjene kot neresnične.

#align(center, image("assets/arhitektura.png", width: 70%)) @sr-with-mri

== Adversarialna funkcija izgube <advloss>

Standardna funkcija izgube, ki jo uporabljajo GAN modeli je t.i. *adversarialna izguba*. Deluje na _min-max_ principu, saj poiskuša generator vrednosti funkcije čimbolj zmanjšati, diskriminator pa zvečati. V sklopu SRResCycGAN modela lahko funkcijo opišemo z enačbo

#par()[]

#let gent = $theta_G$
#let dist = $theta_G$

#m(ts: 10pt)[$
  min_(gent) max_(dist) EE_(x_r) \[ log(D_dist (x_r)) \] + EE_y \[ log(1 - D_dist (G_gent (y))) \]
$]

#par()[]

- #m($x_r$) zaznamuje resnično LR sliko in #m[$G_gent (y)$] generirano neresnično HR sliko z vhodno LR sliko #m[$y$]. #m[$EE_x_r$] je pričakovana vrednost od vseh resničnih slik in #m[$D_dist (x_r)$] diskriminatorjeva ocena verjetnosti, da je vhod #m[$x_r$] resničen.

- #m[$EE_y$] je pričakovana vrednost vseh vhodnih LR slik #m[$y$] in posledično pričakovana vrednost od vseh generiranih slik #m[$G_gent (y)$].

- #m[$D_dist (G_gent (y))$] je diskriminatorjeva ocena verjetnosti, da je generirana slika resnična.

- #m[$gent$] in #m[$dist$] predstavljata uteži in biase generatorja #m[$G$] in diskriminatorja #m[$D$].

#par()[]

Taka kot je deluje kot funkcija izgube za diskriminatorja, medtem ko generatorja zanima samo ocena diskriminatorja na njegovih neresničnih podatkih, zato je levi člen med učenjem generatorja izpuščen. Njegova funkcija izgube je torej

#par()[]

#m(ts: 10pt)[$
  L_G = 1/N sum_(i=1)^N -log (D_dist (G_dist (y_i))
$]

#par()[]

kjer je #m[$N$] število LR vzorcev za učenje (batch) in #m[$y_i$] vhodna LR slika. Bolj kot je ocena diskriminatorja blizu 1, bolj je vrednost logaritma (ocena verjetnosti) blizu 0.

#align(center, image("assets/negative-log.png", width: 80%))

= SRResCycGAN

Večino SR GAN metod uporablja parne podatke LR in HR slik, kjer so LR slike pridobljene z bikubično interpolacijo HR slike. To prisili modele v to, da se naučijo izničiti rezultat tega procesa, kar pa ne odraža realnega sveta - *v realnem svetu je degradacija odvisna od veliko faktorjev*, ki na različne načine degradirajo sliko (šum, artefakti zaradi kompresije, zamegljenost zaradi premikanja, napake v lečah, itd.). Modeli zato postanejo zelo dobri v obračanju bikubične interpolacije, vendar se ne obnesejo dobro na pravih _umazanih_ slikah, kjer je funkcija degradacije bolj kompleksna.

#par()[]

SRResCycGAN poiskuša rešiti ta problem tako, da se ne uči samo super resolucije (LR -> HR), ampak tudi realistično degradacijo (HR -> LR). Zaradi tega je njegova struktura bolj kompleksna - namesto dveh nevronskih mrež ima štiri, dva generatorja in dva diskriminatorja.

#par()[]

#align(center)[
  #figure(
    image("assets/architecture_structure/SRResCycGAN.png", width: 80%),
    caption: "Struktura SRResCycGAN"
  ) <model-struktura>
] @srrescycgan

#let gh = $G_(H R)$
#let gl = $G_(L R)$
#let dh = $D_(H R)$
#let dl = $D_(L R)$

#par()[]

Par #m[#gh], #m[#dh] (generator in diskriminator za HR slike) skrbi za učenje super resolucije, medtem ko par #m[#gl], #m[#dl] (analogno za LR slike) skrbi za učenje degradacije. @srrescycgan

== CycleGAN

Struktura je osnovana, med drugimi, na CycleGAN modelu, katerega cilje je, da se nauči aproksimirati preslikavo med vhodno in izhodno sliko v primerih, ko pravi pari slik za treniranje niso na voljo (npr. pri #link("https://www.tensorflow.org/tutorials/generative/style_transfer")[_style transfer_] metodah). Preslikavo #[$G$] med domeno degradiranih LR slik #m[$X$] in domeno čistih slik visoke resolucije #m[$Y$] zapišemo kot #m[$G:X arrow Y$]. Aproksimacija mora biti taka, da postane porazdelitev generiranih slik #m[$G(X)$] nerazločljiva od porazdelitve #m[$Y$]. Ker je preslikava sama po sebi (z uporabo adversarialne izgube) slabo omejena, obstaja neskončno možnih preslikav. V ta namen je združena z inverzno preslikavo #m[$F:Y arrow X$].

=== Ciklična izguba <cycloss>

Uvedena je tudi nova funkcija izgube, ki se imenuje *cycle (cyclic) consistency loss* oziroma ciklična izguba, ki zagotavlja, da je #m[$F(G(X)) approx X$] in obratno. Intuitivno si lahko to predstavljamo kot "prevod stavka iz slovenščine v angleščino mora biti identičen prevodu stavka nazaj v slovenščino". Za #m[$forall x in X$] in #m[$forall y in Y$] so avtorji cikel preslikav #m[$x -> G(x) -> F(G(x)) approx x$] poimenovali *forward cycle consistency* ter #m[$y -> F(y) -> G(F(y)) approx y$] *backward cycle consistency*. Cikla skupaj tvorita omenjeno funkcijo izgube:

#par()[]

#text(size: 12pt)[$
  L_("cyc")(G, F) = EE_(x) \[ norm( F(G(x)) - x )_1 \] + EE_y \[ norm( G(F(y)) - y )_1 \]
$]

#par()[]

V sklopu strukture SRResCycGAN (glej @model-struktura) vzame model #m[#gh] za vhod LR sliko in vrne generirano HR sliko, medtem ko #m[#gl] vzame HR sliko in vrne generirano LR sliko. #[#gh] nadzoruje #m[#dh], ki ocenjuje kako resnična je vhodna HR slika, #m[#gl] pa #m[#dl], ki ocenjuje kako resnična je vhodna LR slika.  @cyclegan

== HR Generator

Struktura generatorja HR slik je prevzeta iz modela SRResCGAN @srrescgan. Njegova struktura je prikazana spodaj. Deluje na principu "povečaj -> izboljšaj", kar pomeni, da v začetnem delu poveča resolucijo vhodne LR slike na željeno ločljivost, skozi vmesni *residual network* (ali ResNet) izpostavi napake v sliki (šum, artefakti, predstavljene kot feature maps), v zadnjem delu pa se te napake od povečane slike odstranijo. Avtorji so poimenovali njegove dele "Encoder -> ResNet -> Decoder".

#align(center)[
  #text(size: 7pt,
    figure(
      image("assets/architecture_structure/G_HR.png", width: 80%),
      caption: "HR generator model"
    )
  )
] @sr-with-mri

=== Encoder

Struktura se začne z *Encoder* delom, kjer prvi #code[Conv2DTranspose] poskrbi, da se vhodna slika velikosti #code[LR_SHAPE] (80, 80, 3) poveča na (320, 320, 3). Zadnja številka predstavlja število kanalov slike (RGB slike => 3 kanali). Drugi #code[Conv2D] sloj poskrbi, da se slika preslika v iz _image_ v _feature space_.

Privzeta vrednost za filtre/kanale (#code[FILTERS]) slojev Encoder-ja in Decoder-ja je 64 ter 5 za velikost konvolucijskega jedra (#code[kernel_size]).

#cb-nolines(w: 100%)[
  ```python
  def Generator_HR(input_shape=LR_SHAPE):
    lr_input = layers.Input(shape=input_shape)
    encoder_out = layers.Conv2DTranspose(
        filters=FILTERS,  # število kanalov izhoda; rgb slika ima 3, tu jih vrne 64)
        kernel_size=5,    # dimenzije jedra, ki bo procesiral sliko (5x5)
        strides=4,        # premik jedra (4 => 4x upsampling => 320)
        padding="same"    # doda ravno prav ničel na robove, da je končna dimenzija odvisna od strides
    )(lr_input)

    # služi kot vhod in izhod ResNet-a.
    x = layers.Conv2D(filters=FILTERS, kernel_size=5, padding="same")(encoder_out)
  ...
  ```
]

=== Residual Network

ResNet je arhitektura, ki poiskuša rešiti težave učenja globokih nevronskih mrež, npr. problem izginjajočih gradientov (vanishing gradients) in manjšanje natančnosti pri večjem številu slojev. To doseže preko t.i. *globokega residualnega učenja* - namesto, da bi vsak sloj aproksimiral preslikavo izhoda prejšnjega sloja v naslednjega, ResNet uči te sloje, da aproksimirajo t.i. *residualno preslikavo*.

Če je #m[$H(x)$] želena preslikava, potem to pomeni, da ResNet uči svoje sloje, da aproksimirajo preslikavo #m[$F(x) := H(x) - x$], #m[$H(x) = F(x) + x$]. Avtorji so postavili hipotezo, da je *lažje optimizirati residualno preslikavo* kot originalno. Če bi bila v skrajnem primeru optimalna rešitev identična preslikava (t.j. #m[$H(x) = x$]) naj bi bilo lažje potisniti #m[$F(x)$] proti 0, kot pa se naučiti identične preslikave preko zaporednih slojev. V praksi je formulacija #m[$H(x) - x$] realizirana preko t.i. *preskočnih povezav (skip connections)*. @resnet

#align(center)[
  #text(size: 7pt,
    figure(
      image("assets/architecture_structure/residual.png", width: 50%),
      caption: "Residualni blok"
    )
  ) <residual-block>
]

To preprosto izgleda tako, da shranimo izhod prejšnjega sloja v spremenljivko (#m(ts: 10pt)[$x$] na sliki) za poznejšo uporabo.

#cb-nolines(w: 27%)[
  ```python
  ...
  global_skip = encoder_out
  ...
  ```
]

ResNet je sestavljen iz 5 residualnih blokov. Implementirani so z dvema t.i. *pre-activation* konvolucijskema slojema, kar pomeni, da aktivacijski sloj (v tem modelu PReLU) nastopi pred uteženimi sloji. Izkazalo se je, da to izboljša proces optimizacije. @preactivation

Za vsakim konvolucijskim slojem v residualnih blokih nastopi še normalizacijski sloj, ki prav tako rešuje problem izginjajočih gradientov in prekomerno prileganje na podatke (overfitting) tako da normalizira vrednosti. Namesto standardnega #code[BatchNormalization] (BN) je uporabljen #code[InstanceNormalization] (IN), saj BN izračuna povprečje in varianco celotnega batch-a, kar pomeni, da je normalizacija vsake slike odvisna od vseh. Izkazalo se je, da je to krivo za razne artefakte v generiranih slikah, zato priporočajo uporabo IN, ki normalizira vsak kanal neodvisno in tako ohranja značilnosti vsake slike. @instance-norm @sr-with-mri

#cb-nolines(w: 100%)[
  ```python
  ...
  for _ in range(5):
      block_skip = x
      x = layers.PReLU(shared_axes=[1, 2])(block_skip)
      x = layers.Conv2D(filters=FILTERS, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
      x = layers.GroupNormalization(groups=-1, axis=-1)(x)  # -1 za InstanceNormalization

      x = layers.PReLU(shared_axes=[1, 2])(x)
      x = layers.Conv2D(filters=FILTERS, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
      x = layers.GroupNormalization(groups=-1, axis=-1)(x)
      x = layers.Add()([block_skip, x])
  ...
  ```
]

=== Decoder

Decoder vzame izhod residualnih blokov in ga pošlje še skozi dva konvolucijska sloja. Drugi sloj naj bi bil t.i. _projection layer_ parametriziran s #m(ts: 11pt)[$sigma$]. Zaradi kompleksnosti sem implementacijo tega poljubnega sloja izpustil in uporabil preprost konvolucijski sloj.

#cb-nolines(w: 100%)[
  ```python
  ...
  decoder_out = layers.Conv2D(filters=FILTERS, kernel_size=5, strides=1, padding="same")(x)
  decoder_out = layers.Conv2D(filters=FILTERS, kernel_size=5, strides=1, padding="same")(decoder_out)
  ...
  ```
]

Končna _residualna slika_ je po decoder-ju odšteta od prejšnje LR slike, shranjena kot #code[global_skip]. Zadnji konvolucijski sloj poskrbi, da ima izhodna slika 3 kanale, ReLU pa da so vrednosti omejene na interval (0, 255).

#cb-nolines(w: 100%)[
  ```python
  ...
  subtracted = layers.Subtract()([global_skip, decoder_out])
  model_out = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same", use_bias=False)(subtracted)
  model_out = layers.ReLU(max_value=255)(model_out)  # vrednosti spravi na interval [0,255]

  return Model(inputs=lr_input, outputs=model_out, name="G_HR")
  ```
]

== HR Diskriminator

HR Generator je pod nadzorom HR Diskriminatorja, ki ocenjuje kako resnične so vhodne HR slike.

#align(center)[
  #text(size: 7pt,
    figure(
      image("assets/architecture_structure/D_HR.png", width: 80%),
      caption: "HR diskriminator"
    )
  ) <dhr>
] @sr-with-mri

Struktura je posvojena od SRGAN modela. Ima 10 konvolucijskih slojev z izmenjajočimi velikostmi konvolucijskega jedra 3 in 4 ter naraščujočim številom filtrov (64 -> 512). Za vsakim konvolucijskim slojem nastopita LeakyReLU in normalizacijski sloj. Ponovno je BN zamenjan z IN.

#cb-nolines(w: 100%)[
  ```python
  def Discriminator_HR(input_shape=HR_SHAPE):
    hr_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="valid")(hr_input)
    x = layers.LeakyReLU()(x)

    KERNEL_SIZE = (4, 3)
    STRIDES = (2, 1)
    FILTERS = (
        64,
        128, 128,
        256, 256,
        512, 512, 512, 512
    )

    for i in range(9):
        kernel_size = KERNEL_SIZE[0] if i % 2 == 0 else KERNEL_SIZE[1]
        strides = STRIDES[0] if i % 2 == 0 else STRIDES[1]

        x = layers.Conv2D(filters=FILTERS[i], kernel_size=kernel_size, strides=strides, padding="valid")(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(units=100)(x)
    x = layers.LeakyReLU()(x)
    model_out = layers.Dense(units=1)(x)

    return Model(inputs=hr_input, outputs=model_out, name="D_HR")
  ```
]

Izhod modela je v tem primeru logit vrednost, ki predstavlja preslikane vrednosti verjetnosti iz intervala #m[$\[0,1\]$] na #m[$\(-infinity, infinity\)$].

== LR Generator

LR Generator ima bolj preprosto, "Conv -> ResNet -> Conv", strukturo. Vsi konvolucijski sloji, razen zadnji, uporabljajo 64 filtrov. *Glava* s tremi konvolucijskimi sloji poskrbi za downsampling vhodne HR slike.

#cb-nolines(w: 100%)[
  ```python
  def Generator_LR(input_shape=HR_SHAPE):
    hr_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=FILTERS, kernel_size=7, padding="same")(hr_input)
    x = layers.GroupNormalization(groups=-1, axis=-1)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    for _ in range(2):  # downsample
        x = layers.Conv2D(filters=FILTERS, kernel_size=3, strides=2, padding="same")(x)
        x = layers.GroupNormalization(groups=-1, axis=-1)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
  ...
  ```
]

Vmesni *ResNet* je implementiran z residualnimi bloki "Conv -> Norm -> LReLU" s preskočnimi povezavami.

#cb-nolines(w: 100%)[
  ```python
  ...
  for _ in range(6): # ResNet
      block_skip = x
      x = layers.Conv2D(filters=FILTERS, kernel_size=3, padding="same")(x)
      x = layers.GroupNormalization(groups=-1, axis=-1)(x)
      x = layers.LeakyReLU(negative_slope=0.2)(x)

      x = layers.Conv2D(filters=FILTERS, kernel_size=3, padding="same")(x)
      x = layers.GroupNormalization(groups=-1, axis=-1)(x)
      x = layers.Add()([block_skip, x])
      x = layers.LeakyReLU(negative_slope=0.2)(x)
  ...
  ```
]

Zadnji konvolucijski sloj v *repu* poskrbi, da ima izhodna slika 3 kanale in zadnji ReLU sloj, da so vrednosti na intervalu [0,255].

#cb-nolines(w: 100%)[
  ```python
  ...
  for _ in range(2):
      x = layers.Conv2D(filters=FILTERS, kernel_size=3, padding="same")(x)
      x = layers.GroupNormalization(groups=-1, axis=-1)(x)
      x = layers.LeakyReLU(negative_slope=0.2)(x)

  x = layers.Conv2D(filters=3, kernel_size=7, padding="same")(x)
  model_out = layers.ReLU(max_value=255)(x)  # vrednosti spravi na interval [0,255]

  return Model(inputs=hr_input, outputs=model_out, name="G_LR")
  ```
]

== LR Diskriminator

Deluje podobno kot diskriminator v PatchGAN modelu, kjer model ne poiskuša oceniti celotne slike v kosu kot resnično, ampak oceni posamezne kose (patches). To doseže tako, da na koncu ne agregira vseh vrednosti v en #code[Flatten -> Dense] sloj, ampak vrne matriko logitov velikosti 20x20. Vsi konvolucijski sloji uporabljajo velikost jedra 5. Število filtrov se skozi sloje poveča iz 64 na 256, na koncu pa se združijo v enega. @patch-gan

Prvi #code[Conv2D] sloj zmanjša vhodno LR sliko iz 80x80 na 40x40 zaradi #code[strides=2]. Drugi sloj ponovno prepolovi dimenziji na pol, zato je matrika logitov oblike 20x20.

Za vsakim konvolucijskim slojem nastopita normalizacijski in aktivacijski sloj, podobno kot pri ostali modelih.

#cb-nolines(w: 100%)[
  ```python
  KERNEL_SIZE = 5
  FILTERS = [64, 128, 256]

  lr_input = layers.Input(shape=input_shape)

  x = layers.Conv2D(filters=FILTERS[0], kernel_size=KERNEL_SIZE, strides=2, padding="same")(lr_input)
  x = layers.GroupNormalization(groups=-1, axis=-1)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(filters=FILTERS[1], kernel_size=KERNEL_SIZE, strides=2, padding="same")(x)
  x = layers.GroupNormalization(groups=-1, axis=-1)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Conv2D(filters=FILTERS[2], kernel_size=KERNEL_SIZE, padding="same")(x)
  x = layers.GroupNormalization(groups=-1, axis=-1)(x)
  x = layers.LeakyReLU()(x)

  model_out = layers.Conv2D(filters=1, kernel_size=KERNEL_SIZE, padding="same")(x)

  return Model(inputs=lr_input, outputs=model_out, name="D_LR")
  ```
]

== Funkcije izgube

Da naučimo model super resolucije slik, je potrebna uporaba več različnih funkcij izgube. Enačba, ki se bo skozi učenje optimizirala je

#par()[]

#m(ts: 10pt)[$
  L_(#gh) = L_P + L_G + L_(T V) + lambda_("content") dot L_1 + lambda_("cyclic") dot L_(C Y C)
$]

#par()[]

kjer je #m[$L_P$] izguba zaznave (perceptual loss), #m[$L_G$] adversarialna izguba (adversarial loss), #m[$L_(T V)$] izguba popolne variacije (total variation loss), #m[$L_1$] izguba vsebine (content loss) in #m[$L_(C Y C)$] ciklična izguba (cyclic loss). Naloga skalarjev #m[$lambda_("content")$] in #m[$lambda_("cyclic")$] je, da doda vsebinski in ciklični izgubi večjo težo, kar sili učenje generatorja, da postavi več pozornosti na ti dve izgubi.

=== Izguba zaznave

Fokusira se na "zaznavno podobnost" dveh slik. To doseže tako, da računa razdalje med posameznimi piksli v prostoru značilnosti (feature space) namesto v prostoru slik (image space). Definirana je kot L2 (evklidska) norma

#m(ts: 11pt)[$
  L_P = 1/N sum_(i=1)^N norm( Phi(accent(x, hat)_r_i) - Phi(x_g_i) )_2^2
$]

med zemljevidoma značilnosti (feature maps) generirane HR slike #m[$x_g_i$] in njene resnične HR vrednosti #m[$accent(x, hat)_r_i$]. Zemljevida značilnosti, označena z #m[$Phi(I)$], kjer je #m[$I$] vhodna slika, sta pridobljena iz nekega vmesnega konvolucijskega sloja VGG19 klasifikatorja.

=== Izgube popolne variacije

MRI slike so ponavadi podvržene nizkemu razmerju med šumom in signalom (signal-to-noise ratio). To vpliva na učenje modela, saj obravnava pomanjkljivosti v slikah kot njihove značilnosti, kar vodi do popačenih HR slik. Izguba popolne variacije tako poiskuša zagotoviti, da je šum med učenjem pravilno obravnavan. Definirana je kot

#m[$
  L_(T V) = 1/N sum_(i=1)^N \( norm( gradient_h G(y_i) - gradient_h (accent(x, hat)_r_i) )_1 + norm( gradient_v G(y_i) - gradient_v (accent(x, hat)_r_i) )_1 \)
$]

#par()[]

kjer sta #m[$gradient_h$] in #m[$gradient_v$] vodoravna in navpična gradienta slike. Kot je razvidno na spodnji sliki, lahko z gradienti zaznamo prisotnost šuma.

#align(center)[
  #text(size: 7pt,
    figure(
      image("assets/gradients.png", width: 55%),
      caption: "(a) originalna slika, (b) navpični gradienti, (c) vodoravni gradienti"
    )
  )
] @sr-with-mri

=== Izguba vsebine

Preprosto izračuna povprečje razlik med piksli generirane in resnične HR slike. S tem na široko oceni rekonstrukcijo slike. Definirana je kot L1 norma

#m[$
  L_1 = 1/N sum_(i=1)^n norm( G(y_i) - accent(x, hat)_r_i )_1
$]

med generirano HR sliko #m[$G(y_i)$] pridobljena iz vhodne LR slike #m[$y_i$] in resnično HR sliko #m[$accent(x, hat)_r_i$].

#par()[]

#link(<advloss>)[Adversarialna] in #link(<cycloss>)[ciklična izguba] sta bili opisani prej. Vse funkcije izgube so definirane znotraj slovarja #code[losses_dict] v modulu #link("https://github.com/JanPanjan/SRResCycGAN_MRI/blob/main/src/losses.py")[losses.py]

== Učenje

Celotni model je definiran v modulu #link("https://github.com/JanPanjan/SRResCycGAN_MRI/blob/main/src/SRResCycGAN.py")[SRResCycGAN.py]. Za učenje sta za #m[$lambda_("content")$] in #m[$lambda_("cyclic")$] uporabljeni privzeti vrednosti 5 in 10. Za vse modele je uporabljen poljuben Adam optimizator @sr-with-mri, definiran na sledeč način:

#cb-nolines(w: 80%)[
  ```python
def adam_opt(lr=1e-4, b1=0.9, b2=0.999, decay_steps=1e4, decay_rate=0.5):
    """ Custom Adam optimizer s padajočim learning rate. """
    lr_schedule = ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True  # da se lr spremeni na vsakih 10k korakov
    )
    return Adam(learning_rate=lr_schedule, beta_1=b1, beta_2=b2, weight_decay=False)
  ```
]

Metoda #code[ExponentialDecay] poskrbi, da se stopnja učenja (na začetku #m[$1 × 10^(-4)$]) vsakih 10'000 korakov prepolovi.

Vsak korak učenja, model pokliče metodo #code[train_step], ki skrbi za izračun vrednosti funkcij izgub in gradientov ter posodabljanja uteži ob dani vhodni resnični LR in HR sliki.

= Rezultati

Kljub temu, da je vse potrebno sprogramirano, žal nisem uspel pridobiti rezultatov zaradi konstantnih napak v Kaggle in Google Colab okolju, ko sem poiskušal pognati model z GPU ali TPU pospeševalniki. Model je možno pognati na CPU, vendar je, kljub mojem zoožanem izboru podatkov, za 1 epoch predviden čas izvajanja približno 52 ur (slika spodaj). 😭

#par()[]

#align(center)[
  #text(size: 7pt,
    figure(
      image("assets/nemorem-verjet.png", width: 90%),
      caption: "Google Colab, CPU runtime, predviden čas izvajanja za vseh 3254 korakov je približno 52 ur"
    )
  )
]

#pagebreak()
#bibliography("sources.yml")
