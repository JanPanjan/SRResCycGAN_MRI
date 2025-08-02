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

#let m(ts: 11pt, body) = {
  text(size: ts, body)
}

#set math.equation(numbering: "(1)")
#set page(paper: "a4", margin: (y: 2cm, x: 2.3cm))
#set heading(numbering: "1.", bookmarked: true)
#set terms(tight: true, hanging-indent: 10pt)

#set text(
  lang: "si",
  size: 10pt,
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
\ \ \ \ \ \ \ \ \ \ \ \ \
\ \ \ \ \ \ \ \ \ \ \ \ \
\ \ \ \ \ \ \ \ \ \ \ \ \
#align(center, text(12pt)[ 2024/25 ])
#align(center, text(12pt)[ Jan Panjan ])
#pagebreak()
#outline(depth: 2)
#pagebreak()

// šele tu nastavi številčenje strani, da je naslovnica prazna
#set par(justify: true)
#set page(numbering: "1", number-align: center)
#counter(page).update(1)

= Uvod

// predstaviš problem MRI slik
Uporaba magnetne resonance za slikanje pacientov je izjemnega pomena za diagnosticiranje bolezni in sledenju boleznim, ki že motijo zdravje pacientov. Podatki o organih, mehkih tkivih in kosteh, katere pridobijo z MRI slikanjem, omogočajo zdravnikom, da bolj učinkovito ocenijo stopnjo bolezni in posledično primerno prilagodijo način zdravljenja. Pacienti so tako deležni bolj kvalitetnega zdravljenja, kar je še predvsem pomembno pri raznih kompleksnih boleznih. MRI slikanje pa kljub temu, da je izjemno orodje, ni enostavno za uporabljati oziroma ni enostavno pridobiti kvalitetnih podatkov. K temu pripomore več različnih faktorjev, tako človeških kot strojnih.

== Problemi MRI slik

Čas priprave na slikanje je dolg, saj mora biti naprava kalibrirana na pacienta.
Čas slikanja je dolg, saj potrebuje naprava dovolj časa, da zajame toliko informacij, da jih lahko zdravniki natančno ocenijo. Naprava narediti mnogo slik (t.i. _slices_) iz različnih smeri. Vse te slike se na koncu združijo v smiselno celoto (t.i. _volume_).
Med slikanjem mora biti pacient na miru, saj premikanje vmesti v slike nezaželen šum in razne artefakte. Čas pridobivanja slike je tudi sorazmeren s končno kvaliteto slik (manjši čas pridobivanja, manjša ločljivost). Med drugim so MRI slikanja tudi zelo draga za zdravstvene klinike kot posledica oskrbe naprave. Posledično morajo zato tudi pacienti plačati več. Tu pripomorejo t.i. _low-field MRI scanners_, ki so cenejši. To omogoča, da je MRI slikanje dostopno vsem, vendar so slike pridobljene s temi napravami relativno nižje resolucije.

#par()[]

Pri tem problemu lahko pomagajo SR (super resolution) metode, ki so zmožne rekonstruirati slike nizkih ločljivosti v slike visokih ločljivosti.Tu so aktualni tako imenovani GAN (generative adversarial networks) modeli. Z globokim učenjem so zmožni rekonstruirati slike v večjih resolucij z visoko natančnostjo. Razvitih je bilo več GAN modelov katerih namen je super resolucija slik, npr. SRGAN, ESRGAN, Real-ESRGAN, itd. Med njimi je tudi SRResCycGAN oziroma _Super Resolution Residual Cycle-consistent GAN_. @sr-with-mri

== O projektu

Za ta projekt sem si zadal implementirati SRResCycGAN model v jeziku Python z uporabo knjižnice TensorFlow. Za vso programsko kodo je bila uporavljena različica 2.15. Programska koda je dostopna na #link("https://github.com/JanPanjan/SRResCycGAN_MRI")[github]. Model je bil treniran na #link("https://fastmri.med.nyu.edu/")[FastMRI] podatkih, specifično na t.i. _singlecoil_ slikah kolen.

= Predstavitev GAN strutkure

// osnovno o GANsih
GAN modeli so v osnovi sestavljeni iz dveh nevronskih mrež - generator in diskriminator, ki med sabo tekmujeta. Cilj *generatorja* je, da se nauči porazdelitev podatkov, tako da bo sposoben generirati resnične slike. Ker sam po sebi ne more prepoznati kdaj so njegove slike resnične je tu potreben *diskriminator*. Njegova naloga je, da se nauči razlikovati med generiranimi in resničnimi slikami.

#par()[]

Z drgumi besedami, cilj generatorja je, da za nek vzorec LR (low resolution) slik ustvari neresnične HR slike, ki bodo prepričale (preslepile) diskriminatorja v to, da jih oceni kot resnične. Po drugi strani pa je diskiminatorjev cilj, da pravilno oceni resničnost slike. Njegove ocene so potrebne za učenje obeh modelov skozi _backpropagation_, kjer bo diskriminatorja funkcija izgube kaznovala ob napačnih ocenah, generatorja pa ko bodo njegove slike ocenjene (pravilno) kot neresnične.

#align(center, image("assets/arhitektura.png", width: 70%))

== Funkcija izgube

Standardna funkcija izgube, ki jo uporabljajo GAN modeli se imenuje *adversarial loss*. Deluje na _min-max_ principu, saj poiskuša generator vrednosti funkcije čimbolj zmanjšati, diskriminator pa zvečati.

#let gent = $theta_G$
#let dist = $theta_G$

$
  min_(gent) max_(dist) EE_(x_r) \[ log(D_dist (x_r)) \] + EE_y \[ log(1 - D_dist (G_gent (y))) \]
$

#par()[]

- #m($x_r$) zaznamuje resnično sliko in #m[$G_gent (y)$] generirano neresnično HR sliko z vhodno LR sliko #m[$y$]. #m[$EE_x_r$] je pričakovana vrednost od vseh resničnih slik in #m[$D_dist (x_r)$] diskriminatorjeva ocena verjetnosti, da je vhod #m[$x_r$] resničen.

- #m[$EE_y$] je pričakovana vrednost vseh vhodnih LR slik #m[$y$] in posledično pričakovana vrednost od vseh generiranih slik #m[$G_gent (y)$].

- #m[$D_dist (G_gent (y))$] je diskriminatorjeva ocena verjetnosti, da je generirana slika resnična.

- #m[$gent$] in #m[$dist$] predstavljata uteži in biase generatorja #m[$G$] in diskriminatorja #m[$D$]. Oba modela sta skupaj optimizirana z zgornjo funkcijo.

#par()[]

Taka kot je deluje kot funkcija izgube za diskriminatorja, medtem ko generatorja zanima samo ocena diskriminatorja na njegovih neresničnih podatkov, zato je levi člen med učenjem generatorja opuščen. Njegova funkcija izgube je torej

#par()[]

$
  L_G = 1/N sum_(i=1)^N -log (D_dist (G_dist (y_i))
$

#par()[]

kjer je #m[$N$] število LR vzorcev za učenje (en batch) in #m[$y_i$] vhodna LR slika. Bolj kot je ocena diskriminatorja blizu 1, bolj je vrednost logaritma (ocena verjetnosti) blizu 0.

#align(center, image("assets/negative-log.png", width: 80%))

= SRResCycGAN

Večino SR GAN metod uporablja parne podatke LR in HR slik, kjer so LR slike pridobljene z bikubično interpolacijo HR slike. To prisili modele v to, da se naučijo izničiti rezultat tega procesa, kar pa ne odraža realnega sveta - *v realnem svetu je degradacija odvisna od veliko različnih faktorjev*, ki na veliko načinov degradirajo sliko (šum, artefakti zaradi kompreisje, zamegljenost zaradi premikanja, napake v lečah, itd.). Modeli zato postanejo zelo dobri v obračanju npr. bikubične interpolacije, a se ne obnese dobro na pravih _umazanih_ slikah.

#par()[]

SRResCycGAN je poiskuša rešiti ta problem tako, da se ne uči samo super resolucije (LR -> HR), ampak tudi realistično degradacijo (HR -> LR). Zaradi tega je njegova struktura bolj kompleksna - namesto dveh nevronskih mrež ima štiri - dva generatorja in dva diskriminatorja.

#par()[]

#align(center)[
  #figure(
    image("assets/architecture_structure/SRResCycGAN.png", width: 80%),
    caption: "Struktura SRResCycGAN"
  ) <model-struktura>
]

#let gh = $G_(H R)$
#let gl = $G_(L R)$
#let dh = $D_(H R)$
#let dl = $D_(L R)$

#par()[]

Par #m[#gh], #m[#dh] skrbi za učenje super resolucije, medtem ko par #m[#gl], #m[#dl] skrbi za učenje degradacije. @srrescycgan

== CycleGAN

Struktura je osnovana, med drugimi, na CycleGAN modelu, ki se poiskuša naučiti preslikave med vhodno in izhodno sliko v primerih, ko pravi pari slik za treniranje niso na voljo, kar je primer tudi pri npr. _style transfer_ metodah. Cilj modela je se naučiti preslikave #[$G$] med domeno umazanih LR slik #m[$X$] in domeno čistih slik visoke resolucije #m[$Y$], kar zapišemo kot #m[$G:X arrow Y$], tako da je porazdelitev generiranih slik #m[$G(X)$] nerazločljiva od porazdelitve #m[$Y$] z uporabo adversarialne izgube. Ker je preslikava sama po sebi slabo omejena in obstaja neskončno možnih preslikav, je združena z inverzno preslikavo #m[$F:Y arrow X$].

=== Cycle consistency loss

Uvedena je tudi nova funkcija izgube, ki se imenuje *cycle consistency loss*, ki zagotavlja, da je #m[$F(G(X)) approx X$] in obratno. Intuitivno si lahko predstavljamo to kot "prevod stavka iz slovenščine v angleščino mora biti identičen prevodu stavka nazaj v slovenščino". Za #m[$forall x in X$] in #m[$forall y in Y$] so cikel preslikav #m[$x -> G(x) -> F(G(x)) approx x$] poimenovali *forward cycle consistency* ter #m[$y -> F(y) -> G(F(y)) approx y$] *backward cycle consistency*. Skupaj tvorita omenjeno funkcijo izgube:

#par()[]

#text(size: 12pt)[$
  L_("cyc")(G, F) = EE_(x) \[ norm( F(G(x)) - x )_1 \] + EE_y \[ norm( G(F(y)) - y )_1 \]
$]

#par()[]

V sklopu strukture SRResCycGAN (glej @model-struktura) vzame model #m[#gh] za vhod LR sliko in vrne generirano HR sliko, medtem ko #m[#gl] vzame HR sliko in vrne generirano LR sliko. #[#gh] nadzoruje #m[#dh], ki ocenjuje kako resnična je vhodna HR slika (bodisi prava ali generirana), #m[#gl] pa #m[#dl], ki ocenjuje kako resnična je vhodna LR slika.  @cyclegan

// podrobnosti tega modela (layers, loss, activation...)

== HR Generator

Struktura generatorja HR slik je prevzeta iz modela SRResCGAN. Njegova struktura je prikazana spodaj (glej @ghr). Deluje na principu "povečaj -> izboljšaj", kar pomeni, da v začetnem delu poveča resolucijo vhodne LR slike na željeno ločljivost, skozi vmesni *residual network* (ali ResNet) se izpostavijo napake v sliki (šum, artefakti), v zadnjem delu pa se te napake od povečane slike odstranijo.

=== Residual Network

ResNet je arhitektura, ki poiskuša rešiti težave učenja globokih nevronskih mrež, npr. problem izginjajočih gradientov (vanishing gradients) in manjšanje natančnosti pri večjem številu slojev. To doseže preko t.i. *globokega residualnega učenja* - namesto, da bi vsak sloj aproksimiral preslikavo izhoda prejšnjega sloja v naslednjega, ResNet uči te sloje, da se prilagodijo drugi, *residualni preslikavi*.

Formalno to pomeni, če je #m[$H(x)$] želena preslikava, potem ResNet uči svoje sloje, da aproksimirajo preslikavo #m[$F(x) := H(x) - x$], #m[$H(x) = F(x) + x$]. Avtorji so postavili hipotezo, da je *lažje optimizirati residualno preslikavo* kot originalno. To si lahko predstavljamo, da če bi bila v skrajnem primeru optimalna rešitev identična preslikava (t.j. #m[$H(x) = x$]) naj bi bilo lažje potisniti #m[$F(x)$] proti 0, kot pa se naučiti identične preslikave preko zaporednih slojev. V praksi je formulacija #m[$H(x) - x$] realizirana preko t.i. *preskočnih povezav (skip connections)*.

@resnet

#align(center)[
  #figure(
    image("assets/architecture_structure/ghr.png", width: 100%),
    caption: "HR generator model"
  ) <ghr>
]

// komentiraj izbiro med batch normalization ter instance normalization (primerjaj obe implementaciji)

// kako se razlikuje, zakaj je primeren MRI slikam

// link do kode

== Residual GAN // ResNet

= Predstavitev podatkov

// v kakšni obliki so MRI podatki (kspace, resconstruction...)

// kako dobiš slike (primer od fastmri) s fourierjevo transformacijo
= Podrobnosti implementacije

// data loading, data preprocessing (data pipeline?)

// model hyperparameters

// training loop

// evaluation methods

= Rezultati // is this needed? odvisno od cilja projekta

// kvantitativne metrike

// kvalitativna primerjava z GT slikami (LR vs HR slike)

// drawbacks modela

// možnosti izboljšave

= Zaključek

// kako se obnese SR MRI slik v praksi

// kakšna je prihodnost v tej smeri

#pagebreak()
#bibliography("sources.yml")
