// bibliography:
// https://github.com/typst/hayagriva/blob/main/docs/file-format.md
#import "@preview/zebraw:0.5.5": *
#show: zebraw
#let cb-lines(body, hdr: none, no-par: false, w: 80%) = {
  if (no-par == false) { par()[] }
  align(center,
    box(
      width: w,
      fill: rgb("#F5F5F5"),
      inset: 5pt,
      radius: 2pt,
      zebraw(header: strong(hdr), body)
    )
  )
  if (no-par == false) { par()[] }
}
#let cb-nolines(body, hdr: none, no-par: false, w: 80%, fs: 9pt, hl-lines: range(1,1)) = {
  if (no-par == false) { par()[] }
  align(center,
    box(
      width: w,
      fill: rgb("#F5F5F5"),
      inset: 5pt,
      radius: 2pt,
      text(size: fs,
        zebraw(
          numbering: false,
          header: hdr,
          highlight-lines: hl-lines,
          body
        )
      )
    )
  )
  if (no-par == false) { par()[] }
}
#let code(body) = {
  box(fill: rgb("#eeeeee"), radius: 2pt, outset: 2pt, body)
}
#let g(body1, body2) = {
  grid(columns: 2, body1, body2)
}
#set page(paper: "a4", margin: (y: 2cm, x: 2.3cm))
#set heading(numbering: "1.", bookmarked: true)
#set terms(tight: true, hanging-indent: 10pt)
#set text(
  lang: "si",
  size: 8pt,
  font: "JetBrainsMono NF", // najdi bussin mono font
  weight: "light"
)
#show heading.where(level: 1): h => { linebreak(); align(center, h); par()[] }
#show heading.where(level: 2): h => { par()[]; align(right, h); par()[] }
#show heading.where(level: 3): h => { align(left, h) }
#show link: l => { text(rgb("#4B69C6"), underline(l)) }
#show figure: f => { f; par()[] }
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

// NOTE:
// - ne kopirat celotne kode, kopiraj pomembne snippets in redirectaj ljudi na svoj repo kjer je ves code

= Uvod
// predstaviš problem MRI slik
// kako lahko tu pomagajo generativni modeli
// kaj je bil cilj tega projekta
// katere knjižnice boš uporabil
// kje so podatki (FastMRI, github repo do moje kode)
= Predstavitev GAN strutkure
// osnovno o GANsih
yupu @srrescycgan-paper
== Generator
== Diskriminator
== Loss funkcija
// adversarial loss
== Obstoječi pristopi za SR MRI slik // dodelaj naslov
// SRGAN, ESRGAN, BebyGAN...
= Predstavitev podatkov
// v kakšni obliki so MRI podatki (kspace, resconstruction...)
// kako dobiš slike (primer od fastmri) s fourierjevo transformacijo
= SRResCycGAN
// predstaviš posamezne networks preprosto, da bo smiselno kako deluje končni model
== Residual GAN // ResNet
== Cycle-consistent GAN // CycGAN
// podrobnosti tega modela (layers, loss, activation...)
// komentiraj izbiro med batch normalization ter instance normalization (primerjaj obe implementaciji)
// kako se razlikuje, zakaj je primeren MRI slikam
// link do kode
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

#bibliography("sources.yml")