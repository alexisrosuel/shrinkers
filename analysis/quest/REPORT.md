# `shrinkers` vs QuEST (Ledoit–Wolf 2016) — audit

**Objet.** Vérifier si ce que fait notre paquet `shrinkers` est la même chose que
la fonction **QuEST** de Ledoit & Wolf ([arXiv:1601.05870](https://arxiv.org/abs/1601.05870)),
si les calculs coïncident, et comparer les temps d'exécution.

**Méthode.** J'ai porté fidèlement `QuEST.m` (code MATLAB officiel des auteurs,
récupéré via le dépôt [LSS_Bootstrap/QuEST.m](https://github.com/AndoBlando/LSS_Bootstrap/blob/master/QuEST.m))
en NumPy (`quest_reference.py`), validé ce port contre la formule fermée
Marchenko–Pastur et contre des simulations Monte-Carlo, puis comparé à
`shrinkers` 0.1.2 (recompilé depuis HEAD pour cet audit).

---

## 1. Verdict en quatre lignes

1. **Ce ne sont pas les mêmes objets.** QuEST est l'application **directe**
   `population → spectre empirique limite`. `shrinkers` calcule l'application
   **inverse** `spectre → population` — c'est-à-dire précisément l'opérateur que
   Ledoit–Wolf doivent *inverser numériquement* pour estimer les valeurs propres
   de population.
2. **Même noyau mathématique.** Les deux reposent sur la même équation de
   Marčenko–Pastur/Silverstein et, surtout, sur **le même changement de variable**
   `u = z / a(z)`. J'ai vérifié que le `u` de QuEST est exactement le `w` de la
   déconvolution El Karoui de `shrinkers`.
3. **Les calculs coïncident.** `shrinkers.ledoit_wolf_shrinkage` implémente
   *littéralement* la formule de shrinkage optimal que QuEST renvoie dans sa
   sortie `d` (à ceci près que QuEST la calcule depuis la population modèle et
   `shrinkers` depuis le spectre empirique). Sur le support continu, l'écart
   relatif médian tombe à **4·10⁻⁴ à p = 20 000**. Le round-trip
   `τ → QuEST → λ → shrinkers → τ̂` récupère `τ` à **~10⁻³** (spikes) et
   **~10⁻³** (masse du bulk), et `QuEST(τ̂)` reproduit `λ`.
4. **Runtime.** Une évaluation QuEST directe coûte ~8–24 ms (port NumPy, p =
   500–8 000) ; l'inverse complet de `shrinkers` coûte 0,07–16 ms sur un thread.
   Mais LW doivent **inverser** QuEST avec un optimiseur non linéaire :
   **219 évaluations directes mesurées**, soit **1,66 s à p = 100** contre
   **0,0066 ms** pour `shrinkers` (facteur ≈ 2,5·10⁵) — et `shrinkers` est un
   one-shot O(p log p)/O(p²), sans optimiseur.

**Conclusion :** `shrinkers` n'est pas une ré-implémentation de QuEST, c'est son
**opérateur inverse**, augmenté d'un traitement explicite des spikes (BEMA +
BBP inverse) que la formule continue de QuEST ne fournit pas. Pour l'estimand
visé (les valeurs propres de population), les deux sont cohérents ; `shrinkers`
y arrive en un seul passage.

---

## 2. Ce que fait QuEST

Pour `n` échantillons et `p` variables (`c = p/n`), la fonction QuEST
`Q_{n,p}` envoie les valeurs propres de population `τ = (t_1,…,t_p)` sur les
valeurs propres empiriques limites `λ = (q_1,…,q_p)`, chaque `q_i` étant la
moyenne du quantile sur le i-ème intervalle de masse `1/p` :

```
q_i = p ∫_{(i-1)/p}^{i/p} (F^τ)^{-1}(v) dv
```

où `F^τ` est la loi empirique limite des valeurs propres d'échantillon, dont la
transformée de Stieltjes `m = m^τ_{n,p}(z)` est l'unique solution de l'équation
de Marčenko–Pastur :

```
m = (1/p) Σ_i 1 / ( t_i (1 - c - c z m) - z ).          (MP)
```

L'implémentation officielle travaille dans l'espace `u = -1/m_Fbar(z)` et
procède en six étapes (support, grille, résolution de (MP) sur la grille,
densité, CDF, interpolation/quantiles). Sur un point de grille réel `ξ = Re(u)`
on résout en `y = Im(u) ≥ 0` :

```
1 - c Σ_k w_k t_k² / ((t_k - ξ)² + y²) = 0,
u = ξ + iy,
z = u - c u m_LF(u),   m_LF(u) = Σ_k w_k t_k / (t_k - u),
f(z) = Im(u) / (π c |u|²),          densité d'échantillon
d(z) = z / |1 - c m_LF(u)|².        shrinkage optimal (sortie `d`)
```

**Usage.** LW estiment `τ` en **inversant** `Q_{n,p}` numériquement
(optimiseur non linéaire, Jacobien analytique fourni par le papier) ; c'est
l'estimateur « QuEST inverse shrinkage ».

---

## 3. Ce que fait `shrinkers`

`shrinkers.deconvolve_spiked` / `estimate_population_eigenvalues` font le
chemin inverse, en trois étapes : détection de spikes (BEMA) → débiaisage
BBP inverse (DGJ) → déconvolution du bulk. Le bulk utilise la formule RIE /
Ledoit–Wolf ponctuelle

```
ξ(λ_i) = λ_i / |1 - c + c λ_i m_g(λ_i - iη)|²,   m_g(z) = (1/p) Σ_j 1/(z - λ_j)
```

calculée par un noyau de Stieltjes rapide (bloqué exact O(p²), ou treecode
ChebCode O(p log p)) ; `shrink_eigenvalues` ajoute la remise à l'échelle
préservant la trace.

---

## 4. Correspondance exacte des deux noyaux

Posons `a(z) = 1 - c - c z m_F(z)` où `m_F` est la transformée de Stieltjes
de la loi d'échantillon (convention `∫dF(λ)/(λ-z)`). La transformée « barre »
de Ledoit–Wolf vaut
`m_Fbar(z) = (c-1)/z + c m_F(z) = -a(z)/z`. Donc

```
u = -1/m_Fbar(z) = z / a(z).
```

Or `shrinkers` définit précisément `a = 1 - c - c z g(z)` et `w = z / a(z)`
avec `g = m_g = -m_F` (la déconvolution El Karoui). **Le `u` de QuEST et le `w`
de `shrinkers` sont la même variable.**

En injectant `z = a u` dans (MP) :

```
m_F = m_H(u)/a,     a = (1 - c) - c u m_H(u),     z = u a.
```

QuEST résout le problème dans le sens `u → z` (donnée : `H`, population) ;
`shrinkers` le résout dans le sens `z → u → H` (donnée : le spectre empirique).
Ce sont bien deux directions de la **même** équation.

**Identité des formules de shrinkage.** QuEST : `d = x / |1 - c m_LF|²`.
Or `m_LF(u) = Σ w t/(t-u) = 1 + u m_H(u)`, donc
`1 - c m_LF = 1 - c - c u m_H(u) = a(z) = 1 - c - c z m_F(z)`.
Et comme `m_g = -m_F`, le dénominateur de `shrinkers`
`1 - c + c λ m_g` vaut `1 - c - c λ m_F = a` — la même quantité.
Donc **`ledoit_wolf_shrinkage` calcule la même fonction que le `d` de QuEST** ;
la seule différence est la source de `m_F` (résolution modèle depuis `τ` vs
moyenne empirique sur `λ`).

---

## 5. Validation du port QuEST

Avant de comparer, le port a été validé de deux façons :

| Test | Résultat |
|---|---|
| Densité vs formule fermée MP (`τ = I`) pour c = 0,1/0,25/0,5/1 | erreur rel. max **5,5·10⁻¹²** ; `d ≡ 1` exactement |
| CDF empirique Monte-Carlo (120 répl., p=1000, c=0,25, τ spiké) vs `F` QuEST | erreur abs. max **3,6·10⁻⁴** (bruit MC attendu ~1,4·10⁻³) |
| Spectre lisse (τ linéaire 0,5→2) vs MC | erreur abs. max **3,7·10⁻⁴** sur la CDF |
| Top-3 quantiles vs formule BBP fermée `ℓ(1+c/(ℓ-1))` | 12,2776 / 7,2890 / 4,3286 vs 12,2727 / 7,2917 / 4,3333 |

Le port est donc fidèle.

---

## 6. Les calculs coïncident-ils ?

### 6.1 La formule de shrinkage (tableau A)

Population à deux blocs `[3 × p/2, 1 × p/2]` (support d'échantillon **continu**,
donc `d` bien défini partout), `η = 0,1/√p`, on compare le `d` oracle de QuEST
au `ξ` empirique de `shrinkers` (interprété via `stieltjes_transform`) :

| p | erreur rel. médiane | 90ᵉ centile | erreur rel. max |
|---|---|---|---|
| 2 000 | 3,0·10⁻³ | 1,7·10⁻² | 3,9·10⁻² |
| 5 000 | 1,4·10⁻³ | 3,5·10⁻³ | 2,6·10⁻² |
| 10 000 | 7,6·10⁻⁴ | 1,4·10⁻³ | 2,1·10⁻² |
| 20 000 | **3,7·10⁻⁴** | 8,8·10⁻⁴ | 1,7·10⁻² |

Décroissance nette en `p`. Le maximum reste à ~2 % et ne décroît pas : il est
localisé **aux bords du support** (et au bord intérieur), là où la densité a une
singularité en racine carrée et où les deux calculs sont mal conditionnés.
C'est un artefact de bord, pas un désaccord de formule.

### 6.2 Round-trip : `shrinkers` est-il l'inverse numérique de QuEST ? (tableau B)

`τ → QuEST → λ → shrinkers → τ̂`, avec spikes vrais `[12, 7, 4]` :

| p | spikes récupérés | erreur rel. spikes | moyenne bulk | ‖QuEST(τ̂) − λ‖∞ |
|---|---|---|---|---|
| 500 | 12,0087 / 6,9933 / 3,9884 | 2,9·10⁻³ | 1,000037 | 1,1·10⁻² |
| 1 000 | 12,0044 / 6,9967 / 3,9943 | 1,4·10⁻³ | 1,001166 | 5,1·10⁻³ |
| 2 000 | 12,0022 / 6,9984 / 3,9972 | 7,1·10⁻⁴ | 1,001200 | 2,7·10⁻³ |
| 4 000 | 12,0011 / 6,9993 / 3,9986 | 3,5·10⁻⁴ | 1,000983 | 2,2·10⁻³ |

C'est le résultat le plus fort : `shrinkers` récupère l'antécédent de QuEST, et
la recomposition `QuEST(τ̂)` redonne le spectre d'entrée. Les deux opérateurs se
composent bien en identité (à la discrétisation près, en `O(1/p)`).

### 6.3 Sur données finies réelles (tableau C)

p = 1000, c = 0,25, une réalisation MC : `shrinkers` ramène l'erreur moyenne
`|λ − τ|` de **0,4202 → 0,0592**, et `QuEST(τ̂)` recolle au spectre observé avec
`‖·‖∞ = 1,35·10⁻²` (`rel L2 = 6,5·10⁻³`). C'est exactement le résidu que
l'optimiseur de LW minimise — `shrinkers` le trouve sans optimiseur.

### 6.4 Réserves honnêtes

- **Atomes / séparation spectrale.** Pour un spike isolé, la loi d'échantillon
  porte un atome ; la formule ponctuelle `x/|1-c m_LF|²` n'est *pas* l'objet
  pertinent sur un atome (le dénominateur y est piloté par le pôle). QuEST
  n'évalue `d` que sur le support continu ; `shrinkers` traite ces atomes par
  une étape dédiée (BEMA + BBP inverse) qui les récupère à ~10⁻³. Comparer les
  deux *sur les atomes* n'a donc pas de sens ; la comparaison 6.1 est faite sur
  le support continu, et 6.2/6.3 valident les atomes via le pipeline hybride.
- **`η` vs discrétisation.** `shrinkers` régularise `m_g` par `η` ; QuEST prend
  `η → 0` sur la loi *continue*. Utiliser `η → 0` directement sur un spectre
  discret fait diverger `m_g` (les atomes discrets dominent) : il faut
  `η ≫` espacement moyen `~1/(p·f)`. C'est la raison du défaut `0,1/√p` et de
  la nécessité de faire grandir `p` à `η` fixé pour voir converger 6.1.

---

## 7. Runtime

Toutes les mesures : Apple M-series, 1 thread, médiane de 7 appels, port QuEST
en NumPy (vectorisé), `shrinkers` 0.1.2 (Rust + PyO3), p=500–8 000, c=0,25.

### 7.1 Coût d'un appel

| p | QuEST direct | `stieltjes_transform` | `ledoit_wolf_shrinkage` | `deconvolve_spiked` | `estimate_population_…` |
|---|---|---|---|---|---|
| 500 | 7,85 ms | 0,068 ms | 0,065 ms | 0,028 ms | 0,067 ms |
| 1 000 | 8,79 ms | 0,258 ms | 0,256 ms | 0,040 ms | 0,258 ms |
| 2 000 | 12,9 ms | 1,01 ms | 1,01 ms | 0,059 ms | 1,02 ms |
| 4 000 | 15,4 ms | 4,03 ms | 4,04 ms | 0,101 ms | 4,04 ms |
| 8 000 | 23,6 ms | 16,3 ms | 16,1 ms | 0,256 ms | 16,2 ms |

À p = 8 000, `stieltjes_transform(parallel=True)` passe de 16,2 ms à **3,4 ms**
(×4,8) ; `shrinkers` libère le GIL pendant le calcul.

### 7.2 Le vrai écart : inverser QuEST (tableau E)

LW n'utilisent pas QuEST tel quel : ils l'**inversent** par optimisation non
linéaire. Test : cible `λ = QuEST(τ)` déterministe, p = 100, on minimise
`‖QuEST(θ) − λ‖²` sur `θ = (ℓ₁,ℓ₂,ℓ₃,σ²)` (Nelder–Mead) :

| | valeur |
|---|---|
| évaluations QuEST nécessaires | **219** |
| temps total (inversion) | **1 661 ms** |
| `θ` récupéré | `[12.000, 7.000, 4.000, 1.000]` (exact) |
| `shrinkers.estimate_population_eigenvalues` | **0,0066 ms** |
| ratio | **≈ 2,5·10⁵** |

Le ratio brut est gonflé par le fait que mon port QuEST est en Python/NumPy
(~8 ms fixes par appel, contre du Rust compilé), mais l'asymétrie
**structurelle** est le vrai message : QuEST doit payer une boucle
d'optimisation (≈ 200 résolutions de (MP) sur toute une grille), là où
`shrinkers` résout le problème inverse en **un seul passage**. À nombre
d'évaluations comparable, le facteur resterait de l'ordre de 10²–10³.

---

## 8. Conclusion et recommandations

- **Même chose ?** Non : QuEST est l'opérateur **direct**, `shrinkers` son
  **inverse**. Mais ils visent le même estimand et partagent *exactement* le
  même noyau (équation (MP), changement de variable `u = w = z/a`, et la même
  formule de shrinkage optimale).
- **Les calculs coïncident ?** Oui sur le support continu (médiane 4·10⁻⁴ à
  p = 20 000, à la discrétisation et à `η` près), et le round-trip
  `QuEST → shrinkers` récupère la population à ~10⁻³. Les écarts restants sont
  des effets de bord (singularités en racine carrée) et la gestion des atomes,
  que `shrinkers` traite en plus.
- **Runtime ?** Un appel direct QuEST est plus cher qu'un noyau Stieltjes
  `shrinkers` (mais pas dans un rapport scandaleux) ; surtout, pour **estimer
  la population**, LW doivent inverser QuEST itérativement (~200 évaluations
  mesurées), tandis que `shrinkers` le fait en un passage — d'où un facteur
  observé ~10⁵.

**Recommandation docs.** Ajouter à `docs/internals.md` (ou au README, section
« Comparison ») un paragraphe « QuEST is the forward map » qui : (i) nomme QuEST
comme la référence *directe* ; (ii) explicite l'identité `u = w = z/a` et
l'égalité `1 - c m_LF = 1 - c + c z m_g` ; (iii) précise que le `d` de QuEST et
`ledoit_wolf_shrinkage` sont la même formule, l'une modèle, l'autre empirique.
Cela éviterait à un lecteur de croire que `shrinkers` réimplémente QuEST.

---

## Fichiers produits

| Fichier | Rôle |
|---|---|
| `quest_reference.py` | port NumPy fidèle de `QuEST.m` (6 étapes, support, u-space) |
| `validate_quest.py` | validation contre Marchenko–Pastur fermé + Monte-Carlo |
| `compare.py` | les 5 expériences A–E, écrit `results.json` |
| `results.json` | tous les chiffres bruts |
| `make_figure.py`, `quest_vs_shrinkers.png` | figure de synthèse (4 panneaux) |
| `QuEST.m` | code MATLAB de référence des auteurs (copie locale) |

*Références : [Ledoit & Wolf, arXiv:1601.05870](https://arxiv.org/abs/1601.05870)
· [QuEST.m](https://github.com/AndoBlando/LSS_Bootstrap/blob/master/QuEST.m).*
