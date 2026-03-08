# Actual Running Output Example

Below is the actual output from a successful run of `cf-code-example.py`. Your results will vary slightly due to random initialization.

---

## Training Progress

The loss decreases rapidly at first, then stabilizes:

```
epoch=   1  data_loss=1164.1611  total_loss=1164.1727
epoch=  10  data_loss=142.2822   total_loss=143.6097
epoch=  50  data_loss=6.8279     total_loss=8.8732
epoch= 100  data_loss=3.2622     total_loss=5.3931
epoch= 500  data_loss=0.4519     total_loss=2.6927
epoch=1000  data_loss=0.4003     total_loss=2.6417
epoch=2000  data_loss=0.3432     total_loss=2.5929
epoch=5000  data_loss=0.3259     total_loss=2.5791
```

**What to notice**:
- `data_loss` (reconstruction error on training data) drops from ~1164 to ~0.3
- `total_loss` includes regularization — stays slightly higher
- Convergence happens around epoch 1000-2000; later epochs show diminishing returns

---

## Observed Matrix (Training Data)

Shows which ratings were used for training (`5`, `4`, etc.) vs. hidden for testing (`None`):

```
Observed matrix I0 (None = intentionally hidden for testing):
          Shrek    ToyStory FindingN Memento  Se7en    DarkKnig Inceptio Matrix   Interste Avengers Up       PulpFict
Alice           5     None        5        1        1        2        2        2        3        3        5        1
Bob             5        4        5        1        2        2        2        1        2        3     None        1
Cara            1        2        1        5        5        4        4        2        3        2        1     None
Dan             1        1        2        5        5     None        4        3        3        2        1        4
Eve             2        2        2        3        2        4        5     None        5        4        2        2
Frank           2        1        2        3        2        4        5        5        4     None        1        2
Grace           4        4        4     None        2        3        3        3        4        4        4        2
Hank            2        2        2        4     None        5        4        4        4        3        2        4
Ivy             3        3        4        3        2        3        4        4     None        4        4        2
Jake            3     None        3        4        4        4        4        3        3        3        2        5
```

**Pattern check**: Alice has high ratings for family movies (Shrek, FindingNemo, Up) and low for crime (Memento, Se7en). The model must learn this without seeing ToyStory (hidden).

---

## Predicted Matrix (All Entries)

Shows model predictions for ALL user-item pairs, including the hidden ones:

```
Reconstructed (predicted) matrix I_hat = P·U:
          Shrek    ToyStory FindingN Memento  Se7en    DarkKnig Inceptio Matrix   Interste Avengers Up       PulpFict
Alice           4.8      3.8      5.1      0.9      1.1      2.0      1.9      2.0      3.0      3.2      4.9      1.0
Bob             5.0      4.0      4.9      1.0      2.0      1.9      2.0      1.0      2.0      2.9      4.1      1.0
Cara            0.9      2.0      1.1      4.9      5.0      4.0      3.9      2.1      3.0      2.1      1.0      3.7
Dan             1.1      1.0      1.9      5.0      4.9      4.1      4.1      2.9      3.0      2.0      1.1      4.0
Eve             2.0      2.0      2.0      3.1      2.0      4.0      4.9      4.9      4.9      4.0      2.0      2.0
Frank           1.9      1.0      2.0      2.9      2.1      4.0      5.0      4.9      4.0      3.5      1.0      2.0
Grace           4.0      3.9      3.9      2.0      1.9      3.0      3.1      2.9      4.0      4.0      4.0      2.0
Hank            2.1      2.0      2.0      4.0      3.5      4.9      4.0      4.0      4.0      3.0      2.0      4.0
Ivy             3.1      3.0      3.9      2.9      2.0      3.0      3.9      4.0      4.8      3.9      3.9      2.0
Jake            3.0      2.4      3.0      4.0      4.0      4.0      3.9      3.0      3.0      3.0      2.0      4.9
```

**Check predictions on hidden entries**:
- Alice/ToyStory (true=5): predicted 3.8 — underestimated but correct direction (high)
- Bob/Up (true=5): predicted 4.1 — close!
- Dan/DarkKnight (true=5): predicted 4.1 — crime lover correctly predicted high

---

## Evaluation on Hidden Test Entries

The critical test: how well did the model generalize to unseen data?

```
Hidden-entry evaluation:
User         Movie               True       Pred     AbsErr
Alice        ToyStory               5      3.888      1.112
Bob          Up                     5      4.152      0.848
Cara         PulpFiction            5      3.600      1.400
Dan          DarkKnight             5      4.073      0.927
Eve          Matrix                 5      4.931      0.069
Frank        Avengers               5      3.489      1.511
Grace        Memento                2      2.147      0.147
Hank         Se7en                  4      3.466      0.534
Ivy          Interstellar           5      4.789      0.211
Jake         ToyStory               2      2.495      0.495

MAE on hidden entries : 0.7253
RMSE on hidden entries: 0.8750
```

**Interpreting the metrics**:
- **MAE** (Mean Absolute Error): On average, predictions are off by ~0.7 stars. Good for a 1-5 scale!
- **RMSE** (Root Mean Square Error): Penalizes large errors more. ~0.9 indicates most errors are small.
- Eve/Matrix (error 0.069): Near-perfect prediction
- Frank/Avengers (error 1.511): Largest miss, but still correctly predicted "high" rating

---

## Learned Latent Vectors

These are the internal representations the model discovered:

```
Learned P (users x factors):
  Alice       : ['-1.22', '+0.89', '+0.42', '-1.31', '-0.30', '-0.85', '-0.64']
  Bob         : ['-1.29', '+0.59', '+0.38', '-1.57', '+0.55', '-0.66', '-0.31']
  Cara        : ['-1.27', '-0.59', '+1.22', '+0.34', '+0.75', '+0.82', '-0.69']
  Dan         : ['-1.70', '-0.93', '+0.68', '+0.66', '+0.30', '+0.06', '-0.66']
  Eve         : ['-1.37', '+0.09', '-0.14', '-0.28', '-0.64', '+1.14', '-1.18']
  ...

Learned item vectors from U (items as columns):
  Shrek       : ['-1.20', '+0.73', '-0.00', '-1.39', '+0.42', '-0.64', '-0.77']
  ToyStory    : ['-0.68', '+0.62', '+0.92', '-1.25', '+0.28', '-0.02', '-0.85']
  FindingNemo : ['-1.50', '+0.20', '+0.26', '-1.30', '-0.19', '-1.00', '-0.55']
  Memento     : ['+0.65', '+1.55', '-0.88', '-0.86', '+0.21', '+0.44', '-0.86']
  Se7en       : ['+0.58', '+1.62', '-0.90', '-0.67', '+0.95', '+0.19', '-0.52']
  DarkKnight  : ['+0.40', '+1.40', '-1.86', '+0.50', '+0.40', '+0.51', '-1.26']
  Inception   : ['+0.54', '+1.13', '-1.19', '-0.61', '-0.35', '+0.33', '-1.06']
  Matrix      : ['+0.13', '+0.80', '-1.57', '-0.55', '-0.88', '+0.62', '-1.31']
  ...
```

**Pattern in the learned factors**:
- Alice and Bob have similar vectors (both family lovers)
- Shrek, ToyStory, FindingNemo have similar patterns (family movies cluster together)
- Memento, Se7en, DarkKnight cluster (crime/thriller)
- Matrix, Inception, Interstellar cluster (sci-fi)

These 7-dimensional vectors capture taste profiles that the model learned purely from ratings — no movie genre metadata was provided!

### Example: Predicting Alice's Rating for ToyStory

The model predicts a rating by taking the dot product of the user latent vector and the item latent vector:

$$
\hat{r}_{ui} = P_u^\top U_i
$$

For **Alice**:

$$
P_{\text{Alice}} = [-1.219,\; 0.893,\; 0.422,\; -1.308,\; -0.297,\; -0.848,\; -0.640]
$$

For **ToyStory**:

$$
U_{\text{ToyStory}} = [-0.680,\; 0.622,\; 0.917,\; -1.254,\; 0.281,\; -0.023,\; -0.847]
$$

So the prediction is:

$$
\hat{r}_{\text{Alice},\text{ToyStory}}
=
(-1.219)(-0.680)
+ (0.893)(0.622)
+ (0.422)(0.917)
+ (-1.308)(-1.254)
+ (-0.297)(0.281)
+ (-0.848)(-0.023)
+ (-0.640)(-0.847)
$$

Term by term:

$$
\hat{r}_{\text{Alice},\text{ToyStory}}
=
0.829
+ 0.555
+ 0.387
+ 1.640
- 0.083
+ 0.020
+ 0.542
$$

Final result:

$$
\hat{r}_{\text{Alice},\text{ToyStory}} \approx 3.890
$$

So the model predicts that **Alice would rate ToyStory about 3.89**.
