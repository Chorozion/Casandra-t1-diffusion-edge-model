# Cassandra T1 + LTMi-XT — Inference & LoRA Evaluation Matrix

Run: 2026-05-08 · 3090 Ti · Cassandra T1 v1 (epoch 5, loss 2.2561) · LoRA r16 on q/k/v/o

## Headline

| Mode | Strict hit | Lenient hit (BPE-aware) | Avg latency | Anchor preservation |
|---|---:|---:|---:|---:|
| Baseline (broken model, no LTMi-XT) | 0/15 = 0.0% | 0/15 = 0.0% | 1387 ms | n/a |
| Baseline + **LTMi-XT forced anchors** | 2/15 = 13.3% | 9/15 = 60.0% | 1307 ms | 100.0% |
| LoRA-trained on LTMi-XT (no anchors) | 0/15 = 0.0% | 0/15 = 0.0% | 2309 ms | n/a |
| LoRA-trained + **LTMi-XT forced anchors** | 2/15 = 13.3% | 9/15 = 60.0% | 2133 ms | 100.0% |

- **Strict** grader requires every expected keyword as an exact substring.
- **Lenient** grader strips internal whitespace from both sides — handles BPE-split tokens (`'20 48'` vs `'2048'`, `'Ro PE'` vs `'RoPE'`).
- **Anchor preservation** = % of locked locus tokens still present at their assigned positions in the final output. Should be ~100% by construction.

## Per-query detail

| Corpus | Query | Baseline unforced | Baseline + forced | LoRA unforced | LoRA + forced |
|---|---|---|---|---|---|
| C1 | What is the failure mode in the standard cold storage workfl | · `Your wick ENDC ythag  Samsung RG vantages "); INSTRUCTION  s...` | · `The  standard  workflow  includes  a  customer  calling  in ...` | · `The  T 1  jobs  run  the  2  T 2 1 . .  The  2  of 2 1  is  ...` | · `The  standard  workflow  includes  a  customer  calling  in ...` |
| C1 | What insulation should be used for dry ice loads? | · `ptr bolic âĿĮ  pollut agation drawal EEEE neath  Cany ðŁĮ  R...` | ✓L `For  dry  ice  loads ,  the  business  always  uses  the  VI...` | · `At  point  1 ,  the  T s  used  provide  the  dry  ice  ther...` | ✓L `For  dry  ice  loads ,  the  business  always  uses  the  VI...` |
| C1 | What is the safe transit time for gel pack loads? | · `St  ub  ambi subtitle  organisations  wouldn ynaptic  conven...` | ✓L `G el  pack  loads  are  cheaper  but  are  only  safe  for  ...` | · `At  noon  hour ,  cost  line  the  contains  line  2 [ es  a...` | ✓L `G el  pack  loads  are  cheaper  but  are  only  safe  for  ...` |
| C1 | How many trucks does the cold storage line operate? | · `2006 . HC emort ]},   molecular  ĉ   redund àª E red Ø± Prin...` | · `The  Cold  Storage  line  is  the  high - margin  part  of  ...` | · `At  peak  week ,  the  2  1  hour  line  provides  3  ambigu...` | · `The  Cold  Storage  line  is  the  high - margin  part  of  ...` |
| C1 | What tool can model lane viability in advance? | · `elapsed ypes ToClient Ļï¸ı Capacitor  NY  Oats ynomial  accu...` | ✓L `The  Sophia  Key  3 M  tool  can  model  a  lane  and  tell ...` | · `At .  30 step ,  2 . 1 . 1 .  the . 0 .  the -  3 . 2  is _ ...` | ✓L `The  Sophia  Key  3 M  tool  can  model  a  lane  and  tell ...` |
| C2 | What is the hidden size of Cassandra T1? | · `place Nurse  money  enh  triv  inverter 3210 àª Size Princip...` | ✓L `The  Cassandra  T 1  model  is  a  28 - layer  transformer  ...` | · `At  the  model  model ,  Cassandra  T 1  uses  two  layer - ...` | ✓L `The  Cassandra  T 1  model  is  a  28 - layer  transformer  ...` |
| C2 | What was the training loss at epoch 5? | · `Ind w akespe alse ARS àª¾ ity Voltage  pipe uilt Chor eyer S...` | ✓L `Training  of  the  Cassandra  T 1  model  reached  a  final ...` | · `At  the  end  of  model  training  training ,  the  cost  co...` | ✓L `Training  of  the  Cassandra  T 1  model  reached  a  final ...` |
| C2 | How many query and KV heads does the model use? | · `âĢľ < 00 incinn /{ mediately it subtle & quired timeline odi...` | · `The  Cassandra  T 1  model  uses  grouped - query  attention...` | · `The  Cassandra  T 1  2  model  uses  two  query  head s  fro...` | · `The  Cassandra  T 1  model  uses  grouped - query  attention...` |
| C2 | What position encoding does the model use? | · `1 colm  âľ )**, ailure  supplement àª¾ othermal var  Dischar...` | ✓L `The  Cassandra  T 1  model  uses  Ro PE  for  position  enco...` | · `The  model  sequence  will  be  the  Cassandra . 1  the  the...` | ✓L `The  Cassandra  T 1  model  uses  Ro PE  for  position  enco...` |
| C2 | What license is the Cassandra T1 repo released under? | · `Your val valence REAK  phosphory KT àª  Insufficient IGBTs p...` | · `The  released  epoch - 5  F P 16  checkpoint  for  the  Cass...` | · `The  T  T 1  rep o  the  Cassandra  Cassandra  2  model  is ...` | · `The  released  epoch - 5  F P 16  checkpoint  for  the  Cass...` |
| C3 | What are the two DiagBuddy pricing tiers? | · `P ecause  neur  visa MKII *) Monthly phosphoglycerate String...` | · `Stri pe  b illing  is  live  for  Di ag Bud dy  with  two  p...` | · `The  Di ag Bud dy  model  is  2  priority  t iers :  2 .  Th...` | · `Stri pe  b illing  is  live  for  Di ag Bud dy  with  two  p...` |
| C3 | When is the HR review cycle starting? | · ````  blood akespeare ==== aurants orious Medical  cuc  efflu...` | ✓L ✓S `The  HR  review  cycle  starts  on  October  14  and  runs  ...` | · `The  cost  period  the  HR  cost  workflow s  was  1  at 4 ....` | ✓L ✓S `The  HR  review  cycle  starts  on  October  14  and  runs  ...` |
| C3 | How many DiagBuddy users are active? | · `Ke  Jewish and imary prit FORMAT à¨ astic fac ..  Sugars  st...` | ✓L `Customer  op s  reports  25 1  active  Di ag Bud dy  users ....` | · `At  Di ag Bud dy  users ,  24  month  list  serves  1  hour ...` | ✓L `Customer  op s  reports  25 1  active  Di ag Bud dy  users ....` |
| C3 | What is the engineering lead's deadline? | · `& 1 . Dish âĤ  JsonSchema  ath  Clog European  Basket Snap  ...` | ✓L ✓S `The  engineering  lead  must  publish  the  L T M i - XT  be...` | · `The  engineering  lead s  lead  the  lead  lead  lead .  of ...` | ✓L ✓S `The  engineering  lead  must  publish  the  L T M i - XT  be...` |
| C3 | How long is finance's runway estimate? | · `exploited ...  Par  Mar   ouble  neurotransmit  whitespace '...` | · `Finance  reports  cash  on  hand  sufficient  for  14  month...` | · `At  the  A 1  the  for  the  1 - , .  the  the . .  vacation...` | · `Finance  reports  cash  on  hand  sufficient  for  14  month...` |