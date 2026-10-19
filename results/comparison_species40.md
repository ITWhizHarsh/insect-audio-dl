# Model comparison: species40

File-level scores on the official test split; mean ± sample sd over seeds.

| model               | variant   |   seeds |   params (M) | test macro-F1   | test acc      |   val macro-F1 |   test chunk macro-F1 |   recordist-unseen macro-F1 |   train time (s) |
|:--------------------|:----------|--------:|-------------:|:----------------|:--------------|---------------:|----------------------:|----------------------------:|-----------------:|
| MLP (handcrafted)   | -         |       5 |         0.22 | 0.609 ± 0.007   | 0.618 ± 0.008 |          0.635 |                 0.566 |                       0.532 |              134 |
| CNN (log-mel)       | -         |       3 |         1.19 | 0.797 ± 0.013   | 0.805 ± 0.011 |          0.813 |                 0.757 |                       0.732 |             3978 |
| CNN (log-mel)       | noaug     |       1 |         1.19 | 0.782           | 0.786         |          0.817 |                 0.746 |                       0.712 |             3196 |
| CNN (log-mel)       | nomixup   |       1 |         1.19 | 0.795           | 0.802         |          0.796 |                 0.754 |                       0.722 |             3125 |
| CNN (log-mel)       | shift     |       1 |         1.19 | 0.765           | 0.771         |          0.807 |                 0.738 |                       0.687 |             3234 |
| CRNN (CNN + BiLSTM) | -         |       3 |         1.61 | 0.673 ± 0.013   | 0.683 ± 0.013 |          0.699 |                 0.637 |                       0.65  |             3381 |
| CRNN (CNN + BiLSTM) | gru       |       1 |         1.41 | 0.718           | 0.724         |          0.751 |                 0.688 |                       0.655 |             3382 |
| AST (AudioSet)      | lite      |       3 |        85.45 | 0.831 ± 0.015   | 0.833 ± 0.015 |          0.8   |                 0.768 |                       0.775 |             2884 |
