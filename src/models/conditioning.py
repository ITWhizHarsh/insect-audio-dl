"""Temperature conditioning: FiLM on top of any of our classifiers.

Insects are ectotherms, so their song changes with ambient temperature: the
chirp rate of a tree cricket is close to a linear function of temperature
(Dolbear 1897), and in Orthoptera generally pulse rate and carrier frequency
both drift with it. The same species can therefore sound different at 15 C
and at 30 C. If the model is told the temperature it could, in principle,
undo that shift instead of learning every temperature variant separately.

18.7% of InsectSet459 files (4,906 of 26,297, all of them Orthoptera) carry
a recorded temperature; for the rest the field is empty. Each file gets two
numbers: the temperature standardised with the training-split mean / std (0
when missing) and a missing flag (1 = missing). FiLM (Perez et al., AAAI
2018) turns these into a per-dimension scale and shift of the backbone's
pooled embedding, h * (1 + gamma) + beta, before the usual dropout + linear
head. The last FiLM layer starts at zero, so at initialisation the model is
exactly the unconditioned head and any use of temperature has to be learned.

The confound. analysis/temperature_audit.py shows the field is entered by
very few people: 5 of the 87 recordists who give a temperature supply 92% of
the values, and most active recordists record it on nearly all or nearly
none of their files. Whether a file has a temperature, and often its value,
therefore says who recorded it and where. A model can use that as a
recordist / region shortcut, which would raise accuracy without any link to
song physiology. So the experiment compares three runs on identical files,
splits and seeds: no temperature (the plain backbone), true temperature, and
shuffled temperature (src/train.py --temperature shuffled permutes the values
among the training files that have one, so every missing flag stays where it
was). Shuffling keeps the distribution and the missingness pattern but breaks
the link between a file and its temperature. A gain of true over shuffled
says the value carries information; a gain that also appears with shuffled
values comes from the missing flag or the extra parameters. Information is
not yet physiology, because a recordist's usual temperatures still point to
their site and season. The recordist-unseen test subset is the second check:
a physiological effect should hold on recordists the model never saw in
training, a shortcut should not.
"""
import numpy as np
import torch.nn as nn

from src.data.recordist import TEMP_RANGE   # same plausibility limits as the audit (two entries of 60 C are errors)


def temperature_inputs(temps, mean, std):
    """Temperatures in C (NaN = missing) -> float32 (N, 2): [standardised value, missing flag]."""
    t = np.asarray(temps, dtype=np.float64)
    missing = np.isnan(t) | (t < TEMP_RANGE[0]) | (t > TEMP_RANGE[1])
    z = np.where(missing, 0.0, (t - mean) / std)
    return np.stack([z, missing], axis=1).astype(np.float32)


class TemperatureConditioned(nn.Module):
    def __init__(self, backbone, n_classes, hidden=64, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.film = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, 2 * self.embed_dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.embed_dim, n_classes)

    def embed(self, x, temp):                  # temp: (B, 2) from temperature_inputs
        h = self.backbone.embed(x)
        gamma, beta = self.film(temp).chunk(2, dim=1)
        return h * (1 + gamma) + beta

    def forward(self, x, temp):
        return self.fc(self.dropout(self.embed(x, temp)))
