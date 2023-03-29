# Bach-like music generation

## introduction

The project aim to create an artificial intelligence using the technologie of deep learning.

## tech spec

### input format

The input format will be `midi` file; it will be parsed using the python library `mido`.

### output format

The output format is not decided yet, but likely to be `midi` file. A simple graphical representation might be used as well.

### media format

The input music should be transformed into a format that can be understood by IA network. As the artical from Google suggested, it is important to reveal the relation between notes. For example, note $E1$ have a likelihood to be present with $E2$, this is information can be captured by the system.

Each note can be considered as an element in a $\mathbb{Z}^3$ plane: it is defined by the start time $t_s$, the ending time $t_e$ and the pitch $p$. (Knowing that the strength of note is relatively less important and not taken into consideration.)

After the first parse through, we can then sort the ensemble of notes $(n_i)_{i \in \mathbb{N}}$. Then for each and every note we can find its nearest neighbors within a distance of $r$. The mesure of distance is defined by the difference of time:

$$
d = min(
    | t_e (n_1) - t_s(n_2) |,\\
    | t_e (n_2) - t_s(n_1) |
)
$$

We now have all the notes that is close to a given note, which will be the information feeding the network.
