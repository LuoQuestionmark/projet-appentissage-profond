# Bach-like music generation

## introduction

The project aim to create an artificial intelligence using the technologie of deep learning.

## tech spec

### input format

The input format will be `midi` file; it will be parsed using the python library `mido`.

### output format

The output format is not decided yet, but likely to be `midi` file. A simple graphical representation might be used as well.

### medium format (extract from midi file)

The input music should be transformed into a format that can be understood by IA network. As the article from Google suggested, it is important to reveal the relation between notes. For example, note $E1$ have a likelihood to be present with $E2$, this is information can be captured by the system.

Each note can be considered as an element in a $\mathbb{Z}^3$ plane: it is defined by the start time $t_s$, the ending time $t_e$ and the pitch $p$. (Knowing that the strength of note is relatively less important and not taken into consideration.)

After the first parse through, we can then sort the ensemble of notes $(n_i)_{i \in \mathbb{N}}$. Then for each and every note we can find its nearest neighbors within a distance of $r$. The measure of distance is defined by the difference of time:

$$
d = min(
    | t_e (n_1) - t_s(n_2) |,\\
    | t_e (n_2) - t_s(n_1) |
)
$$

We now have all the notes that is close to a given note, which will be the information feeding the network.

### medium format (input to the network)

Ideally, the network should be able to generate new note from existing notes. Assuming at a given moment $t$, there are already note $n_1$, $n_2$, $n_3$, the network should take all these values as input, then claim that a new note $n_4$ can be inserted. This can be done since it has the conditional probability:

$$
P(n_4 | n_1, n_2, n_3)
$$

Of course, this model is over simplified, it does not take the "subject" of music into consideration at all; this should be done, if the time allows, with a second model that can generate/identify the likelihood of a generated sheet, i.e. "How Bach it is".

At this moment, the first network will be defined in the following way:

- input: about 128 nodes, with each one represent a certain pitch. The value 0/1 indicate if a certain note exists;

- output: about 128 nodes, with each one represent the likelihood of existing of another note at a certain pitch (from 0 to 1);

- a random noise should be put as input as well, for a better generation of new music.

## first model: the note auto-completion

The first model I created is to auto-complete the missing note(s) basing on the known notes. The network works as a bayers model: by giving the existing notes  $n_1, n_2, \cdots$ as input, the model is capable of finding the most possible note in such situation, i.e. the note $n_p$ where:

$$
\forall i \in [1, n], P(n_p|n_1, n_2, \cdots) \geq P(n_p|n_1, n_2, \cdots)
$$.

## second model: the rythme auto-completion

The second model is to predict the length of a note basing on existing notes. We can encode the length of a music note $n$ as a number $d(n)$. Knowing a suite of notes with their length $d(n_1), d(n_2), \cdots, d(n_i)$, the model should be able to predict the length of the next note $d(n_{i+1})$.
