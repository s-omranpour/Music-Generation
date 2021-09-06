# Music-Generation

This is a toolkit for symbolic music generation in pyTorch. You can generate tracks from scratch (unconditional) or continuation for a given track.

<br/>

## Features
- Using [pytorch_lightning](https://www.pytorchlightning.ai) as backend.
- Using [Deepmusic](https://github.com/s-omranpour/DeepMusic) for encoding/decoding MIDI files.
- Supporting different transformer models + vanilla rnn. (vanilla, linear, huggingface transformers)
- Supporting Compound Word and REMI representations. (see references)

<br/>

## Usage
Please see [main.ipynb](main.ipynb) to learn how to train a model and generate music!

<br/>

## Generated Samples
check out [assets](assets/) directory!

<br/>

## References
[1] [https://github.com/YatingMusic/compound-word-transformer](https://github.com/YatingMusic/compound-word-transformer)

[2] [https://github.com/YatingMusic/remi](https://github.com/YatingMusic/remi)

[3] [https://github.com/idiap/fast-transformers](https://github.com/idiap/fast-transformers)

[4] [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
