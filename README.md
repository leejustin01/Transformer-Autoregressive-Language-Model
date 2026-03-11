# Transformer-Based Autoregressive Language Model

An autoregressive language model based on the Transformer architecture that generates text conditioned on a given prompt. The model is trained on the TinyStories dataset to produce coherent and creative story continuations.

## Overview

This project implements a Transformer-based autoregressive model for text generation. Given an input prompt, the model predicts the next token in the sequence repeatedly to generate full sentences or stories, leveraging self-attention to capture long-range dependencies in text.

## Features

- Transformer architecture for autoregressive text generation  
- Generates text based on input prompts  
- Trained on the TinyStories dataset for story generation  
- Handles variable-length sequences and context  

## Example

```bash
python generate.py

Prompt:
Once upon a time, there was a hog on a log

Output:
Once upon a time, there was a hog on a log. it was a very special log that it could make things better.  one day,
a little girl named lily went to the forest. she saw the log and thought it would be fun to play with it. she
picked up the log and started to play with it.  suddenly, the log started to move and lily was scared. she tried
to run away, but the log was too strong. she tried to get away from the log, but it was too fast.  lily was sad
and angry. she decided to go back to the log and play with the log. she was so happy that she had found something
to do. from that day on, she never went to the log again.
```

Acknowledgement

This project was developed as part of a university assignment for **CS 435 - Applied Deep Learning** at Oregon State University.
