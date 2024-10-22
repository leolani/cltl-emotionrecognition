# cltl-emotion-detection
Detects emotions in texts and in faces and annotates the signal with the emotion and sentiment labels and scores.

## Getting started

### Prerequisites

This repository uses Python >= 3.9

Be sure to run in a virtual python environment (e.g. conda, venv, mkvirtualenv, etc.)

### Installation

1. Create a virtual environment:
    ```commandline
    python -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    ```

3. In the root directory of this repo run

    ```bash
    pip install -r requirements.txt
    ```

### Usage


## Examples

Please take a look at the example scripts provided to get an idea on how to run and use this package. Each example has a
comment at the top of the script describing the behaviour of the script.

For these example scripts, you need

1. A repository on [GraphDB Free](http://graphdb.ontotext.com/) called `sandbox`. To run any example script you first
   have to launch GraphDB, and then you can run the example script.

2. To change your current directory to `./examples/`

3. Run some examples (e.g. `python carl.py`)


The **cltl-emotion-detection** has several APIs for usage:

Detects emotions in texts and in faces and annotations the signals with the emotion labels and scores.

1) Calling the emotion detection modules directly
2) Reading an interaction saved in the EMISSOR format and annotate this with emotions
3) Integrated in the Leolani event-bus platform

For calling the separate classifiers directly, each of the submodules need to be called as defined separately. 
For 2) and 3), a uniform wrapper is provided.

The annotations are pushed to the event bus and can be taken up for further processing:

1) respond directly to the emotion using the emotion_responder
2) create a capsule to add the emotion as a perspective to the episodic Knowledge Graph

## Integration in the Leolani event-bus

Both emotion extractors and the responder can be integrated in the event-bus and to generate annotations in EMISSOR  through a service.py that is included.
In the configuration file of the event-bus,the input and output topics need to specified as well as the emotion detectors.



### Emotion responder

A simple emotion responder is included that generates a direct text response to the emotion of a certain type (GO, emotic, Ekman, Sentiment) and given other properties such as score and source.



# 
We implemented two separate emotion detectors modules: 1) text based emotion extraction and 2) face emotion.

## Text based emotion extraction

We implemented three different extractors:

1) a BERT model fine-tuned with the 27 GO emotions provided by Google
2) a RoBERTa model fine-tuned with labeled utterances from the Friends sitcom (MELD data with Ekman emotions)
3) VADER-NLTK, which is a rule based system using a lexicon derived from Tweets annotated with sentiment values 

We provided a mapping from GO to Ekman and from GO to sentiment, as well as a mapping from Ekman to sentiment.
Each extractor thus minimally generates sentiment values, but possibly also Ekman labels and GO labels above a threshold.

## Face based emotion extraction

We implemented the "emotic" module for face detection in context. This module is trained using the emotic database with 26 emotions. 
The module determines the emotion on the basis of facial features, posture and the situation:

References:
- Kosti R., J.M. Alvarex, A. Recasens, and A. Paedriza, (2019), "Context based emotion recognition using emotic dataset", 
IEEE Transactions on patterns analysis and machine intelligence.
- http://sunai.uoc.edu/emotic/index.html
- https://github.com/rkosti/emotic
- https://github.com/Tandon-A/emotic

The classifier returns emotion labels with a confidence score but 
also with a so-called Continuous Dimension score for Arousal (strength), Valence (positive or negative) and Dominance:

- Mehrabian A. Framework for a comprehensive description and measurement of emotional states. Genet Soc Gen Psychol Monogr. 1995 Aug;121(3):339-61. PMID: 7557355.

Just as with the text based emotion, we map the emotic labels to Ekman labels and sentiment. 
