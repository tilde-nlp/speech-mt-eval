# Speech MT Evaluation

This repository contains scripts for multi-layered evaluation of cascade speech translation systems. The goal of the
multi-layered evaluation approach is to determine the error contribution for each component of the system.

## Methodology

The evaluation is split into three steps:
- Automatic speech recognition (ASR)
- Segmentation
- Machine translation (MT)

For ASR evaluation we use word error rate (WER) and character error rate (CER) metrics from [JiWER](https://github.com/jitsi/jiwer).
For segmentation evaluation we use segmentation error rate (SER), which measures the proportion between the length of
overlapping reference/hypothesis segments and the total length of reference segments. Segment lengths are calculated
based on their timestamps in the speech. See the full implementation in [metrics.py](https://github.com/tilde-nlp/speech-mt-eval/blob/main/metrics.py).
Part of the implementation is taken from [SimpleDER](https://github.com/wq2012/SimpleDER/tree/master).
For MT evaluation we use translation error rate (TER), ChrF++, and BLEU metrics from [sacreBLEU](https://github.com/mjpost/sacrebleu)
and the COMET metric from [COMET](https://github.com/Unbabel/COMET).

Given that the hypothesis segmentation can differ from the reference segmentation, we use [mwerSegmenter](https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz)
to align the hypothesis to the reference.

## Setup

The required dependencies can be installed with pip:
```
pip install -r requirements.txt
```

Download [mwerSegmenter](https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz)

## Usage

The evaluation script can be run as follows:
```
python evaluate.py /path/to/ref_dir /path/to/hyp_dir
```

The directories should have the following structure:
```
.
├── asr
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
├── mt
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
└── segmentation
    ├── 1.json
    ├── 2.json
    └── ...
```

Each file represents a separate audio file. The TXT files in `asr` and `mt` folders should contain
transcriptions/translations separated by newlines according to the segmentation:
```text
This is the transcription of the first segment.
This is the transcription of the second segment.
```
The JSON files in `segmentation` folder should contain an array of the segment start/end timestamps:
```json
[
  [5.64, 7.39],
  [9.21, 11.42]
]
```

## Acknowledgements

This is the prototype created in activity 3.2 of the project "AI Assistant for Multilingual Meeting Management" (No. of the Contract/Agreement: 1.1.1.1/19/A/082).
