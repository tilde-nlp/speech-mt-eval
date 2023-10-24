import os
import glob
import json
import argparse

import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint

from align import align_mwer_segmenter
from metrics import calculate_wer, calculate_cer, calculate_ser


def evaluate_asr(ref_dir, hyp_dir):
    ref_paths = glob.glob(os.path.join(ref_dir, "asr", "*"))

    total_wer_edits = 0
    total_wer_tokens = 0
    total_cer_edits = 0
    total_cer_tokens = 0

    for ref_path in tqdm(ref_paths, desc="ASR"):
        file_name = ref_path.split(os.sep)[-1]
        hyp_path = os.path.join(hyp_dir, "asr", file_name)

        with open(ref_path, "r", encoding="utf-8") as f:
            refs = f.readlines()

        with open(hyp_path, "r", encoding="utf-8") as f:
            hyps = f.readlines()

        ref = " ".join([ref.strip() for ref in refs])
        hyp = " ".join([hyp.strip() for hyp in hyps])

        edits, tokens = calculate_wer(ref, hyp)
        total_wer_edits += edits
        total_wer_tokens += tokens

        edits, tokens = calculate_cer(ref, hyp)
        total_cer_edits += edits
        total_cer_tokens += tokens

    wer = (float(total_wer_edits) / float(total_wer_tokens)) * 100
    cer = (float(total_cer_edits) / float(total_cer_tokens)) * 100

    return wer, cer


def evaluate_segmentation(ref_dir: str, hyp_dir: str):
    ref_paths = glob.glob(os.path.join(ref_dir, "segmentation", "*"))

    sers = []
    for ref_path in tqdm(ref_paths, desc="Segmentation"):
        file_name = ref_path.split(os.sep)[-1]
        hyp_path = os.path.join(hyp_dir, "segmentation", file_name)

        with open(ref_path) as f:
            ref_segs = json.load(f)

        with open(hyp_path) as f:
            hyp_segs = json.load(f)

        ser = calculate_ser(ref_segs, hyp_segs)
        sers.append(ser)

    total_ser = np.average(sers) * 100
    return total_ser


def evaluate_mt(ref_dir: str, hyp_dir: str, mwer_path: str):
    ref_paths = glob.glob(os.path.join(ref_dir, "mt", "*"))

    ref_transcriptions = []
    ref_translations = []
    hyp_translations = []

    for ref_path in tqdm(ref_paths, desc="MT"):
        file_name = ref_path.split(os.sep)[-1]
        src_path = os.path.join(ref_dir, "asr", file_name)
        hyp_path = os.path.join(hyp_dir, "mt", file_name)

        with open(ref_path, "r", encoding="utf-8") as f:
            ref_mt = f.readlines()
            ref_translations += ref_mt

        with open(src_path, "r", encoding="utf-8") as f:
            ref_transcriptions += f.readlines()

        with open(hyp_path, "r", encoding="utf-8") as f:
            hyp_mt = f.readlines()

            aligned_hyp = align_mwer_segmenter(
                ref_mt,
                hyp_mt,
                mwer_path
            )

            hyp_translations += aligned_hyp

    ter = TER()
    chrf = CHRF(word_order=2)
    bleu = BLEU()

    ter_results = ter.corpus_score(hyp_translations, [ref_translations])
    chrf_results = chrf.corpus_score(hyp_translations, [ref_translations])
    bleu_results = bleu.corpus_score(hyp_translations, [ref_translations])

    # COMET
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)
    comet_data = [{
        "src": ref_transcriptions[i],
        "mt": hyp_translations[i],
        "ref": ref_translations[i]
    } for i in range(len(ref_transcriptions))]

    seg_scores, comet_score = model.predict(comet_data, batch_size=8, gpus=0)

    return ter_results.score, chrf_results.score, bleu_results.score, comet_score


def evaluate(ref_dir: str, hyp_dir: str, mwer_path: str):
    wer, cer = evaluate_asr(ref_dir, hyp_dir)
    ser = evaluate_segmentation(ref_dir, hyp_dir)
    ter_score, chrf_score, bleu_score, comet_score = evaluate_mt(ref_dir, hyp_dir, mwer_path)

    print("==== ASR")
    print(f"WER: {wer:.2f}")
    print(f"CER: {cer:.2f}")

    print("==== SER")
    print(f"SER: {ser:.2f}")

    print("==== MT")
    print(f"TER: {ter_score:.2f}")
    print(f"ChrF++: {chrf_score:.2f}")
    print(f"BLEU: {bleu_score:.2f}")
    print(f"COMET: {comet_score:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ref', '--ref_dir', type=str, required=True, help='path to reference data directory')
    parser.add_argument('-hyp', '--hyp_dir', type=str, required=True, help='path to hypothesis data directory')
    parser.add_argument('-mwer', '--mwer_path', type=str, required=True, help='path to mwerSegmenter executable')
    args = parser.parse_args()

    evaluate(args.ref_dir, args.hyp_dir, args.mwer_path)
