from typing import List, Tuple

import jiwer

from utils import clean_text, remove_nonalpha, fix_overlap


def calculate_wer(ref: str, hyp: str):
    ref = " ".join(remove_nonalpha(clean_text(ref.lower())).split())
    hyp = " ".join(remove_nonalpha(clean_text(hyp.lower())).split())

    if len(ref) == 0:
        return len(hyp), len(hyp)

    measures = jiwer.compute_measures(ref, hyp)
    edits = measures["substitutions"] + measures["deletions"] + measures["insertions"]
    tokens = measures["hits"] + measures["substitutions"] + measures["deletions"]
    return edits, tokens


def calculate_cer(ref: str, hyp: str):
    if len(ref) == 0:  # return length of characters
        return len(hyp), len(hyp)

    ref = " ".join(remove_nonalpha(clean_text(ref.lower())).split())
    hyp = " ".join(remove_nonalpha(clean_text(hyp.lower())).split())
    measures = jiwer.cer(ref, hyp, return_dict=True)
    edits = measures["substitutions"] + measures["deletions"] + measures["insertions"]
    tokens = measures["hits"] + measures["substitutions"] + measures["deletions"]
    return edits, tokens


def is_overlapping(segments: List[Tuple[float, float]]):
    """
    Checks if any of the segments are overlapping each other.
    :param segments:
    :return:
    """
    segments.sort(key=lambda seg: seg[0])
    for i in range(len(segments)-1):
        if segments[i+1][0] < segments[i][1]:
            return True

    return False


def get_intersecting(ref_segment: Tuple[float, float], hyp: List[Tuple[float, float]]):
    intersecting_segments = []
    for seg in hyp:
        if ref_segment[0] <= seg[0] <= ref_segment[1] or \
           ref_segment[0] <= seg[1] <= ref_segment[1]:
            intersecting_segments.append(seg)

    return intersecting_segments


def calculate_intersection(ref: Tuple[float, float], hyp: Tuple[float, float]):
    max_start = max(ref[0], hyp[0])
    min_end = min(ref[1], hyp[1])
    return max(0.0, min_end - max_start)


def compute_merged_total_length(ref, hyp):
    """
    Source: https://github.com/wq2012/SimpleDER/blob/master/simpleder/der.py
    Compute the total length of the union of reference and hypothesis.

    :param ref: a list of tuples for the ground truth, where each tuple is (speaker, start, end) of type
    (string, float, float)
    :param hyp: a list of tuples for the diarization result hypothesis, same type as `ref`
    :return: a float number for the union total length
    """
    merged = ref + hyp
    # Sort by start.
    merged = sorted(merged, key=lambda element: element[0])
    i = len(merged) - 2
    while i >= 0:
        if merged[i][1] >= merged[i + 1][0]:
            max_end = max(merged[i][1], merged[i + 1][1])
            merged[i] = (merged[i][0], max_end)
            del merged[i + 1]
            if i == len(merged) - 1:
                i -= 1
        else:
            i -= 1
    total_length = 0.0
    for element in merged:
        total_length += element[1] - element[0]
    return total_length


def calculate_ser(ref: List[Tuple[float, float]], hyp: List[Tuple[float, float]]):
    """
    Calculates the proportion between overlapping ref/hyp segments and the total length of ref segments.
    :param ref:
    :param hyp:
    :return:
    """
    fix_overlap(ref)
    fix_overlap(hyp)

    ref_length = sum([seg[1] - seg[0] for seg in ref])
    merged_total_length = compute_merged_total_length(ref, hyp)
    overlap_sum = 0
    for ref_seg in ref:
        intersecting_segments = get_intersecting(ref_seg, hyp)
        if len(intersecting_segments) > 0:
            largest_overlap = max([calculate_intersection(ref_seg, hyp_seg) for hyp_seg in intersecting_segments])
            overlap_sum += largest_overlap

    return (merged_total_length - overlap_sum) / ref_length
