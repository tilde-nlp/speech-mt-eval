import os
import tempfile
import subprocess
from typing import List

from utils import remove_nonalphanumeric


def align_mwer_segmenter(ref: List[str], hyp: List[str], mwer_segmenter_path: str):
    """
    Re-segments hyp file according to ref segmentation with mwerSegmenter.
    :param ref: list of reference translations
    :param hyp: list of hypothesis translations
    :param mwer_segmenter_path: path to the mwerSegmenter executable
    :return:
    """
    temp_dir = tempfile.TemporaryDirectory()

    clean_ref = [" ".join(remove_nonalphanumeric(l.lower()).split()) + "\n" for l in ref]
    clean_hyp = [" ".join(remove_nonalphanumeric(l.lower()).split()) + "\n" for l in hyp]

    ref_path = os.path.join(temp_dir.name, "ref.txt")
    with open(ref_path, "w", encoding="utf-8") as f:
        f.writelines(clean_ref)

    hyp_path = os.path.join(temp_dir.name, "hyp.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.writelines(clean_hyp)

    p = subprocess.Popen(f"{mwer_segmenter_path} -hypfile {hyp_path} -mref {ref_path}", shell=True)
    ret_code = p.wait()
    if ret_code != 0:
        raise RuntimeError(f"Failed to align hyp\nmwerSegmenter return code: {ret_code}")

    with open("__segments", "r", encoding="utf-8") as f:
        aligned_hyp = f.readlines()

    # restore punctuation and capitalization
    hyp_words = " ".join(hyp).split()
    w_idx = 0
    clean_aligned_hyp = []
    for l in aligned_hyp:
        words = l.split()
        clean_words = []
        for w in words:
            if all([not c.isalpha() for c in hyp_words[w_idx]]):  # word does not contain letters
                clean_words.append(hyp_words[w_idx])
                w_idx += 1

            if remove_nonalphanumeric(w.lower()) == remove_nonalphanumeric(hyp_words[w_idx].lower()):
                clean_words.append(hyp_words[w_idx])
                w_idx += 1
                continue
            else:
                raise RuntimeError("Word mismatch after alignment")

        clean_aligned_hyp.append(" ".join(clean_words))

    temp_dir.cleanup()

    return clean_aligned_hyp
