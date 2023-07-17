import shutil
import subprocess


def align_mwer_segmenter(ref_path: str, hyp_path: str, output_path: str, mwer_segmenter_path):
    """
    Re-segments hyp file according to ref segmentation with mwerSegmenter.
    :param ref_path: path to the reference file
    :param hyp_path: path to the hypothesis file
    :param output_path: path to the output file where the result will be saved
    :param mwer_segmenter_path: path to the mwerSegmenter binary
    :return:
    """
    p = subprocess.Popen(f"{mwer_segmenter_path} -hypfile {hyp_path} -mref {ref_path}", shell=True)
    ret_code = p.wait()
    if ret_code != 0:
        raise RuntimeError(f"Failed to align hyp: {hyp_path}\nmwerSegmenter return code: {ret_code}")

    shutil.copyfile("__segments", output_path)
