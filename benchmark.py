import argparse
import time
import numpy as np
import torch

from matcher import LightGlue, SuperPoint
from matcher.utils import load_image
from matcher.backbone_adapter import BackboneKeypointExtractor
from models import SimCLRModel
from utils import load_checkpoint


def measure(matcher, data, device, r=100):
    timings = np.zeros((r, 1))
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        _ = matcher(data)
    with torch.no_grad():
        for rep in range(r):
            if device.type == "cuda":
                starter.record()
                _ = matcher(data)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
            else:
                start = time.perf_counter()
                _ = matcher(data)
                curr_time = (time.perf_counter() - start) * 1e3
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / r
    std_syn = np.std(timings)
    return {"mean": mean_syn, "std": std_syn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone_ckpt", default=None)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--num_keypoints", nargs="+", type=int, default=[256, 512, 1024])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    print(f"Benchmarking on: {device}")

    extractor = SuperPoint(max_num_keypoints=None, detection_threshold=-1)
    extractor = extractor.eval().to(device)

    matcher = LightGlue(features="superpoint")
    matcher = matcher.eval().to(device)

    dummy0 = torch.randn(1, 3, 480, 640).to(device)
    dummy1 = torch.randn(1, 3, 480, 640).to(device)

    for num_kpts in args.num_keypoints:
        extractor.conf.max_num_keypoints = num_kpts
        with torch.no_grad():
            feats0 = extractor({"image": dummy0[:, :1]})
            feats1 = extractor({"image": dummy1[:, :1]})

        data = {"image0": feats0, "image1": feats1}
        result = measure(matcher, data, device, r=args.repeat)
        print(f"  kpts={num_kpts}: {result['mean']:.1f} +/- {result['std']:.1f} ms")


if __name__ == "__main__":
    main()
