from flask import Flask, render_template, Response, request
import json
import os
import pathlib
from imageio_ffmpeg import get_ffmpeg_exe
import imageio
import numpy as np
from skimage.transform import resize
from face_alignment import FaceAlignment, LandmarksType
from scipy.optimize import least_squares
from PIL import Image
import regex
import torch
import subprocess
import hashlib
import sys
from demo import load_checkpoints
from animate import normalize_kp

app = Flask(__name__)

generator, kp_detector = load_checkpoints(
    config_path="first-order-model/config/vox-adv-256.yaml",
    checkpoint_path="vox-adv-cpk.pth.tar",
)
fa = FaceAlignment(LandmarksType._2D)


@app.route("/")
def index():
    return render_template("upload.html")


def data(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.route("/upload", methods=["POST"])
def upload():
    for key, file in request.files.items():
        pathlib.Path("static", key).mkdir(exist_ok=True)
        file.save(os.path.join("static", key, file.filename))
    return render_template(
        "progress.html",
        video=request.files["video"].filename,
        image=request.files["image"].filename,
    )


def get_video_length(filename):
    return sum(1 for _ in imageio.get_reader(f"static/video/{filename}").iter_data())


def hash_video(filename):
    BLOCK_SIZE = 1024 ** 2
    file_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        while True:
            fb = f.read(BLOCK_SIZE)
            if fb is not None:
                file_hash.update(fb)
            else:
                break
    return file_hash.hexdigest()


def generate_face_keypoints(filename):
    reader = imageio.get_reader(f"static/video/{filename}")
    stream = reader.iter_data()
    for img in stream:
        kp = fa.get_landmarks(img)
        if kp is not None:
            yield kp[0]
        else:
            yield None


def transform(x, points=None, return_transform=False):
    M = np.eye(3)
    scale = x[3]
    [theta] = x[:1]
    c, s = np.cos(theta), np.sin(theta)
    M[:2, :2] = scale * np.array(((c, -s), (s, c)))
    M[:2, 2] = x[1:3]

    if return_transform:
        return M
    else:
        X = M @ np.pad(points.T, ((0, 1), (0, 0)), "constant", constant_values=1)
        X = X[:2] / X[2]

        return X.T


def generate_alignment_candidates(image_keypoints, video_keypoints):
    for query in video_keypoints:
        if query is not None:

            def error(x, return_transform=False):
                return (transform(x, image_keypoints) - query).flatten()

            res = least_squares(error, np.ones((4,)), method="lm")
            yield res.x, res.cost
        else:
            yield None, np.inf


def generate_warped_video(filename, best_transform):
    for image in imageio.get_reader(f"static/video/{filename}").iter_data():
        yield np.array(
            Image.fromarray(image).transform(
                (256, 256),
                Image.AFFINE,
                data=transform(best_transform, return_transform=True).flatten()[:6],
                resample=Image.BICUBIC,
            )
        ).astype(np.float32) / 255


def generate_morphed_video(image, sub_video):
    with torch.no_grad():
        source = (
            torch.tensor(np.float32(image), device="cuda").permute(2, 0, 1).unsqueeze(0)
        )
        kp_source = kp_detector(source)

        for i, driving_frame in enumerate(sub_video):
            driving_frame = (
                torch.tensor(np.float32(driving_frame))
                .cuda()
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            if i == 0:
                kp_driving_initial = kp_detector(driving_frame)

            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(
                kp_source=kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=True,
                use_relative_jacobian=True,
                adapt_movement_scale=True,
            )
            yield generator(source, kp_source=kp_source, kp_driving=kp_norm)[
                "prediction"
            ].squeeze().permute(1, 2, 0).cpu().numpy()


@app.route("/progress/<video>/<image>")
def progress(video, image):
    def generate():
        N = get_video_length(video)

        # video keypoints
        driving_keypoints = []
        for i, kp in enumerate(generate_face_keypoints(video)):
            driving_keypoints.append(kp)
            yield data(
                {
                    "type": "update",
                    "field": "extract_keypoints",
                    "iteration": i,
                    "total": N + 1,
                }
            )

        # image keypoints
        source_image = resize(imageio.imread(f"static/image/{image}"), (256, 256))
        [source_keypoints] = fa.get_landmarks(source_image * 255)
        yield data(
            {
                "type": "update",
                "field": "extract_keypoints",
                "iteration": i + 1,
                "total": N + 1,
            }
        )

        # alignment optimization
        errors = []
        transforms = []
        for i, (trans, err) in enumerate(
            generate_alignment_candidates(source_keypoints, driving_keypoints)
        ):
            errors.append(err)
            transforms.append(trans)
            yield data(
                {
                    "type": "update",
                    "field": "compute_alignment",
                    "iteration": i,
                    "total": N,
                }
            )

        best_frame = np.argmin(errors)
        best_transform = transforms[best_frame]

        # warp video to align with image
        warped = []
        for i, img in enumerate(generate_warped_video(video, best_transform)):
            warped.append(img)
            yield data({"type": "update", "field": "warp", "iteration": i, "total": N})

        # morph
        morphed_forward = []
        for i, out in enumerate(
            generate_morphed_video(source_image, warped[best_frame:])
        ):
            morphed_forward.append(out)
            yield data(
                {"type": "update", "field": "morphed", "iteration": i, "total": N}
            )

        morphed_backward = []
        for j, out in enumerate(
            generate_morphed_video(source_image, warped[: (best_frame + 1)][::-1])
        ):
            morphed_backward.append(out)
            yield data(
                {"type": "update", "field": "morphed", "iteration": i + j, "total": N}
            )

        morphed = morphed_backward[::-1] + morphed_forward[1:]

        # figure out input video fps
        ffmpeg = get_ffmpeg_exe()
        p = subprocess.Popen(
            f"{ffmpeg} -i static/video/{video}".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, info = p.communicate()
        match = regex.search("([0-9\\.]+) fps", info.decode("utf8"))
        rate = float(match.groups()[0])

        # Video 1/3: write out soundless video
        soundless_video = f"static/video/soundless_{video}"
        imageio.mimwrite(
            soundless_video,
            [(x * 255).astype(np.uint8) for x in morphed],
            fps=rate,
            quality=10,
        )
        yield data({"type": "update", "field": "result", "iteration": 0, "total": 3})

        # Video 2/3: extract sound from original video
        base_video_name, _ = os.path.splitext(video)
        video_full = f"static/video/{video}"
        audio_full = f"static/video/audio_{base_video_name}.aac"
        os.system(f"{ffmpeg} -y -i {video_full} -vn -acodec copy {audio_full}")
        yield data({"type": "update", "field": "result", "iteration": 1, "total": 3})

        # Video 3/3: extract sound from original video
        audio_scale = 0.8
        result_video = f"static/video/result_{video}"
        os.system(
            f"{ffmpeg} -y -i {soundless_video} -i {audio_full} -af asetrate=48000*{audio_scale},atempo={1/audio_scale} -c:v copy -c:a aac {result_video}"
        )
        yield data({"type": "update", "field": "result", "iteration": 2, "total": 3})

        # all done, download video
        yield data({"type": "download", "url": result_video})

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="80")
