import os
import pathlib
import gdown

URL_LINKS = {
    "model_best": "https://drive.google.com/file/d/1bduk4t_2lpBduSBpkycQydCaINyp4wFU/view?usp=sharing",
}

def download_best_model(pretrained_path=None):
    path = ""
    if os.path.isabs(pretrained_path):
        if os.path.exists(pretrained_path):
            return
    else:
        absolute_path = os.path.abspath(pretrained_path)
        if os.path.exists(absolute_path):
            return
        else:
            path = absolute_path

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    gdown.download(url=URL_LINKS[pathlib.Path(pretrained_path).stem], output=path, fuzzy=True)