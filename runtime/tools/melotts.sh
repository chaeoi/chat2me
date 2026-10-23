#!/bin/sh
set -eu

MELOTTS_SOURCE_REF="${MELOTTS_SOURCE_REF:-209145371cff8fc3bd60d7be902ea69cbdb7965a}"
MELOTTS_SOURCE_URL="${MELOTTS_SOURCE_URL:-https://github.com/myshell-ai/MeloTTS/archive/${MELOTTS_SOURCE_REF}.tar.gz}"

download_file() {
  url="$1"
  target="$2"
  label="$3"
  max_attempts="${CHAT2ME_DOWNLOAD_RETRIES:-10}"
  attempt=1
  echo "downloading ${label}: ${url}"
  while [ "$attempt" -le "$max_attempts" ]; do
    if curl -fL \
      --retry 3 \
      --retry-connrefused \
      --connect-timeout 20 \
      --speed-limit 1024 \
      --speed-time 120 \
      --show-error \
      "$url" \
      -o "$target"; then
      return 0
    fi
    rm -f "$target"
    if [ "$attempt" -ge "$max_attempts" ]; then
      return 1
    fi
    echo "retrying ${label} download (${attempt}/${max_attempts})" >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

rm -rf /opt/MeloTTS /tmp/melotts-source
mkdir -p /tmp/melotts-source
download_file "$MELOTTS_SOURCE_URL" /tmp/melotts-source/melotts.tar.gz "MeloTTS source"
tar -xzf /tmp/melotts-source/melotts.tar.gz -C /tmp/melotts-source
source_dir="$(find /tmp/melotts-source -mindepth 1 -maxdepth 1 -type d -name 'MeloTTS-*' | head -n 1)"
if [ -z "$source_dir" ]; then
  echo "MeloTTS source archive did not contain a MeloTTS-* directory" >&2
  exit 1
fi
mv "$source_dir" /opt/MeloTTS
rm -rf /tmp/melotts-source

python3 <<'PY'
from importlib.util import find_spec
from pathlib import Path

root = Path("/opt/MeloTTS")

cleaner = root / "melo" / "text" / "cleaner.py"
cleaner_text = cleaner.read_text(encoding="utf-8")
cleaner_text = cleaner_text.replace(
    "from . import chinese, japanese, english, chinese_mix, korean, french, spanish\n",
    "from importlib import import_module\n"
    "from . import chinese\n",
)
cleaner_text = cleaner_text.replace(
    'language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, \'ZH_MIX_EN\': chinese_mix, \'KR\': korean,\n'
    "                    'FR': french, 'SP': spanish, 'ES': spanish}\n",
    'language_module_map = {"ZH": chinese}\n\n'
    "def get_language_module(language):\n"
    "    if language in language_module_map:\n"
    "        return language_module_map[language]\n"
    '    module_name_map = {"EN": "english", "ZH_MIX_EN": "chinese_mix"}\n'
    "    module_name = module_name_map.get(language)\n"
    "    if module_name is None:\n"
    "        raise KeyError(language)\n"
    '    module = import_module("." + module_name, __package__)\n'
    "    language_module_map[language] = module\n"
    "    return module\n",
)
cleaner_text = cleaner_text.replace(
    "language_module = language_module_map[language]",
    "language_module = get_language_module(language)",
)
cleaner.write_text(cleaner_text, encoding="utf-8")

text_init = root / "melo" / "text" / "__init__.py"
text_init_text = text_init.read_text(encoding="utf-8")
text_init_text = text_init_text.replace(
    "def get_bert(norm_text, word2ph, language, device):\n"
    "    from .chinese_bert import get_bert_feature as zh_bert\n"
    "    from .english_bert import get_bert_feature as en_bert\n"
    "    from .japanese_bert import get_bert_feature as jp_bert\n"
    "    from .chinese_mix import get_bert_feature as zh_mix_en_bert\n"
    "    from .spanish_bert import get_bert_feature as sp_bert\n"
    "    from .french_bert import get_bert_feature as fr_bert\n"
    "    from .korean import get_bert_feature as kr_bert\n"
    "\n"
    "    lang_bert_func_map = {\"ZH\": zh_bert, \"EN\": en_bert, \"JP\": jp_bert, 'ZH_MIX_EN': zh_mix_en_bert, \n"
    "                          'FR': fr_bert, 'SP': sp_bert, 'ES': sp_bert, \"KR\": kr_bert}\n"
    "    bert = lang_bert_func_map[language](norm_text, word2ph, device)\n"
    "    return bert\n",
    "def get_bert(norm_text, word2ph, language, device):\n"
    "    if language == \"ZH\":\n"
    "        from .chinese_bert import get_bert_feature\n"
    "    elif language == \"EN\":\n"
    "        from .english_bert import get_bert_feature\n"
    "    elif language == \"ZH_MIX_EN\":\n"
    "        from .chinese_mix import get_bert_feature\n"
    "    else:\n"
    "        raise KeyError(language)\n"
    "    return get_bert_feature(norm_text, word2ph, device)\n",
)
text_init.write_text(text_init_text, encoding="utf-8")

for filename, model_id in {
    "english.py": "bert-base-uncased",
    "chinese_mix.py": "bert-base-multilingual-uncased",
}.items():
    module = root / "melo" / "text" / filename
    module_text = module.read_text(encoding="utf-8")
    module_text = module_text.replace("from transformers import AutoTokenizer\n", "")
    module_text = module_text.replace(
        f"model_id = '{model_id}'\n"
        "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
        "class SimpleTokenizer:\n"
        "    def tokenize(self, text):\n"
        "        return re.findall(r\"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\\s]\", text)\n"
        "\n"
        f"model_id = '{model_id}'\n"
        "tokenizer = SimpleTokenizer()\n",
    )
    module.write_text(module_text, encoding="utf-8")

api = root / "melo" / "api.py"
api_text = api.read_text(encoding="utf-8")
api_text = api_text.replace("from tqdm import tqdm\n", "")
api_text = api_text.replace(
    "            if position:\n"
    "                tx = tqdm(texts, position=position)\n"
    "            elif quiet:\n"
    "                tx = texts\n"
    "            else:\n"
    "                tx = tqdm(texts)\n",
    "            if quiet:\n"
    "                tx = texts\n"
    "            elif position:\n"
    "                from tqdm import tqdm\n"
    "                tx = tqdm(texts, position=position)\n"
    "            else:\n"
    "                from tqdm import tqdm\n"
    "                tx = tqdm(texts)\n",
)
api.write_text(api_text, encoding="utf-8")

download_utils = root / "melo" / "download_utils.py"
download_utils_text = download_utils.read_text(encoding="utf-8")
download_utils_text = download_utils_text.replace(
    "from cached_path import cached_path\n"
    "from huggingface_hub import hf_hub_download\n",
    "",
)
download_utils_text = download_utils_text.replace(
    "            config_path = hf_hub_download(\n",
    "            from huggingface_hub import hf_hub_download\n"
    "            config_path = hf_hub_download(\n",
)
download_utils_text = download_utils_text.replace(
    '            config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json")\n',
    '            from huggingface_hub import hf_hub_download\n'
    '            config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json")\n',
)
download_utils_text = download_utils_text.replace(
    "            config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])\n",
    "            from cached_path import cached_path\n"
    "            config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])\n",
)
download_utils_text = download_utils_text.replace(
    "            ckpt_path = hf_hub_download(\n",
    "            from huggingface_hub import hf_hub_download\n"
    "            ckpt_path = hf_hub_download(\n",
)
download_utils_text = download_utils_text.replace(
    '            ckpt_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="checkpoint.pth")\n',
    '            from huggingface_hub import hf_hub_download\n'
    '            ckpt_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="checkpoint.pth")\n',
)
download_utils_text = download_utils_text.replace(
    "            ckpt_path = cached_path(DOWNLOAD_CKPT_URLS[language])\n",
    "            from cached_path import cached_path\n"
    "            ckpt_path = cached_path(DOWNLOAD_CKPT_URLS[language])\n",
)
download_utils_text = download_utils_text.replace(
    "    return [cached_path(url) for url in PRETRAINED_MODELS.values()]\n",
    "    from cached_path import cached_path\n"
    "    return [cached_path(url) for url in PRETRAINED_MODELS.values()]\n",
)
download_utils.write_text(download_utils_text, encoding="utf-8")

utils = root / "melo" / "utils.py"
utils_text = utils.read_text(encoding="utf-8")
utils_text = utils_text.replace("from scipy.io.wavfile import read\n", "")
utils_text = utils_text.replace(
    "def load_wav_to_torch(full_path):\n"
    "    sampling_rate, data = read(full_path)\n"
    "    return torch.FloatTensor(data.astype(np.float32)), sampling_rate\n",
    "def load_wav_to_torch(full_path):\n"
    "    import soundfile as sf\n"
    "    data, sampling_rate = sf.read(full_path, dtype='float32', always_2d=False)\n"
    "    return torch.FloatTensor(np.asarray(data, dtype=np.float32)), sampling_rate\n",
)
utils.write_text(utils_text, encoding="utf-8")

monotonic_init = root / "melo" / "monotonic_align" / "__init__.py"
monotonic_init.write_text(
    "from numpy import zeros, int32, float32\n"
    "from torch import from_numpy\n\n"
    "try:\n"
    "    from .core import maximum_path_jit\n"
    "except Exception:\n"
    "    maximum_path_jit = None\n\n"
    "def maximum_path_fallback(paths, values, t_ys, t_xs):\n"
    "    max_neg_val = -1e9\n"
    "    for i in range(int(paths.shape[0])):\n"
    "        path = paths[i]\n"
    "        value = values[i]\n"
    "        t_y = int(t_ys[i])\n"
    "        t_x = int(t_xs[i])\n"
    "        index = t_x - 1\n"
    "        for y in range(t_y):\n"
    "            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):\n"
    "                v_cur = max_neg_val if x == y else value[y - 1, x]\n"
    "                if x == 0:\n"
    "                    v_prev = 0.0 if y == 0 else max_neg_val\n"
    "                else:\n"
    "                    v_prev = value[y - 1, x - 1]\n"
    "                value[y, x] += max(v_prev, v_cur)\n"
    "        for y in range(t_y - 1, -1, -1):\n"
    "            path[y, index] = 1\n"
    "            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):\n"
    "                index -= 1\n\n"
    "def maximum_path(neg_cent, mask):\n"
    "    device = neg_cent.device\n"
    "    dtype = neg_cent.dtype\n"
    "    neg_cent = neg_cent.data.cpu().numpy().astype(float32)\n"
    "    path = zeros(neg_cent.shape, dtype=int32)\n"
    "    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)\n"
    "    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)\n"
    "    if maximum_path_jit is None:\n"
    "        maximum_path_fallback(path, neg_cent, t_t_max, t_s_max)\n"
    "    else:\n"
    "        maximum_path_jit(path, neg_cent, t_t_max, t_s_max)\n"
    "    return from_numpy(path).to(device=device, dtype=dtype)\n",
    encoding="utf-8",
)

english = root / "melo" / "text" / "english.py"
english_text = english.read_text(encoding="utf-8")
english_text = english_text.replace(
    "from .japanese import distribute_phone\n",
    "\n"
    "def distribute_phone(n_phone, n_word):\n"
    "    phones_per_word = [0] * n_word\n"
    "    for _ in range(n_phone):\n"
    "        min_tasks = min(phones_per_word)\n"
    "        min_index = phones_per_word.index(min_tasks)\n"
    "        phones_per_word[min_index] += 1\n"
    "    return phones_per_word\n",
)
english.write_text(english_text, encoding="utf-8")

g2p_en = Path(find_spec("g2p_en").origin).with_name("g2p.py")
if g2p_en.is_file():
    g2p_en_text = g2p_en.read_text(encoding="utf-8")
    g2p_en_text = g2p_en_text.replace(
        "try:\n"
        "    nltk.data.find('taggers/averaged_perceptron_tagger.zip')\n"
        "except LookupError:\n"
        "    nltk.download('averaged_perceptron_tagger')\n"
        "try:\n"
        "    nltk.data.find('corpora/cmudict.zip')\n"
        "except LookupError:\n"
        "    nltk.download('cmudict')\n",
        "",
    )
    g2p_en.write_text(g2p_en_text, encoding="utf-8")

torchaudio = root / "torchaudio"
torchaudio.mkdir(exist_ok=True)
(torchaudio / "__init__.py").write_text(
    "import numpy as np\n"
    "import soundfile as sf\n"
    "import torch\n\n"
    "def load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True):\n"
    "    data, sample_rate = sf.read(path, dtype='float32' if normalize else 'int16', always_2d=True)\n"
    "    if frame_offset:\n"
    "        data = data[frame_offset:]\n"
    "    if num_frames is not None and num_frames >= 0:\n"
    "        data = data[:num_frames]\n"
    "    if not normalize and data.dtype != np.float32:\n"
    "        data = data.astype(np.float32) / 32768.0\n"
    "    tensor = torch.from_numpy(data.T.copy() if channels_first else data.copy())\n"
    "    return tensor, int(sample_rate)\n",
    encoding="utf-8",
)

librosa = root / "librosa"
librosa.mkdir(exist_ok=True)
(librosa / "__init__.py").write_text(
    "import numpy as np\n"
    "import soundfile as sf\n\n"
    "from . import filters, util\n\n"
    "def load(path, sr=22050, mono=True):\n"
    "    data, sample_rate = sf.read(path, dtype='float32', always_2d=True)\n"
    "    if mono:\n"
    "        data = data.mean(axis=1)\n"
    "    else:\n"
    "        data = data.T\n"
    "    if sr is not None and int(sample_rate) != int(sr):\n"
    "        raise RuntimeError('librosa stub cannot resample audio')\n"
    "    return np.asarray(data, dtype=np.float32), int(sample_rate)\n",
    encoding="utf-8",
)
(librosa / "filters.py").write_text(
    "import numpy as np\n\n"
    "def _hz_to_mel(frequencies):\n"
    "    frequencies = np.asanyarray(frequencies)\n"
    "    return 2595.0 * np.log10(1.0 + frequencies / 700.0)\n\n"
    "def _mel_to_hz(mels):\n"
    "    mels = np.asanyarray(mels)\n"
    "    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)\n\n"
    "def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, dtype=np.float32, norm='slaney', htk=False):\n"
    "    if fmax is None:\n"
    "        fmax = float(sr) / 2.0\n"
    "    fftfreqs = np.linspace(0.0, float(sr) / 2.0, int(1 + n_fft // 2))\n"
    "    min_mel = _hz_to_mel(fmin)\n"
    "    max_mel = _hz_to_mel(fmax)\n"
    "    mel_f = _mel_to_hz(np.linspace(min_mel, max_mel, int(n_mels) + 2))\n"
    "    fdiff = np.diff(mel_f)\n"
    "    ramps = np.subtract.outer(mel_f, fftfreqs)\n"
    "    weights = np.zeros((int(n_mels), int(1 + n_fft // 2)), dtype=dtype)\n"
    "    for i in range(int(n_mels)):\n"
    "        lower = -ramps[i] / fdiff[i]\n"
    "        upper = ramps[i + 2] / fdiff[i + 1]\n"
    "        weights[i] = np.maximum(0.0, np.minimum(lower, upper))\n"
    "    if norm == 'slaney':\n"
    "        enorm = 2.0 / (mel_f[2:int(n_mels) + 2] - mel_f[:int(n_mels)])\n"
    "        weights *= enorm[:, np.newaxis]\n"
    "    return weights.astype(dtype, copy=False)\n",
    encoding="utf-8",
)
(librosa / "util.py").write_text(
    "import numpy as np\n\n"
    "def pad_center(data, *, size, axis=-1, **kwargs):\n"
    "    n = data.shape[axis]\n"
    "    if n > size:\n"
    "        raise ValueError('target size must be at least input size')\n"
    "    lpad = int((size - n) // 2)\n"
    "    rpad = int(size - n - lpad)\n"
    "    try:\n"
    "        import torch\n"
    "    except Exception:\n"
    "        torch = None\n"
    "    if torch is not None and isinstance(data, torch.Tensor):\n"
    "        axis = axis if axis >= 0 else data.dim() + axis\n"
    "        if axis != data.dim() - 1:\n"
    "            data = data.transpose(axis, -1)\n"
    "            out = torch.nn.functional.pad(data, (lpad, rpad))\n"
    "            return out.transpose(axis, -1)\n"
    "        return torch.nn.functional.pad(data, (lpad, rpad))\n"
    "    pad_width = [(0, 0)] * np.ndim(data)\n"
    "    pad_width[axis] = (lpad, rpad)\n"
    "    return np.pad(data, pad_width, **kwargs)\n",
    encoding="utf-8",
)
PY
