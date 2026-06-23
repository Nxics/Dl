from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from projects.image_captioning.inference import generate_caption
from projects.image_captioning.evaluation import load_caption_checkpoint
from projects.image_captioning.transforms import get_eval_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'best_model.pt'
SMOKE_CHECKPOINT = PROJECT_ROOT / 'checkpoints' / 'smoke_test_model.pt'


def default_checkpoint_path() -> Path:
    if BEST_CHECKPOINT.is_file():
        return BEST_CHECKPOINT
    if SMOKE_CHECKPOINT.is_file():
        return SMOKE_CHECKPOINT
    return BEST_CHECKPOINT


def resolve_checkpoint_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def available_devices() -> list[str]:
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.insert(0, 'mps')
    if torch.cuda.is_available():
        devices.insert(0, 'cuda')
    return devices


@st.cache_resource(show_spinner='Зареждане на модела...')
def load_cached_checkpoint(path: str, modified_time_ns: int, device: str):
    del modified_time_ns
    return load_caption_checkpoint(path, device)


st.set_page_config(
    page_title='Flickr8k Image Captioning',
    page_icon='🖼️',
    layout='centered',
)
st.title('Генериране на описание за изображение')
st.write('VGG16 encoder + LSTM decoder, обучен върху Flickr8k.')

with st.sidebar:
    st.header('Настройки')
    default_checkpoint = default_checkpoint_path()
    checkpoint_value = st.text_input(
        'Път до checkpoint',
        str(default_checkpoint.relative_to(PROJECT_ROOT)),
    )
    device = st.selectbox('Устройство', available_devices())
    max_length_override = st.slider(
        'Максимална дължина',
        min_value=5,
        max_value=50,
        value=20,
    )

checkpoint_path = resolve_checkpoint_path(checkpoint_value)

if checkpoint_path.is_file():
    if checkpoint_path.name == SMOKE_CHECKPOINT.name:
        st.warning(
            'Използва се smoke-test модел, обучен само върху един batch. '
            'Приложението работи технически, но описанията няма да бъдат качествени.'
        )
    else:
        st.success(f'Зареден checkpoint: {checkpoint_path.name}')
else:
    st.error(
        'Няма обучен checkpoint на посочения път. '
        'Изпълнете full training секцията в modeling notebook-а, '
        f'за да създадете `{BEST_CHECKPOINT.relative_to(PROJECT_ROOT)}`.'
    )

uploaded_file = st.file_uploader('Изображение', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except (OSError, ValueError) as error:
        st.error(f'Изображението не може да бъде прочетено: {error}')
    else:
        st.image(image, caption='Входно изображение', use_container_width=True)

        if st.button(
            'Генерирай описание',
            type='primary',
            disabled=not checkpoint_path.is_file(),
            use_container_width=True,
        ):
            try:
                model, vocabulary, checkpoint = load_cached_checkpoint(
                    str(checkpoint_path),
                    checkpoint_path.stat().st_mtime_ns,
                    device,
                )
                trained_max_length = checkpoint.get(
                    'max_caption_length',
                    max_length_override,
                )
                caption = generate_caption(
                    model,
                    image,
                    vocabulary,
                    get_eval_transforms(),
                    max_length=min(max_length_override, trained_max_length),
                    device=device,
                )
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                st.error(f'Checkpoint файлът е невалиден или несъвместим: {error}')
            else:
                st.subheader('Генерирано описание')
                st.info(caption or 'Моделът не генерира думи.')

                column_epoch, column_loss, column_vocab = st.columns(3)
                column_epoch.metric('Epoch', checkpoint.get('epoch', '—'))
                validation_loss = checkpoint.get('validation_loss')
                column_loss.metric(
                    'Validation loss',
                    f'{validation_loss:.4f}' if validation_loss is not None else '—',
                )
                column_vocab.metric('Vocabulary', f'{len(vocabulary):,}')
