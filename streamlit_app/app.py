from pathlib import Path

import streamlit as st
import pandas as pd
import torch
from PIL import Image

from projects.image_captioning.inference import generate_caption
from projects.image_captioning.evaluation import load_caption_checkpoint
from projects.image_captioning.pretrained_captioning import load_blip_captioner
from projects.image_captioning.retrieval import VggRetrievalCaptioner
from projects.image_captioning.transforms import get_eval_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'DATA' / 'flickr8k'
TRAIN_SPLIT = DATA_DIR / 'splits' / 'train.csv'
FEATURES_DIR = DATA_DIR / 'features' / 'vgg16_pool7'
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


@st.cache_resource(show_spinner='Зареждане на pretrained BLIP модела...')
def load_cached_blip_captioner(device: str):
    return load_blip_captioner(device)


@st.cache_resource(show_spinner='Зареждане на VGG retrieval index...')
def load_cached_retrieval_captioner(train_split_path: str,
                                    features_dir: str,
                                    device: str):
    train_captions = pd.read_csv(train_split_path)
    return VggRetrievalCaptioner.from_cached_features(
        train_captions,
        features_dir,
        get_eval_transforms(),
        device=device,
    )


st.set_page_config(
    page_title='Flickr8k Image Captioning',
    page_icon='🖼️',
    layout='centered',
)
st.title('Генериране на описание за изображение')
st.write('Image captioning приложение с VGG retrieval, VGG16 + LSTM и BLIP режим.')
st.caption(
    'При малък dataset като Flickr8k LSTM decoder-ът понякога дава твърде общи '
    'или несвързани описания. Затова използвам и VGG retrieval режим, който '
    'избира caption от визуално най-близко training изображение.'
)

with st.sidebar:
    st.header('Настройки')
    model_mode = st.selectbox(
        'Модел',
        [
            'Project VGG retrieval',
            'Project baseline VGG16 + LSTM',
            'Pretrained BLIP',
        ],
        help=(
            'VGG retrieval използва визуално подобни train изображения. '
            'VGG16+LSTM е генеративният модел, обучен в проекта. BLIP е '
            'предварително обучен модел за сравнение.'
        ),
    )
    default_checkpoint = default_checkpoint_path()
    checkpoint_value = st.text_input(
        'Път до checkpoint',
        str(default_checkpoint.relative_to(PROJECT_ROOT)),
        disabled=model_mode != 'Project baseline VGG16 + LSTM',
    )
    device = st.selectbox('Устройство', available_devices())
    max_length_override = st.slider(
        'Максимална дължина',
        min_value=5,
        max_value=50,
        value=20,
    )
    decoding = st.selectbox(
        'Метод за генериране',
        ['beam', 'greedy'],
        help='Beam search пробва няколко възможни captions и често е по-стабилен от greedy.',
        disabled=model_mode != 'Project baseline VGG16 + LSTM',
    )
    beam_size = st.slider(
        'Beam size',
        min_value=2,
        max_value=7,
        value=3,
        disabled=decoding != 'beam',
    )
    retrieval_top_k = st.slider(
        'Retrieval top-k',
        min_value=1,
        max_value=10,
        value=5,
        disabled=model_mode != 'Project VGG retrieval',
    )

checkpoint_path = resolve_checkpoint_path(checkpoint_value)

if model_mode == 'Project VGG retrieval':
    if TRAIN_SPLIT.is_file() and FEATURES_DIR.is_dir():
        st.success('Избран е project VGG retrieval режим.')
        st.info(
            'Този режим намира визуално най-близки Flickr8k train изображения '
            'и връща caption от тях. Не е външен pretrained captioning модел.'
        )
    else:
        st.error('Липсва train split или VGG feature cache за retrieval режима.')
elif model_mode == 'Pretrained BLIP':
    st.success('Избран е pretrained BLIP режим за сравнение.')
    st.info(
        'При първо пускане този режим може да свали pretrained модел от Hugging Face. '
        'Основният разработен модел в проекта остава VGG16 + LSTM.'
    )
elif checkpoint_path.is_file():
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
            disabled=(
                model_mode == 'Project baseline VGG16 + LSTM'
                and not checkpoint_path.is_file()
            ),
            use_container_width=True,
        ):
            try:
                if model_mode == 'Project VGG retrieval':
                    retrieval_captioner = load_cached_retrieval_captioner(
                        str(TRAIN_SPLIT),
                        str(FEATURES_DIR),
                        device,
                    )
                    caption, retrieval_matches = retrieval_captioner.generate(
                        image,
                        top_k=retrieval_top_k,
                    )
                    checkpoint = None
                    vocabulary = None
                elif model_mode == 'Pretrained BLIP':
                    captioner = load_cached_blip_captioner(device)
                    caption = captioner.generate(
                        image,
                        max_length=max_length_override,
                    )
                    checkpoint = None
                    vocabulary = None
                    retrieval_matches = []
                else:
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
                        decoding=decoding,
                        beam_size=beam_size,
                    )
                    retrieval_matches = []
            except (KeyError, RuntimeError, TypeError, ValueError, ModuleNotFoundError) as error:
                st.error(f'Моделът не може да генерира описание: {error}')
            else:
                st.subheader('Генерирано описание')
                st.info(caption or 'Моделът не генерира думи.')
                if model_mode == 'Project VGG retrieval':
                    st.caption(
                        'Използван е VGG retrieval baseline. Caption-ът идва от '
                        'визуално подобно Flickr8k training изображение, което '
                        'намалява несвързаните captions спрямо LSTM decoder-а.'
                    )
                    if retrieval_matches:
                        st.subheader('Най-близки training captions')
                        st.dataframe(
                            [
                                {
                                    'image': match.image,
                                    'caption': match.caption,
                                    'similarity': round(match.similarity, 4),
                                }
                                for match in retrieval_matches
                            ],
                            use_container_width=True,
                        )
                elif model_mode == 'Pretrained BLIP':
                    st.caption(
                        'Използван е pretrained BLIP модел за сравнение с '
                        'моделите, разработени в проекта.'
                    )
                elif decoding == 'beam':
                    st.caption(
                        'Използван е beam search. Това може да подобри избора на думи, '
                        'но моделът остава ограничен от Flickr8k training данните.'
                    )

                if checkpoint is not None and vocabulary is not None:
                    column_epoch, column_loss, column_vocab = st.columns(3)
                    column_epoch.metric('Epoch', checkpoint.get('epoch', '—'))
                    validation_loss = checkpoint.get('validation_loss')
                    column_loss.metric(
                        'Validation loss',
                        f'{validation_loss:.4f}' if validation_loss is not None else '—',
                    )
                    column_vocab.metric('Vocabulary', f'{len(vocabulary):,}')
