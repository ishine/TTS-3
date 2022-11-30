import os
import torch
from tqdm import tqdm
import random
from glob import glob
from torch.functional import F
from TTS.tts.models.ecyourtts import EcyourTTS
import soundfile as sf
from argparse import RawTextHelpFormatter, ArgumentParser


torch.set_num_threads(8)


def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Evaluate zero-shot speaker adaptation of a TTS model.
                       Can be used either with a folder of audio references, or a speaker embedding file.\n\n
                       Example run:
                            python eval_zero_shot.py
                                --ref_path /raid/datasets/zero-shot-ref/daps/
                                --model_path /path/to/tts/model.pth
                                --config_path /path/to/tts/config.json
                                --gen_path /tmp/gen_audios/
                                --use_cuda False

                            or with a speaker embedding file:

                            python eval_zero_shot.py
                                --ref_speaker_file /path/to/speaker/embedding/speakers.json
                                --model_path /path/to/tts/model.pth
                                --config_path /path/to/tts/config.json
                    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--ref_path",
        type=str,
        default=None,
        required=False,
        help="Path of the folder containing the audio files to be used as zeroshot references.",
    )

    parser.add_argument(
        "--ref_speaker_file",
        type=str,
        default=None,
        required=False,
        help="Path of the json/pth speakerfile containing the speaker embeddings to be used as zeroshot references.",
    )

    parser.add_argument(
        "--gen_path",
        type=str,
        default="/tmp/gen_audios/",
        required=False,
        help="Path of the folder where the generated audio files will be saved.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path of the model checkpoint to be tested.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        required=True,
        help="Path of the model config to be tested.",
    )

    parser.add_argument(
        "--encoder_model_path",
        type=str,
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar",
        required=False,
        help="Path of the speaker encoder model.",
    )

    parser.add_argument(
        "--encoder_config_path",
        type=str,
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json",
        required=False,
        help="Path of the speaker encoder config.",
    )

    parser.add_argument(
        "--use_cuda", type=bool, help="Run model on CUDA.", default=True
    )

    def save_wav(
        *, wav, path: str, sample_rate: int = None, **kwargs
    ) -> None:
        """Save float waveform to a file using Scipy.

        Args:
            wav (np.ndarray): Waveform with float values in range [-1, 1] to save. Shape (n_values,).
            path (str): Path to a output file.
            sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        """
        sf.write(path, wav, sample_rate, **kwargs)

    args = parser.parse_args()

    assert (
        args.ref_path is not None or args.ref_speaker_file is not None
    ), "You must provide a reference path or a reference speaker file."
    assert (
        args.ref_path is None or args.ref_speaker_file is None
    ), "You must provide a reference path or a reference speaker file, not both."

    model_config = EcyourTTS.load_config(args.config_path)
    model = EcyourTTS.init_from_config(model_config, verbose=True)
    model.load_checkpoint(model_config, args.model_path, eval=True, strict=False)

    if args.ref_speaker_file is not None:
        model.speaker_manager.load_embeddings_from_file(args.ref_speaker_file)

    try:
        os.mkdir(args.gen_path)
    except:
        print("Path already exist: ", args.gen_path)

    def inference(name, d_vector=None):
        outputs = model.synthesize(
            text="This is a test sentence to evaluate the zero shot speaker similarity.",
            # speaker_id=name,
            d_vector=d_vector,
            language_id="en" if model.language_manager is not None else None,
            emotion_id="Neutral" if model.speaker_manager is not None else None,
        )
        out_path = os.path.join(args.gen_path, name)
        save_wav(wav=outputs["wav"][0], path=out_path, sample_rate=model.config.audio.sample_rate)
        return out_path

    random.seed(42)
    ref_speakers = random.sample(
        model.speaker_manager.speaker_names,
        min(len(model.speaker_manager.speaker_names), 100),
    )

    # avg = []
    # for speaker in tqdm(ref_speakers):
    #     ref_d_vector = model.speaker_manager.get_mean_embedding(speaker)
    #     out = inference(speaker)
    #     out_d_vector = model.speaker_manager.compute_embedding_from_clip(out)
    #     ref = torch.tensor(ref_d_vector)
    #     gen = torch.tensor(out_d_vector)
    #     sim = cosine_similarity(ref, gen)
    #     print(speaker, sim)
    #     avg.append(sim)
    # print("\n\n###################")
    # print("Average similarity score: ", sum(avg) / len(avg))

    ref_paths = sorted(glob(args.ref_path+"/**/*.wav", recursive=True))
    avg = []
    for path in tqdm(ref_paths):
        name = path.split("/")[-1]
        ref_d_vector = model.speaker_manager.compute_embedding_from_clip(path)
        ref = torch.tensor(ref_d_vector)
        out = inference(name, d_vector=ref_d_vector)
        out_d_vector = model.speaker_manager.compute_embedding_from_clip(out)
        ref = torch.tensor(ref_d_vector)
        gen = torch.tensor(out_d_vector)
        sim = cosine_similarity(ref, gen)
        print(name, sim)
        avg.append(sim)
    print("\n\n###################")
    print("Average similarity score: ", sum(avg)/len(avg))
