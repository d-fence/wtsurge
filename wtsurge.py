#!/usr/bin/env python3
import argparse
import librosa
import numpy as np
import sys

from pathlib import Path
from scipy.io import wavfile


def find_zero_crossings_indices(samples, cross_type='positive_ascending'):
    zc = librosa.zero_crossings(samples, pad=False)
    match cross_type:
        case 'all':
            return np.nonzero(zc)[0]
        case 'positive_ascending':
            for i,is_zero in enumerate(zc[:-10]):
                if is_zero:
                    seg = samples[i:i+10]
                    d = np.diff(seg)
                    d = d > 0
                    if not d.all():
                        zc[i] = False
    return np.nonzero(zc)[0]

def get_segments(samples, zero_crossings_indices):
    segments = []
    for a,b in zip(zero_crossings_indices[:-1], zero_crossings_indices[1:]):
        segment = samples[a:b]
        segments.append(segment)
    return segments


class SurgeryException(Exception):
    ...


class Surgery:

    def __init__(self, wavefile_path, mono=False, nb_segments=512, offset=0, pre_normalize=False, post_normalize=False, chunksize=1024):
        print(f'Opening "{wavefile_path}"')
        self.wavefile = wavefile_path
        if not self.wavefile.exists():
            print(f'Wavefile "{self.wavefile}" not found', sys.stderr)
            sys.exit(1)
        self.raw_filename = self.wavefile.name.removesuffix(".wav")
        self.audio_data, self.sample_rate = librosa.load(str(wavefile_path), sr=None, mono = mono)
        if pre_normalize:
            self.audio_data = librosa.util.normalize(self.audio_data, fill=None, norm=1)
        self.offset = offset
        self.nb_segments = nb_segments
        self.number_channel = self.audio_data.ndim
        self.split_dir = None
        self.outfile_path = None
        self.post_normalize = post_normalize
        self.chunksize = chunksize
        self.segments = []
        self.wavetable_data = []

    def set_split_dir(self, split_dir):
        n = 0
        initial_split_name = str(split_dir)
        while split_dir.exists():
            n += 1
            split_dir = Path(f'{initial_split_name}_{n:04}')
        self.split_dir = split_dir
        self.split_dir.mkdir()

    def set_outfile_path(self, outfile_path):
        n = 0
        initial_outfile_path = outfile_path
        while outfile_path.exists():
            n += 1
            outfile_path = outfile_path.parent / Path(f'{initial_outfile_path.stem}_{n:04}{initial_outfile_path.suffix}')
        self.outfile_path = outfile_path

    def find_segments(self):
        if self.number_channel == 1:
            zero_crossings_indices = find_zero_crossings_indices(self.audio_data)
            segments = get_segments(self.audio_data, zero_crossings_indices)
        else:
            segments = []
            for i in range(self.number_channel):
                zero_crossings_indices = find_zero_crossings_indices(self.audio_data[i])
                segments.extend(get_segments(self.audio_data[i], zero_crossings_indices))
        for segment in segments:
            if len(segment) < 10:
                continue
            target_sample_rate = (self.sample_rate * self.chunksize) / len(segment)
            stretched_data = librosa.resample(segment, orig_sr=self.sample_rate, target_sr=target_sample_rate, res_type='sinc_best')
            if self.post_normalize:
                segment = segment / np.max(np.abs(segment))
            # after the normalize and stretch, it happens that the last samples are positive
            if np.all(stretched_data[-10:] <= 0):
                self.segments.append(stretched_data[:self.chunksize])

    def split(self):
        print(f'Writing {len(self.segments)} into "{self.split_dir}"')
        for i, segment in enumerate(self.segments[self.offset:self.offset+self.nb_segments]):
            outfile_full_path = self.split_dir.absolute() / f"{self.raw_filename}_{i+1:03d}.wav"
            wavfile.write(outfile_full_path, self.sample_rate, segment)

    def write_surge_wt(self):
        # surge wavetable format
        # byte
        # 0-3	'vawt'		as big-endian text - all the following bytes are little-endian
        # 4-7	wave_size 	size of each wave; 2 ... 4096, must be a power of 2 (v1.3)
        # 8-9	wave_count 	number of waves; 1 ... 512 (v1.3)
        # 10-11	flags
        #         0001: is a sample instead of a wavetable
        #         0002: is a looped sample
        #         0004: if not set, wave data is in float32 format
        #               if set, wave data is in int16 format
        #         0008: if not set, a sample with 0 dBFS peak will end up having a peak of 2^14 (uses 15 bits, -6 dBFS peak)
        #               if set, int16 data uses the full 16-bit range
        #         0010: if set, there is a metadata block after the wave data
        # 12-(12+size)	wave data
        #         float32 format: size = 4 * wave_size * wave_count bytes
        #         int16 format:   sise = 2 * wave_size * wave_count bytes
        # 12+size+1-end   metadata (if flags & 0x10)
        #         metadata is a null terminated XML string with the tag <wtmeta> as the root and application
        #         dependent beyond that

        # Normalize audio data to the range of -1.0 to 1.0
        audio_data = self.wavetable_data / np.max(np.abs(self.wavetable_data))

        print(f'Writing wavetable file "{self.outfile_path}"')
        with self.outfile_path.open("wb") as f:
            f.write(b"vawt")
            print(f"frames_size = {self.chunksize}")
            f.write(int(self.chunksize).to_bytes(4, byteorder="little"))
            frames_count = len(audio_data) / self.chunksize
            print(f"frames_counts = {frames_count}")
            f.write(int(frames_count).to_bytes(2, byteorder="little"))
            # flags
            f.write(int(0).to_bytes(2, byteorder="little"))
            for sample in audio_data:
                f.write(sample)

    def write_surge_wavefile(self):
        # adds the RIFF chunk for surge
        wavfile.write(str(self.outfile_path), self.sample_rate, self.wavetable_data)
        with self.outfile_path.open("rb") as f:
            raw_wav_data = bytearray(f.read())
            assert(raw_wav_data[:4] == b"RIFF")
            assert(raw_wav_data[8:12] == b"WAVE")
            assert(raw_wav_data[12:16] == b"fmt ")
            fmt_chunk_lenght = int.from_bytes(raw_wav_data[16:20], "little")

        surge_chunk = b'srge'
        surge_chunk += int.to_bytes(8, 4, 'little')  # chunk size
        surge_chunk += int.to_bytes(1, 4, 'little')  # version (probably always 1)
        surge_chunk += int.to_bytes(self.chunksize, 4, 'little')

        new_filesize = int.from_bytes(raw_wav_data[4:8], "little") + len(surge_chunk)
        new_filesize_bytes = int.to_bytes(new_filesize, 4, "little")
        raw_wav_data[4:8] = new_filesize_bytes

        with self.outfile_path.open("wb") as f:
            f.write(raw_wav_data[:20 + fmt_chunk_lenght])
            f.write(surge_chunk)
            f.write(raw_wav_data[20 + fmt_chunk_lenght:])

    def surge(self):
        self.wavetable_data = np.concatenate(self.segments[self.offset:self.offset+self.nb_segments])
        if self.outfile_path.suffix == ".wt":
            self.write_surge_wt()
        elif self.outfile_path.suffix == ".wav":
            self.write_surge_wavefile()


def analyze(surgeon):
     print(f'Number of channel in wavefile: {surgeon.number_channel}')
     print(f'Sample rate: {surgeon.sample_rate}')
     print('Searching for Wavetable segments candidates')
     surgeon.find_segments()
     print(f'Wavetable segments found: {len(surgeon.segments)}')

def split(surgeon, args):
    surgeon.set_split_dir(args.outdir)
    surgeon.find_segments()
    surgeon.split()

def surge(surgeon, args):
    surgeon.set_outfile_path(args.wavetable_file)
    surgeon.find_segments()
    surgeon.surge()

def main(args):
    surgeon = Surgery(
        args.wavefile,
        mono=args.mono,
        offset=args.offset,
        nb_segments=args.nb_segments,
        pre_normalize=args.pre_normalize,
        post_normalize=args.post_normalize,
    )
    match args.command:
        case None | 'analyze':
            analyze(surgeon)
        case 'split':
            split(surgeon, args)
        case 'surge':
            surge(surgeon, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Wavetable surgery for surge XT",
            epilog=f"Example: {sys.argv[0]} foghorn.wav surge foghorn.wt",
    )
    parser.add_argument('wavefile', type=Path)
    parser.add_argument('--mono', '-m', action='store_true', default=False, help='Convert to mono')
    parser.add_argument('--offset', '-o', type=int, default=0, help='Sarting segment')
    parser.add_argument('--nb-segments', '-n', type=int, default=512, help='Number of segments')
    parser.add_argument('--pre-normalize', '-p', action='store_true', default=False, help='Pre-normalize wave file')
    parser.add_argument('--post-normalize', '-r', action='store_true', default=False, help='Normalize each segment')
    parser.add_argument("--chunksize", type=int, default=1024, choices=[256, 512, 1024, 2048, 4096], help="Chunk size")

    subparsers = parser.add_subparsers(dest='command', title="Commands", help='Select one of:')

    analyze_parser = subparsers.add_parser('analyze', help='Analyze a wavfile')

    surge_parser = subparsers.add_parser('surge', help='Create a Surge XT wavetable wt file')
    surge_parser.add_argument('wavetable_file', type=Path, help='Wavetable file path')

    split_parser = subparsers.add_parser('split', help='Split wav file into multiple wav files with wavetable segments')
    split_parser.add_argument('outdir', type=Path, help='Directory where to put segment files (default: wavefile basen ame)')

    args = parser.parse_args()
    main(args)
