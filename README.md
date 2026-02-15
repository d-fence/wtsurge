# wtsurge
A tool to create Surge XT wavetables from any wavefile.

## Why

Because I saw [this amazing video](https://www.youtube.com/watch?v=FEdjIcmWj1Y) about creating watable for surge-xt.
But the manual process is a bit tedious.

## Python Requirements

- [librosa](https://librosa.org/)
- [numpy](https://numpy.org/)

  ## Usage

```
usage: wtsurge.py [-h] [--mono] [--offset OFFSET] [--nb-segments NB_SEGMENTS] [--pre-normalize] [--post-normalize] [--chunksize {256,512,1024,2048,4096}] wavefile {analyze,surge,split} ...

Wavetable surgery for surge XT

positional arguments:
  wavefile

options:
  -h, --help            show this help message and exit
  --mono, -m            Convert to mono
  --offset, -o OFFSET   Sarting segment
  --nb-segments, -n NB_SEGMENTS
                        Number of segments
  --pre-normalize, -p   Pre-normalize wave file
  --post-normalize, -r  Normalize each segment
  --chunksize {256,512,1024,2048,4096}
                        Chunk size

Commands:
  {analyze,surge,split}
                        Select one of:
    analyze             Analyze a wavfile
    surge               Create a Surge XT wavetable wt file
    split               Split wav file into multiple wav files with wavetable segments

Example: ./wtsurge.py foghorn.wav surge foghorn.wt
```

Once the wavetable `.wt` file or `.wav` file is generated, you can drag and drop it on the surge-xt wavetable widget !

That's it.
