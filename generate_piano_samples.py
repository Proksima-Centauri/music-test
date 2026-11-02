#!/usr/bin/env python3
"""Render a browser-ready piano sample bank from a local FluidSynth SoundFont.

Install dependencies with:
    python3 -m pip install pyfluidsynth numpy

Then run, for example:
    python3 generate_piano_samples.py --soundfont /usr/share/sounds/sf2/TimGM6mb.sf2
"""

import argparse
import json
import wave
from pathlib import Path

import fluidsynth
import numpy as np


SAMPLE_RATE = 22050
FIRST_MIDI_NOTE = 48  # C3
LAST_MIDI_NOTE = 83   # B5
NOTE_HOLD_SECONDS = 1.55
NOTE_RELEASE_SECONDS = 0.8
SILENCE_BETWEEN_NOTES_SECONDS = 0.12


def render_sample_bank(soundfont_path: Path, output_path: Path, manifest_path: Path) -> None:
    if not soundfont_path.is_file():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    synth = fluidsynth.Synth(gain=0.7, samplerate=SAMPLE_RATE, channels=16)
    try:
        soundfont_id = synth.sfload(str(soundfont_path))
        if soundfont_id < 0:
            raise RuntimeError(f"FluidSynth could not load SoundFont: {soundfont_path}")

        synth.program_select(0, soundfont_id, 0, 0)  # GM program 0: acoustic grand piano
        for setting in ("synth.reverb.active", "synth.chorus.active"):
            try:
                synth.setting(setting, 0)
            except (TypeError, ValueError):
                pass

        hold_frames = int(SAMPLE_RATE * NOTE_HOLD_SECONDS)
        release_frames = int(SAMPLE_RATE * NOTE_RELEASE_SECONDS)
        silence_frames = int(SAMPLE_RATE * SILENCE_BETWEEN_NOTES_SECONDS)
        silence = np.zeros(silence_frames * 2, dtype=np.int16)  # interleaved stereo
        audio_segments = []
        notes = {}
        frame_cursor = 0

        for midi_note in range(FIRST_MIDI_NOTE, LAST_MIDI_NOTE + 1):
            synth.system_reset()
            synth.noteon(0, midi_note, 104)
            held_audio = np.asarray(synth.get_samples(hold_frames), dtype=np.int16).copy()
            synth.noteoff(0, midi_note)
            release_audio = np.asarray(synth.get_samples(release_frames), dtype=np.int16).copy()

            note_audio = np.concatenate((held_audio, release_audio))
            note_frames = note_audio.size // 2
            notes[str(midi_note)] = {
                "offset": frame_cursor / SAMPLE_RATE,
                "duration": note_frames / SAMPLE_RATE,
            }
            audio_segments.extend((note_audio, silence))
            frame_cursor += note_frames + silence_frames

        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(np.concatenate(audio_segments).astype("<i2", copy=False).tobytes())

        manifest = {
            "sampleRate": SAMPLE_RATE,
            "channels": 2,
            "program": "GM Acoustic Grand Piano (program 0)",
            "sourceSoundFont": soundfont_path.name,
            "firstMidiNote": FIRST_MIDI_NOTE,
            "lastMidiNote": LAST_MIDI_NOTE,
            "notes": notes,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Rendered {len(notes)} piano notes to {output_path}")
        print(f"Sample offsets written to {manifest_path}")
    finally:
        synth.delete()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=Path("/usr/share/sounds/sf2/TimGM6mb.sf2"),
        help="path to a locally installed piano SoundFont (.sf2)",
    )
    parser.add_argument("--output", type=Path, default=Path("assets/piano_notes.wav"))
    parser.add_argument("--manifest", type=Path, default=Path("assets/piano_notes.json"))
    args = parser.parse_args()
    render_sample_bank(args.soundfont, args.output, args.manifest)


if __name__ == "__main__":
    main()
