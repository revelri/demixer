"""Accuracy evaluation harness for the demixer pipeline.

Two complementary evals:
  - synth_groundtruth: known MIDI → SF2 render → transcribe → mir_eval F1.
    Gives a real (if optimistic — clean synth audio) accuracy number.
  - organic_consistency: library audio → pipeline → re-render → re-transcribe,
    measuring stability of the real end-to-end path on organic recordings.
"""
