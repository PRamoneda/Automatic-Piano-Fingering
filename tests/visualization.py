import music21
from music21 import note
from music21.articulations import Fingering


def visualize(notes, onsets, fingers, save=None):
    notes = notes * 127
    print(notes)
    print(onsets)
    print(fingers)
    p = music21.stream.Part()
    p.timeSignature = music21.meter.TimeSignature('11/4')
    idx = 0
    time = 0

    while idx < len(notes):
        if notes[idx] != 0:
            chord = [(n, o, f) for n, o, f in zip(notes, onsets, fingers) if onsets[idx] == o]
            # print(len(chord))
            if len(chord) == 1:
                n = note.Note(int(notes[idx]))
                if fingers[idx] != 0:
                    n.articulations = [Fingering(fingers[idx])]
                p.insert(time, n)
                idx += 1
            elif len(chord) >= 1:
                c = music21.chord.Chord([int(n) for n in notes[idx:idx+len(chord)]])
                if sum(fingers) != 0:
                    c.articulations = [Fingering(f) for f in fingers[idx:idx+len(chord)] if f != 0]
                p.insert(time, c)
                idx += len(chord)
            time += 1
        else:
            idx += 1
    if save is None:
        p.show()
    else:
        p.write('musicxml', fp=save)