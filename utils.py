import json
import music21


def strm2map(strm, hand):
    converted = []
    om = []
    for o in strm.flat.secondsMap:
        if o['element'].isClassOrSubclass(('Note',)):
            finger = [art.fingerNumber for art in o["element"].articulations if
                      type(art) == music21.articulations.Fingering]
            if len(finger) == 1 and finger[0] in [1, 2, 3, 4, 5]:
                o['finger'] = finger[0]
            else:
                o['finger'] = 0
            om.append(o)
        elif o['element'].isClassOrSubclass(('Chord',)):
            articulations = [
                art.fingerNumber for art in o["element"].articulations
                if type(art) == music21.articulations.Fingering and art.fingerNumber in [0, 1, 2, 3, 4, 5]
            ]
            if len(articulations) == len(o['element']):
                if hand == 'left':
                    fingers = list(sorted(articulations, reverse=True))
                else:
                    fingers = list(sorted(articulations))
            else:
                fingers = [0] * len(o['element'])

            om_chord = [
                {
                    'element': oc,
                    'offsetSeconds': o['offsetSeconds'],
                    'endTimeSeconds': o['endTimeSeconds'],
                    'chord': o['element'],
                    'finger': finger
                }
                for oc, finger in zip(sorted(o['element'].notes, key=lambda a: a.pitch), fingers)
            ]
            om.extend(om_chord)
    om_filtered = []
    for o in om:
        offset = o['offsetSeconds']
        duration = o['endTimeSeconds']
        pitch = o['element'].pitch
        simultaneous_notes = [o2 for o2 in om if
                              o2['offsetSeconds'] == offset and o2['element'].pitch.midi == pitch.midi]
        max_duration = max([float(x['endTimeSeconds']) for x in simultaneous_notes])
        if len(simultaneous_notes) > 1 and duration < max_duration and str(offset) + ':' + str(pitch) not in converted:
            continue
        else:
            converted.append(str(offset) + ':' + str(pitch))

        if not (o['element'].tie and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop')) and \
                not ((hasattr(o['element'], 'tie') and o['element'].tie
                      and (o['element'].tie.type == 'continue' or o['element'].tie.type == 'stop'))) and \
                not (o['element'].duration.quarterLength == 0):
            om_filtered.append(o)

    return sorted(om_filtered, key=lambda a: (a['offsetSeconds'], a['element'].pitch))


def save_json(dictionary, name_file):
    with open(name_file, 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True, indent=4)


def load_json(name_file):
    data = None
    with open(name_file, 'r') as fp:
        data = json.load(fp)
        fp.close()
    return data