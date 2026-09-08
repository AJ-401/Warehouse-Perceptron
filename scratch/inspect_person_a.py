import json
import glob
import os

files = glob.glob('outputs_person_a/*_tracking_results.json')
print(f'Total tracking files: {len(files)}')
for f in sorted(files):
    with open(f) as fp:
        d = json.load(fp)
    meta = d.get('video_metadata', {})
    frames = d.get('frames', [])
    classes = set()
    states = set()
    events = set()
    has_keypoints = 0
    total_tracks = 0
    box_tracks = 0
    person_tracks = 0
    for fr in frames:
        for tr in fr.get('tracks', []):
            total_tracks += 1
            classes.add(tr.get('class'))
            if tr.get('class') == 'person':
                person_tracks += 1
            else:
                box_tracks += 1
            if 'state' in tr:
                states.add(tr.get('state'))
            if 'event' in tr:
                events.add(tr.get('event'))
            if tr.get('keypoints'):
                has_keypoints += 1
    print(f'=== {os.path.basename(f)} ===')
    print(f'  Frames: {len(frames)}, Resolution: {meta.get("resolution")}, FPS: {meta.get("fps")}')
    print(f'  Classes: {classes}')
    print(f'  Person tracks: {person_tracks}, Non-person tracks: {box_tracks}')
    print(f'  Box States: {states}')
    print(f'  Box Events: {events}')
    print(f'  Tracks with keypoints: {has_keypoints}')
