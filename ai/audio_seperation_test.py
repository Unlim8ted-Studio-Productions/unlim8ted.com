import random
import math
from midiutil import MIDIFile
import os

# Ensure the output directory exists
os.makedirs("ai", exist_ok=True)

# Set a seed for reproducibility
def set_seed(seed):
    random.seed(seed)

# Generate a musical scale procedurally
def generate_scale(root_note=60, scale_type="major"):
    scale_intervals = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "pentatonic": [0, 2, 4, 7, 9]
    }
    intervals = scale_intervals.get(scale_type, scale_intervals["major"])
    return [root_note + i for i in intervals]

# Procedurally generate a chord progression
def generate_chord_progression(scale, length=4):
    chord_templates = [
        [0, 2, 4],  # I chord
        [1, 3, 5],  # ii chord
        [3, 5, 0],  # IV chord
        [4, 6, 1],  # V chord
        [5, 0, 2]   # vi chord
    ]
    progression = []
    for _ in range(length):
        chord = random.choice(chord_templates)
        progression.append([scale[note % len(scale)] for note in chord])
    return progression

# Generate a melody procedurally
def generate_melody(scale, length, rhythm_pattern):
    melody = []
    for i in range(length):
        note = random.choice(scale)
        duration = rhythm_pattern[i % len(rhythm_pattern)]
        melody.append((note, duration))
    return melody

# Generate a bassline from chord roots
def generate_bassline(chord_progression, rhythm_pattern):
    bassline = []
    for chord in chord_progression:
        root_note = chord[0]  # Root note of the chord
        duration = rhythm_pattern[random.randint(0, len(rhythm_pattern) - 1)]
        bassline.append((root_note, duration))
    return bassline

# Generate percussion patterns procedurally
def generate_percussion_pattern(length):
    percussion = []
    for i in range(length):
        if i % 4 == 0:  # Strong beat
            percussion.append((36, 1))  # Kick drum
        elif i % 4 == 2:  # Off-beat
            percussion.append((38, 1))  # Snare drum
        else:
            percussion.append((42, 0.5))  # Hi-hat
    return percussion

# Add a track to the MIDI file
def add_track(midi, track, channel, notes, start_time, volume):
    time = start_time
    for note, duration in notes:
        midi.addNote(track, channel, note, time, duration, volume)
        time += duration

# Main function to create procedural music
def create_procedural_music(seed, output_file, scale_type="major"):
    set_seed(seed)
    midi = MIDIFile(5)  # Five tracks: Piano, Strings, Bass, Lead, Percussion
    tempo = 120

    # Set tempo and track names
    for track, name in enumerate(["Piano", "Strings", "Bass", "Lead", "Percussion"]):
        midi.addTrackName(track, 0, name)
        midi.addTempo(track, 0, tempo)

    # Procedural scale and rhythm patterns
    scale = generate_scale(root_note=60, scale_type=scale_type)
    chord_progression = generate_chord_progression(scale, length=4)
    rhythm_pattern = [1, 0.5, 0.5, 1]  # Simple rhythmic pattern

    # Generate parts procedurally
    piano = generate_melody(scale, 16, rhythm_pattern)
    strings = generate_melody(scale, 16, rhythm_pattern)
    bass = generate_bassline(chord_progression, rhythm_pattern)
    lead = generate_melody(scale, 16, [0.5, 0.5, 1])  # Faster rhythm for the lead
    percussion = generate_percussion_pattern(16)

    # Add tracks to MIDI
    add_track(midi, 0, 0, piano, 0, 100)  # Piano
    add_track(midi, 1, 1, strings, 0, 80)  # Strings
    add_track(midi, 2, 2, bass, 0, 90)  # Bass
    add_track(midi, 3, 3, lead, 0, 110)  # Lead
    add_track(midi, 4, 9, percussion, 0, 100)  # Percussion (channel 9 for drums)

    # Write the MIDI file
    with open(output_file, "wb") as output:
        midi.writeFile(output)

    print(f"MIDI file '{output_file}' created successfully!")

# Run the program
seed = 12345  # Procedural seed for reproducibility
output_file = "ai/filename.mid"
create_procedural_music(seed, output_file, scale_type="pentatonic")
