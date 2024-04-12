from MidiOutStream import MidiOutStream
from MidiInFile import MidiInFile

"""
This prints all note on events on midi channel 0
"""


class Transposer(MidiOutStream):
    
    "Transposes all notes by 1 octave"
    
    def note_on(self, channel=0, note=0x40, velocity=0x40):
        if channel == 0:
            print(channel, note, velocity, self.rel_time())


event_handler = Transposer()

in_file = 'DiffusionModels/data/Nottingham/test/ashover_simple_chords_5.mid'
midi_in = MidiInFile(event_handler, in_file)
print(midi_in.raw_in)

