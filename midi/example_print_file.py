"""
This is an example that uses the MidiToText eventhandler. When an 
event is triggered on it, it prints the event to the console.

It gets the events from the MidiInFile.

So it prints all the events from the infile to the console. great for 
debugging :-s
"""


# get data
test_file = 'DiffusionModels/data/Nottingham/test/ashover_simple_chords_5.mid'

# do parsing
from MidiInFile import MidiInFile
from MidiToText import MidiToText # the event handler
midiIn = MidiInFile(MidiToText(), test_file)
midiIn.read()
