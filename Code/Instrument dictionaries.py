import mingus
Piano20 = {
  "C-2": mingus.containers.note.Note(name="C", octave=2).to_hertz(),
  "E-2": mingus.containers.note.Note(name="E", octave=2).to_hertz(),
  "G-2": mingus.containers.note.Note(name="G", octave=2).to_hertz(),
  "B-2": mingus.containers.note.Note(name="B", octave=2).to_hertz(),
  "D-3": mingus.containers.note.Note(name="D", octave=3).to_hertz(),
  "F-3": mingus.containers.note.Note(name="F", octave=3).to_hertz(),
  "A-3": mingus.containers.note.Note(name="A", octave=3).to_hertz(),
  "C-4": mingus.containers.note.Note(name="C", octave=4).to_hertz(),
  "E-4": mingus.containers.note.Note(name="E", octave=4).to_hertz(),
  "G-4": mingus.containers.note.Note(name="G", octave=4).to_hertz(),
  "B-4": mingus.containers.note.Note(name="B", octave=4).to_hertz(),
  "D-5": mingus.containers.note.Note(name="D", octave=5).to_hertz(),
  "F-5": mingus.containers.note.Note(name="F", octave=5).to_hertz(),
  "A-5": mingus.containers.note.Note(name="A", octave=5).to_hertz(),
  "C-6": mingus.containers.note.Note(name="C", octave=6).to_hertz(),
  "E-6": mingus.containers.note.Note(name="E", octave=6).to_hertz(),
  "G-6": mingus.containers.note.Note(name="G", octave=6).to_hertz(),
  "B-6": mingus.containers.note.Note(name="B", octave=6).to_hertz(),
  "D-7": mingus.containers.note.Note(name="D", octave=7).to_hertz(),
  "F-7": mingus.containers.note.Note(name="F", octave=7).to_hertz(),

}
Piano20_config = {
    "instrument1_nr" : 0,
    "instrument1" : Piano20.keys(),
    "bin_count" : 20
}

Guitar_full_loss = {

    'C-3':mingus.containers.note.Note(name="C", octave=3).to_hertz(),
    'C#-3':mingus.containers.note.Note(name="C#", octave=3).to_hertz(),
    'D-3':mingus.containers.note.Note(name="D", octave=3).to_hertz(),
    'D#-3':mingus.containers.note.Note(name="D#", octave=3).to_hertz(),
    'E-3':mingus.containers.note.Note(name="E", octave=3).to_hertz(),
    'F-3':mingus.containers.note.Note(name="F", octave=3).to_hertz(),
    'F#-3':mingus.containers.note.Note(name="F#", octave=3).to_hertz(),
    'G-3':mingus.containers.note.Note(name="G", octave=3).to_hertz(),
    'G#-3':mingus.containers.note.Note(name="G#", octave=3).to_hertz(),
    'A-3':mingus.containers.note.Note(name="A", octave=3).to_hertz(),
    'A#-3':mingus.containers.note.Note(name="A#", octave=3).to_hertz(),
    'B-3':mingus.containers.note.Note(name="B", octave=3).to_hertz(),
    'C-4':mingus.containers.note.Note(name="C", octave=4).to_hertz(),
    'C#-4':mingus.containers.note.Note(name="C#", octave=4).to_hertz(),
    'D-4':mingus.containers.note.Note(name="D", octave=4).to_hertz(),
    'D#-4':mingus.containers.note.Note(name="D#", octave=4).to_hertz(),
    'E-4':mingus.containers.note.Note(name="E", octave=4).to_hertz(),
    'F-4':mingus.containers.note.Note(name="F", octave=4).to_hertz(),
    'F#-4':mingus.containers.note.Note(name="F#", octave=4).to_hertz(),
    'G-4':mingus.containers.note.Note(name="G", octave=4).to_hertz(),
    'G#-4':mingus.containers.note.Note(name="G#", octave=4).to_hertz(),
    'A-4':mingus.containers.note.Note(name="A", octave=4).to_hertz(),
    'A#-4':mingus.containers.note.Note(name="A#", octave=4).to_hertz(),
    'B-4':mingus.containers.note.Note(name="B", octave=4).to_hertz(),
    'C-5':mingus.containers.note.Note(name="C", octave=5).to_hertz(),
    'C#-5':mingus.containers.note.Note(name="C#", octave=5).to_hertz(),
    'D-5':mingus.containers.note.Note(name="D", octave=5).to_hertz(),
    'D#-5':mingus.containers.note.Note(name="D#", octave=5).to_hertz(),
    'E-5':mingus.containers.note.Note(name="E", octave=5).to_hertz(),
    'F-5':mingus.containers.note.Note(name="F", octave=5).to_hertz(),
    'F#-5':mingus.containers.note.Note(name="F#", octave=5).to_hertz(),
    'G-5':mingus.containers.note.Note(name="G", octave=5).to_hertz(),
    'G#-5':mingus.containers.note.Note(name="G#", octave=5).to_hertz(),
    'A-5':mingus.containers.note.Note(name="A", octave=5).to_hertz(),
    'A#-5':mingus.containers.note.Note(name="A#", octave=5).to_hertz(),
    'B-5':mingus.containers.note.Note(name="B", octave=5).to_hertz(),
}

EGuitar_jazz_full_loss_config = {
    "instrument1_nr" : 26,
    "instrument1" : Guitar_full_loss.keys(),
    "bin_count" : 36
}

Bass12_maj = {
    "C-1":mingus.containers.note.Note(name="C", octave=1).to_hertz(),
    "E-1":mingus.containers.note.Note(name="E", octave=1).to_hertz(),
    "G-1":mingus.containers.note.Note(name="G", octave=1).to_hertz(),
    "B-1":mingus.containers.note.Note(name="B", octave=1).to_hertz(),
    "D-2":mingus.containers.note.Note(name="D", octave=2).to_hertz(),
    "F#-2":mingus.containers.note.Note(name="F#", octave=2).to_hertz(),
    "A-2":mingus.containers.note.Note(name="A", octave=2).to_hertz(),
    "C#-3":mingus.containers.note.Note(name="C#", octave=3).to_hertz(),
    "E-3":mingus.containers.note.Note(name="E", octave=3).to_hertz(),
    "G#-3":mingus.containers.note.Note(name="G#", octave=3).to_hertz(),
    "B-3":mingus.containers.note.Note(name="B", octave=3).to_hertz(),
    "D#-4":mingus.containers.note.Note(name="D#", octave=4).to_hertz(),
}

Bass12_maj_config = {
    "instrument1_nr" : 33,
    "instrument1" : Bass12_maj.keys(),
    "bin_count" : 12
}

Piano12_maj = {
    "A-4":mingus.containers.note.Note(name="A", octave=4).to_hertz(),
    "C-5":mingus.containers.note.Note(name="C", octave=5).to_hertz(),
    "E-5":mingus.containers.note.Note(name="E", octave=5).to_hertz(),
    "G-5":mingus.containers.note.Note(name="G", octave=5).to_hertz(),
    "B-5":mingus.containers.note.Note(name="B", octave=5).to_hertz(),
    "D-6":mingus.containers.note.Note(name="D", octave=6).to_hertz(),
    "F#-6":mingus.containers.note.Note(name="F#", octave=6).to_hertz(),
    "A-6":mingus.containers.note.Note(name="A", octave=6).to_hertz(),
    "C#-7":mingus.containers.note.Note(name="C#", octave=7).to_hertz(),
    "E-7":mingus.containers.note.Note(name="E", octave=7).to_hertz(),
    "G#-7":mingus.containers.note.Note(name="G#", octave=7).to_hertz(),
    "B-7":mingus.containers.note.Note(name="B", octave=7).to_hertz(),
}

Piano12_maj_config = {
    "instrument1_nr" : 0,
    "instrument1" : Piano12_maj.keys(),
    "bin_count" : 12
}

