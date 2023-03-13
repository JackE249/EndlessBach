import numpy as np
import pandas as pd
import keras

rows_size = 200 # rows_size*4 individual notes
maxlen = 60
temperature = [.25, .5, .75, .9]

model = keras.models.load_model("bach_model1.h5")
csv = pd.read_csv('chorale_377.csv')
notes = [0, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
         61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 
         74, 75, 76, 77, 78, 79, 80, 81]
note_indices = dict((note, notes.index(note)) for note in notes)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


full_generated_piece = []
generated_notes = csv.iloc[:,:].to_numpy().flatten().tolist()[:maxlen]
full_generated_piece += [generated_notes]
full_generated_piece += [generated_notes]
full_generated_piece += [generated_notes]
full_generated_piece += [generated_notes]
for j in range(len(temperature)):
  for i in range(rows_size*4):
      print(i)
      sampled = np.zeros(shape=(1,maxlen,len(notes)))
      for t in range(len(generated_notes)):
          note = generated_notes[t]
          sampled[0, t, note_indices[note]] = 1
      preds = model.predict(sampled, verbose=0)[0]
      next_index = sample(preds, temperature[j])
      next_note = notes[next_index]
      generated_notes += [next_note]
      generated_notes = generated_notes[1:]
      full_generated_piece[j] += [next_note]

  f = open("generated_377piece" + str(temperature[j]) + ".csv", 'w')
  f.write("note0,note1,note2,note3\n")
  for i in range(0,(rows_size*4)+60,4):
      f.write(str(full_generated_piece[j][i])+',')
      f.write(str(full_generated_piece[j][i+1])+',')
      f.write(str(full_generated_piece[j][i+2])+',')
      f.write(str(full_generated_piece[j][i+3])+'\n')
  f.close()


