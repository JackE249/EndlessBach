import os
import numpy as np
import pandas as pd
import random as rand
from keras import layers
from keras import models

size = 4

def prep():
    train_data_and_labels = []
    test_data_and_labels = []
    valid_data_and_labels = []
    
    ([train_bach_data, test_bach_data, valid_bach_data], maxes, mins) = bach_prep()
    train_data_and_labels += train_bach_data
    test_data_and_labels += test_bach_data
    valid_data_and_labels += valid_bach_data
    
    print(len(train_bach_data))
    print(len(test_bach_data))
    print(len(valid_bach_data))
    [train_not_bach_data, test_not_bach_data, valid_not_bach_data] = not_bach_prep([len(train_bach_data),len(test_bach_data),len(valid_bach_data)],maxes,mins)
    train_data_and_labels+=train_not_bach_data
    test_data_and_labels+=test_not_bach_data
    valid_data_and_labels+=valid_not_bach_data

    train_data_and_labels = np.array(train_data_and_labels)
    test_data_and_labels = np.array(test_data_and_labels)
    valid_data_and_labels = np.array(valid_data_and_labels)
    np.random.shuffle(train_data_and_labels)
    print(train_data_and_labels.shape)
    np.random.shuffle(test_data_and_labels)
    print(test_data_and_labels.shape)
    np.random.shuffle(valid_data_and_labels)
    print(valid_data_and_labels.shape)
    
    train_data = train_data_and_labels[0:,:size]
    train_labels = train_data_and_labels[0:,size,0]
    test_data = test_data_and_labels[0:,:size]
    test_labels = test_data_and_labels[0:,size,0]
    valid_data = valid_data_and_labels[0:,:size]
    valid_labels = valid_data_and_labels[0:,size,0]
    
    return [(train_data,train_labels),(test_data,test_labels),(valid_data,valid_labels)]

def bach_prep():
    og_dir = r'C:\Users\zackr\Downloads\archive'
    dirs = ['train','test','valid']
    print(dirs)
    data = [[],[],[]]
    (note1_max, note2_max, note3_max, note4_max) = (-1,-1,-1,-1)
    (note1_min, note2_min, note3_min, note4_min) = (128,128,128,128)
    for j in range(len(dirs)): 
        curr_dir = os.path.join(og_dir, dirs[j])
        print(curr_dir)
        for filename in os.listdir(curr_dir):
            print(filename)
            csv = pd.read_csv(os.path.join(curr_dir, filename))
            temp_note1_max = np.amax(csv.iloc[0:,0:1].to_numpy())
            if temp_note1_max > note1_max:
                note1_max = temp_note1_max
            temp_note2_max = np.amax(csv.iloc[0:,1:2].to_numpy())
            if temp_note2_max > note2_max:
                note2_max = temp_note2_max
            temp_note3_max = np.amax(csv.iloc[0:,2:3].to_numpy())
            if temp_note3_max > note3_max:
                note3_max = temp_note3_max
            temp_note4_max = np.amax(csv.iloc[0:,3:].to_numpy())
            if temp_note4_max > note4_max:
                note4_max = temp_note4_max
            temp_note1_min = np.amin(csv.iloc[0:,0:1].to_numpy())
            if temp_note1_min < note1_min and temp_note1_min > 0:
                note1_min = temp_note1_min
            temp_note2_min = np.amin(csv.iloc[0:,1:2].to_numpy())
            if temp_note2_min < note2_min and temp_note2_min > 0:
                note2_min = temp_note2_min
            temp_note3_min = np.amin(csv.iloc[0:,2:3].to_numpy())
            if temp_note3_min < note3_min and temp_note3_min > 0:
                note3_min = temp_note3_min
            temp_note4_min = np.amin(csv.iloc[0:,3:].to_numpy())
            if temp_note4_min < note4_min and temp_note4_min > 0:
                note4_min = temp_note4_min
            csv=csv.div(127)
            harmonies = csv.iloc[0:,0:].to_numpy()
            temp_data=[]
            for i in range(len(harmonies[:-size])):
                harm_of_size = harmonies[i:i+size].tolist()
                harm_of_size += [[1]*4]
                temp_data += [harm_of_size]
            data[j]+=temp_data
    print("Note1 max:", note1_max, " min:", note1_min)
    print("Note2 max:", note2_max, " min:", note2_min)
    print("Note3 max:", note3_max, " min:", note3_min)
    print("Note4 max:", note4_max, " min:", note4_min)
    maxes = [note1_max, note2_max, note3_max, note4_max]
    mins = [note1_min, note2_min, note3_min, note4_min]
    return (data, maxes, mins)

def not_bach_prep(sizes, maxes, mins):
    bigger_sizes = list([round(sizes[i]/4)*2 for i in range(len(sizes))])
    smaller_sizes = list([round(sizes[i]/4) for i in range(len(sizes))])
    data = [[],[],[]]
    temp_data4 = not_bach_group_n(bigger_sizes, maxes, mins, 4)
    for i in range(len(data)):
        data[i] += temp_data4[i]
    temp_data2 = not_bach_group_n(smaller_sizes, maxes, mins, 2)
    for i in range(len(data)):
        data[i] += temp_data2[i]
    temp_data1 = not_bach_group_n(smaller_sizes,maxes, mins, 1)
    for i in range(len(data)):
        data[i] += temp_data1[i]
    #data += note_bach_group_rand(smaller_sizes,maxes, mins)
    return data

def not_bach_group_n_rpt(sizes, maxes, mins, n, rpt):
    temp_data = [[],[],[]]
    data = [[],[],[]]
    for i in range(len(sizes)):
        for _ in range(0,sizes[i]+size, n):
            note1 = int(rand.randint(mins[0], maxes[0]))
            note2 = int(rand.randint(mins[1], min(note1, maxes[1])))
            note3 = int(rand.randint(mins[2], min(note2, maxes[2])))
            note4 = int(rand.randint(mins[3], min(note3, maxes[3])))
            temp_harm = [[note1/127, note2/127, note3/127, note4/127]]*n
            temp_data[i] += [[note1/127, note2/127, note3/127, note4/127]]*n
        for j in range(len(temp_data[i][:-size])):
            harm_of_size = temp_data[i][j:j+size]
            harm_of_size += [[0]*4]
            data[i] += [harm_of_size]
    print("size:",len(data))
    return data

def not_bach_group_n(sizes, maxes, mins, n):
    temp_data = [[],[],[]]
    data = [[],[],[]]
    for i in range(len(sizes)):
        for _ in range(0,sizes[i]+size, n):
            note1 = int(rand.randint(mins[0], maxes[0]))
            note2 = int(rand.randint(mins[1], min(note1, maxes[1])))
            note3 = int(rand.randint(mins[2], min(note2, maxes[2])))
            note4 = int(rand.randint(mins[3], min(note3, maxes[3])))
            temp_data[i] += [[note1/127, note2/127, note3/127, note4/127]]*n
        for j in range(len(temp_data[i][:-size])):
            harm_of_size = temp_data[i][j:j+size]
            harm_of_size += [[0]*4]
            data[i] += [harm_of_size]
    print("size:",len(data))
    return data

# def note_bach_group2(num, maxes, mins):
#     data = []
#     return data

# def note_bach_group1(num, maxes, mins):
#     data = []
#     return data

# def note_bach_group_rand(num, maxes, mins):
#     data = []
#     return data

def main():
    [(train_data,train_labels),(test_data,test_labels),(valid_data,valid_labels)] = prep()

    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1, 4, 4), padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print("acc:", test_acc, "loss:", test_loss)


if __name__ == '__main__':
    main()
