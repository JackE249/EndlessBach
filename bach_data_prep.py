import os
import numpy as np
import pandas as pd
import random as rand

size = 4

def prep():
    train_data_and_labels = []
    test_data_and_labels = []
    valid_data_and_labels = []
    
    ([train_bach_data, test_bach_data, valid_bach_data], maxes, mins) = bach_prep()
    train_data_and_labels += train_bach_data
    test_data_and_labels += test_bach_data
    valid_data_and_labels += valid_bach_data
    
    # print(len(train_bach_data))
    # print(len(test_bach_data))
    # print(len(valid_bach_data))

    all_bach = np.append(train_bach_data, test_bach_data, axis=0)
    all_bach = np.append(all_bach, valid_bach_data, axis=0)[:,:size].tolist()
    #print(len(all_bach))
    
    #this part (the list comprehensions) takes by far most time
    [train_not_bach_data, test_not_bach_data, valid_not_bach_data] = not_bach_prep([len(train_bach_data),len(test_bach_data),len(valid_bach_data)],maxes,mins)
    train_not_bach_data = [item for item in train_not_bach_data if item[:size] not in all_bach]
    test_not_bach_data = [item for item in test_not_bach_data if item[:size] not in all_bach]
    valid_not_bach_data = [item for item in valid_not_bach_data if item[:size] not in all_bach]
    train_data_and_labels+=train_not_bach_data
    test_data_and_labels+=test_not_bach_data
    valid_data_and_labels+=valid_not_bach_data

    print(len(train_not_bach_data))
    print(len(test_not_bach_data))
    print(len(valid_not_bach_data))

    train_data_and_labels = np.array(train_data_and_labels)
    test_data_and_labels = np.array(test_data_and_labels)
    valid_data_and_labels = np.array(valid_data_and_labels)
    np.random.shuffle(train_data_and_labels)
    np.random.shuffle(test_data_and_labels)
    np.random.shuffle(valid_data_and_labels)
    
    print(train_data_and_labels.shape)
    print(test_data_and_labels.shape)
    print(valid_data_and_labels.shape)

    train_data = train_data_and_labels[:,:size]
    test_data = test_data_and_labels[:,:size]
    valid_data = valid_data_and_labels[:,:size]
    
    train_labels = train_data_and_labels[:,size,0]
    test_labels = test_data_and_labels[:,size,0]
    valid_labels = valid_data_and_labels[:,size,0]
    
    return [(train_data,train_labels),(test_data,test_labels),(valid_data,valid_labels)]

def bach_prep():
    og_dir = r'C:\Users\zackr\Downloads\archive' # this is for Zac's computer
    dirs = ['train','test','valid']
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
    bigger_sizes = list([round(sizes[i]/16)*3 for i in range(len(sizes))])
    smaller_sizes = list([round(sizes[i]/8) for i in range(len(sizes))])
    data = [[],[],[]]
    temp_data = not_bach_group_n(bigger_sizes, maxes, mins, 4)
    for i in range(len(data)):
        data[i] += temp_data[i]
    temp_data = not_bach_group_n(smaller_sizes, maxes, mins, 2)
    for i in range(len(data)):
        data[i] += temp_data[i]
    temp_data = not_bach_group_n(bigger_sizes, maxes, mins, -1)
    for i in range(len(data)):
        data[i] += temp_data[i]
    temp_data = not_bach_group_n_rpt(bigger_sizes,maxes, mins, 4)
    for i in range(len(data)):
        data[i] += temp_data[i]
    temp_data = not_bach_group_n_rpt(smaller_sizes, maxes, mins, 2)
    for i in range(len(data)):
        data[i] += temp_data[i]
    temp_data = not_bach_group_n_rpt(bigger_sizes, maxes, mins, -1)
    for i in range(len(data)):
        data[i] += temp_data[i]
    #rand groups now implemented with n<=0
    #data += note_bach_group_rand(smaller_sizes,maxes, mins)
    return data

def not_bach_group_n_rpt(sizes, maxes, mins, n):
    initial_n = n #if n<=0 random group sizes (1-4 length) are generated
    temp_data = [[],[],[]]
    data = [[],[],[]]
    for i in range(len(sizes)):
        last = []
        #for _ in range(0,sizes[i]+size, n):
        x = 0
        while x < sizes[i]+size:
            if initial_n <= 0:
                n = rand.randint(0,4)%4
                if n == 0:
                    n = 4
            notes = [-1,-1,-1,-1]
            if last:
                saved_from_last = list(set(rand.choices(last, k=rand.randint(1,7)%4)))
                saved_indexes = []
                for saved in saved_from_last:
                    saved_index = last.index(saved)
                    saved_indexes += [saved_index]
                    notes[saved_index] = saved
                if len(saved_indexes) == 1:
                    if saved_indexes[0] == 3:
                        if notes[3] > mins[2]:
                            notes[2] = notes[3]
                            notes[3] = -1
                            saved_indexes[0] = 2
                    if saved_indexes[0] == 2:
                        if notes[2] > mins[1]:
                            notes[1] = notes[2]
                            notes[2] = -1
                elif len(saved_indexes) == 2:
                    if sorted(saved_indexes) == [0,3]:
                        if notes[3] > mins[2]:
                            notes[2] = notes[3]
                            notes[3] = -1
                    if sorted(saved_indexes) == [2,3]:
                        if notes[2] > mins[1]:
                            notes[1] = notes[2]
                            notes[2] = -1
                            if notes[3] > mins[2]:
                                notes[2] = notes[3]
                                notes[3] = -1
                #print(notes)
                for k in range(4):
                    if notes[k] > -1:
                        continue
                    elif k < 3 and notes[k+1] > -1:
                        if k > 0:
                            notes[k] = rand.randint(max(mins[k],notes[k+1]), min(notes[k-1], maxes[k]))
                        else: # i==0
                            notes[k] = rand.randint(max(mins[k],notes[k+1]), maxes[0])
                    elif k != 0:
                        notes[k] = rand.randint(mins[k], min(notes[k-1], maxes[k]))
                    else: #i==0
                        notes[0] = rand.randint(mins[0], maxes[0])
            else:
                notes[0] = rand.randint(mins[0], maxes[0])
                notes[1] = rand.randint(mins[1], min(notes[0], maxes[1]))
                notes[2] = rand.randint(mins[2], min(notes[1], maxes[2]))
                notes[3] = rand.randint(mins[3], min(notes[2], maxes[3]))
            last = [notes[0], notes[1], notes[2], notes[3]]
            temp_harm = np.array(last)
            temp_data[i] += [(np.divide(temp_harm,127)).tolist()]*n
            x+=n
        for j in range(len(temp_data[i][:-size])):
            harm_of_size = temp_data[i][j:j+size]
            harm_of_size += [[0]*4]
            data[i] += [harm_of_size]
    return data

def not_bach_group_n(sizes, maxes, mins, n):
    initial_n = n #if n<=0 random group sizes (1-4 length) are generated
    temp_data = [[],[],[]]
    data = [[],[],[]]
    for i in range(len(sizes)):
        k = 0
        while k < sizes[i]+size:
            if initial_n <= 0:
                n = rand.randint(0,4)%4
                if n == 0:
                    n = 4
            note1 = int(rand.randint(mins[0], maxes[0]))
            note2 = int(rand.randint(mins[1], min(note1, maxes[1])))
            note3 = int(rand.randint(mins[2], min(note2, maxes[2])))
            note4 = int(rand.randint(mins[3], min(note3, maxes[3])))
            temp_data[i] += [[note1/127, note2/127, note3/127, note4/127]]*n
            k+=n
        for j in range(len(temp_data[i][:-size])):
            harm_of_size = temp_data[i][j:j+size]
            harm_of_size += [[0]*4]
            data[i] += [harm_of_size]
    return data

if __name__ == '__main__':
    prep()
