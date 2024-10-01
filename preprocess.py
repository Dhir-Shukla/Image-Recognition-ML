import csv
from challenge_basic import get_number_list_clean


def preprocess():
    f = open("clean_quercus.csv", encoding="utf8")
    data = csv.reader(f)
    d = [[], []]

    data.__next__()
    skip = 0
    counter = 0
    for row in data:
        counter+=1
        if row[0] == '' or row[6] == '':
            skip+=1
            continue
        else:
            d[0].append([])

            d[0][-1].append(int(row[0][0]))
            d[0][-1].append(int(row[1][0]))
            d[0][-1].append(int(row[2][0]))

            if int(row[0][0]) == 4 and int(row[1][0]) == 3 and int(row[2][0]) == 3 and 'People' in row[3]:
                print(counter)

            if 'People' in row[3]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Cars' in row[3]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Cats' in row[3]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Fireworks' in row[3]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Explosions' in row[3]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Parents' in row[4]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Siblings' in row[4]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Friends' in row[4]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            if 'Teacher' in row[4]:
                d[0][-1].append(1)
            else:
                d[0][-1].append(0)

            d[0][-1].extend(get_number_list_clean(row[5]))

            d[0][-1].append(float(row[6].replace(',', '')))
            d[0][-1].append(float(row[7].replace(',', '')))

            # Append label
            d[1].append(int(row[10]))

            # Save as a NumPy array
            #np.save("data" , np.array(d))


if __name__ == "__main__":
    preprocess()