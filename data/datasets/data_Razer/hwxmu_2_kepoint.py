import os


def one_object():
    with open(r'G:\Projects\DATASET\dots数据集\红外小目标\data_label_2\data2.txt') as f:
        lines = f.readlines()
        for line in lines:
            a = line.split('\t')
            if a[0][:5] == 'frame':
                x1 = a[3]
                y1 = a[4]
                with open('./labels/{}.txt'.format(a[0][6:]), mode='a+') as w:
                    w.write('1\n')
                    w.write(str(int(x1) / 256) + ' ' + str(int(y1) / 256) + '\n')

def two_object():
    with open(r'G:\Projects\DATASET\dots数据集\红外小目标\data_label_2\data2.txt') as f:
        lines = f.readlines()
        for line in lines:
            a = line.split('\t')
            if a[0][:5] == 'frame':
                x1 = a[3]
                y1 = a[4]
                x2 = a[6]
                y2 = a[7]
                with open('./labels/{}.txt'.format(a[0][6:]), mode='a+') as w:
                    w.write('1\n')
                    w.write(str(int(x1) / 256) + ' ' + str(int(y1) / 256) + '\n')
                    w.write(str(int(x2) / 256) + ' ' + str(int(y2) / 256) + '\n')


if __name__ == '__main__':
    two_object()