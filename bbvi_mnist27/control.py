import os


for i in range(10):
    print('bbvi_basic')
    os.system('python3 bbvi_basic.py ./elbos/bbvi_basic/{:}.npy'.format(i))
    print('bbvi_cv')
    os.system('python3 bbvi_cv.py ./elbos/bbvi_cv/{:}.npy'.format(i))
    print('abbvi_basic')
    os.system('python3 abbvi_basic.py ./elbos/abbvi_basic/{:}.npy'.format(i))