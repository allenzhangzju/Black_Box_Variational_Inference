import os

for i in range(0,10):
    print('bbvi_basic')
    os.system('python bbvi_basic.py ./elbos/bbvi_basic/{:}.npy'.format(i))
    print('bbvi_cv')
    os.system('python bbvi_cv.py ./elbos/bbvi_cv/{:}.npy'.format(i))
    print('abbvi_basic')
    os.system('python abbvi_basic.py ./elbos/abbvi_basic/{:}.npy'.format(i))
    