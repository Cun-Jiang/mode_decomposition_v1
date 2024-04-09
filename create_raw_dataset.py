import time
from generateFunc import init, random_data_img_generate


pro_r, pro, E_mode, mode_count = init()

epoch_num = 350000
# epoch_num = 3
for epoch in range(0, epoch_num + 1):
    begin = time.time()
    random_data_img_generate(pro, E_mode, mode_count, epoch)
    end = time.time()
    if epoch % 500 == 0:
        print(f'epoch: {epoch}, run time: {end - begin} s')
