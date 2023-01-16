import glob
import os


def get_cnt_limit(total_n) :
    for i in range(1,100) :
        if total_n // i > 1500 :
            continue
        else :
            return i - 1


def print_cnt(path):
    for i in range(2, 10):
        print('='*5, i * 10, '='*5)
        all_data_path = glob.glob(os.path.join(path, str(i) + '*', '111', '*'))
        print('man   - ',  len(all_data_path))
        all_data_path = glob.glob(os.path.join(path, str(i) + '*', '112', '*'))
        print('woman - ', len(all_data_path))


def remove20304050() :
    path = "./AFAD_remove/"
    age = ['2*', '3*', '4*', '5*']
    sex = ['111', '112']

    for a in age :
        for s in sex :
            all_img = glob.glob(os.path.join(path, a, s, '*'))
            n = len(all_img)
            if n > 1500 :
                cnt_limit = get_cnt_limit(n)
                cnt = 0
                for img in all_img:
                    cnt += 1
                    temp = img
                    if cnt == cnt_limit:
                        cnt = 0
                    else:
                        os.remove(temp)
    print_cnt(path)

def dataset_remove(path, n) :
    labels = os.listdir(path)

    for label in labels :
        imgs = glob.glob(os.path.join(path, label,'*.png'))

        if len(imgs) > n :
            over = len(imgs) - n
            for img in imgs :
                os.remove(img)
                over -= 1
                if over <= 0 :
                    break
        print(label, len(glob.glob(os.path.join(path, label, '*.png'))))

# remove20304050()
dataset_remove('./dataset/train/', 1512)



