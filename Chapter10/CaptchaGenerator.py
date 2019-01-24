from claptcha import Claptcha
import os
import numpy as np
import cv2
import fire 
from elapsedtimer import ElapsedTimer

def generate_captcha(outdir,font,num_captchas=20000):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    alphabets = alphabets.upper()
    try:
        os.mkdir(outdir)
    except:
        'Directory already present,writing captchas to the same'
    #rint(char_num_ind)
    # select one alphabet if indicator 1 else number 
    for i in range(num_captchas):
        char_num_ind = list(np.random.randint(0,2,4))
        text = ''
        for ind in char_num_ind:
            if ind == 1:
                loc = np.random.randint(0,26,1)
                text = text + alphabets[np.random.randint(0,26,1)[0]]
            else:
                text = text + str(np.random.randint(0,10,1)[0])
        c = Claptcha(text,font)
        text,image = c.image
        image.save(outdir + text + '.png')

def main_process(outdir_train,num_captchas_train,
                 outdir_val,num_captchas_val,
                 outdir_test,num_captchas_test,
                font):

    generate_captcha(outdir_train,font,num_captchas_train)
    generate_captcha(outdir_val,font,num_captchas_val)
    generate_captcha(outdir_test,font,num_captchas_test)


if __name__ == '__main__':
    with ElapsedTimer('main_process'):
        fire.Fire(main_process)



