import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import color
import numpy as np

noise_imgs = []
line_imgs = []
img_num = 1000
try:
    for n in xrange(0, img_num):
        noise = io.imread('noise_img_' + str(n) + '.bmp')
        line = io.imread('line_img_' + str(n) +'.bmp')

        io.imsave('../line_error_test_' + str(n) + '.bmp', noise - line)
        gray_line = color.rgb2gray(line)
        rows, cols = gray_line.shape
        matrix = np.zeros((rows, cols), dtype=np.uint8)
        # if n > 0:
        #     print n
        # plt.figure(100 + n)
        # io.imshow(gray_line, cmap='gray')
        line_coordinates = []
        file = open('../line_pos_' + str(n) + '.txt', 'w')
        for i in xrange(0, cols):
            threshold = 0.0
            grays = []
            for j in xrange(0, rows):
                if gray_line[j][i] > threshold:
                    grays += [j]
            max_y = grays[len(grays) / 2]
            line_coordinates += [max_y]
            matrix[max_y][i] = 255
            noise[max_y][i][0] = 255

            file.write(str(max_y) + '\n')
#     plt.figure(n)
#     io.imshow(noise)
#     plt.figure(100 + n)
#     io.imshow(matrix, cmap='gray')
# plt.show()
except:
    print 'loaded', n, 'images'
