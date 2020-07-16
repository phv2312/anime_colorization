from keras.models import load_model
from .sketch_keras.helper import *

class SketchModel:
    def __init__(self, weight):
        self.mod = load_model('mod.h5')

    def process(self, input_image):
        width = float(input_image.shape[1])
        height = float(input_image.shape[0])
        new_width = 0
        new_height = 0
        if (width > height):
            from_mat = cv2.resize(input_image, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
            new_width = 512
            new_height = int(512 / width * height)
        else:
            from_mat = cv2.resize(input_image, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
            new_width = int(512 / height * width)
            new_height = 512
        # cv2.imshow('raw', from_mat)
        # cv2.imwrite('raw.jpg',from_mat)
        from_mat = from_mat.transpose((2, 0, 1))
        light_map = np.zeros(from_mat.shape, dtype=np.float)
        for channel in range(3):
            light_map[channel] = get_light_map_single(from_mat[channel])
        light_map = normalize_pic(light_map)
        light_map = resize_img_512_3d(light_map)
        line_mat = self.mod.predict(light_map, batch_size=1)
        line_mat = line_mat.transpose((3, 1, 2, 0))[0]
        line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
        # show_active_img_and_save('sketchKeras_colored', line_mat, 'sketchKeras_colored.jpg')
        line_mat = np.amax(line_mat, 2)

        return line_mat