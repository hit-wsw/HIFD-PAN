import imageio.v2 as imageio # pip install imageio
import re
# create gif
def create_gif(image_list, gif_name, duration=1.5):
    frames = []
    for img in image_list:
        frames.append(imageio.imread(img))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)


class sortimg():

    def tryint(self, s):
        try:
            return int(s)
        except ValueError:
            return s

    def str2int(self, v_str):
        return [self.tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

    def sort_humanly(self, v_list):
        return sorted(v_list, key=self.str2int)


if __name__ == '__main__':
    from glob import glob
    imgs = glob('gif/*.png')
    # 排序
    sorted_imgs = sortimg().sort_humanly(imgs)

    create_gif(sorted_imgs, 'HIDF.gif', duration=0.1)