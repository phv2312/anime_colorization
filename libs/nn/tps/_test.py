from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from libs.nn.tps_model import TPS_SpatialTransformerNetwork


def main():
    model = TPS_SpatialTransformerNetwork(F=40, I_size=(512, 512), I_r_size=(512, 512), I_channel_num=3).cuda()

    input_path  = "/home/kan/Desktop/Cinnamon/gan/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions/geek/full_data/HOR01_Full_Formated/hor01_044_046_k_R_B/color_processed/B0004.png"
    input_image = Image.open(input_path).convert('RGB').resize((512, 512))


    to_tensor = transforms.Compose([
        #transforms.Resize(size=(256,256)),
        transforms.ToTensor()
    ])

    output = model(to_tensor(input_image).unsqueeze(0).cuda())
    print (output)

    output_cpu = output.detach().cpu().squeeze(0).permute(1,2,0).numpy() * 255
    output_cpu = output_cpu.astype(np.uint8)

    Image.fromarray(output_cpu).show()
    input_image.show()

if __name__ == '__main__':
    main()