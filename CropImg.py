from PIL import Image
import os

def crop_image(input_path, output_path):
    # 加载图片
    image = Image.open(input_path)
    # 转换为具有Alpha通道（透明度）
    image = image.convert("RGBA")
    # 获得图像的边界
    # bboxRaw = image.getbbox() (0, 0, 5120, 3840)
    bbox = (150, 300, 4700, 3840)   # (150, 300, 4700, 3840), (40, 500, 2330, 4700), (30, 20, 810, 656)
    # 裁剪图像
    image = image.crop(bbox)
    # 保存裁剪后的图像
    image.save(output_path)

def batch_crop(input_folder, output_folder):
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历目录中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 检查文件格式
            file_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            crop_image(file_path, output_path)
            print(f"Cropped {file_name}")

# 设置输入和输出目录
input_folder = r'D:\ZJU\嗑盐\大论文\figs备份'
output_folder = r'D:\ZJU\嗑盐\大论文\tmp'

# 执行批量裁剪
batch_crop(input_folder, output_folder)