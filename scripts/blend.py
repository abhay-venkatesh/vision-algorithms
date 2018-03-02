from PIL import Image
import cv2
from utils.DataPostprocessor import DataPostprocessor

dp = DataPostprocessor()
img = cv2.imread("./flowset/pic100-0.png")
seg = cv2.imread("./flowset/seg100-0.png", cv2.IMREAD_GRAYSCALE)
seg = seg/8
dp.write_out(0, img, seg, 0)

background = Image.open("./images/pic63.png")
overlay = Image.open("./images/predicted-real-0-iter0.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
new_img.save("new.png","PNG")