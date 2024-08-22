import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['FangSong']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.lines as mlines
import itertools
from matplotlib.pyplot import MultipleLocator
import openpyxl
from matplotlib.font_manager import FontProperties


file_path = r'D:\ZJU\嗑盐\代码备份\RadarRGBFusionNet2_20240402\TrackResults\20240331Draw.xlsx'  # 20231114UsefulData, JustTest, 20231121, delete, 20231113AllData
DataPath = openpyxl.load_workbook(file_path)
AllMOTA = []
sheet1 = DataPath.active
cells = sheet1['B:F']
datasets_path = []
for cell_columns in cells:
    cnt = 0
    for cell_rows in cell_columns:
        if cnt >= 1 and cnt <= 20:
            AllMOTA.append(cell_rows.value)
        cnt += 1
        # print(cell_rows.value)
# groups_length = len(datasets_path)
MOTA_CAM, MOTA_Peo, MOTA_app, MOTA_Multi, MOTA_pro = AllMOTA[0:20], AllMOTA[20:40], AllMOTA[40:60], AllMOTA[60:80], AllMOTA[80:100]

print('finally')
size_sequence = 20
x_num = [i for i in range(1, 21)]
plt.plot(x_num, MOTA_pro, 'ro-',markerfacecolor='white')
# plt.plot(x_num, MOTA_Multi, 'gv-',markerfacecolor='white')
# plt.plot(x_num, MOTA_app, 'b^-',markerfacecolor='white')
plt.plot(x_num, MOTA_CAM, 'cs-',markerfacecolor='white')
plt.plot(x_num, MOTA_Peo, 'b<-',markerfacecolor='white')
plt.ylim([0.0, 1.1])

plt.ylabel('多目标跟踪准确率', fontsize=15)
plt.xlabel('视频序列', fontsize=15)
plt.tick_params(labelsize=15)
#
#
# # plt.plot(x_num, motp1, 'ro-',markerfacecolor='white')
# # plt.plot(x_num, motp2, 'gv-',markerfacecolor='white')
# # plt.plot(x_num, motp3, 'b^-',markerfacecolor='white')
# # plt.plot(x_num, motp4, 'cs-',markerfacecolor='white')
# # plt.plot(x_num, motp5, 'y<-',markerfacecolor='white')
# # plt.ylim([0, 1])
# # plt.ylabel('Multiple object tracking precision (MOTP)')
# # plt.xlabel("Sequence")
#
# save_mot = {'motp1':motp1,'motp2':motp2,'motp3':motp3,'motp4':motp4,'motp5':motp5,'mota1':mota1,'mota2':mota2,'mota3':mota3,'mota4':mota4,'mota5':mota5}
# np.save('mot.npy', save_mot)
#
pd_line1 = mlines.Line2D([],[],  0.8, '-', 'c', marker='s', label='单相机跟踪算法', markersize=5, markerfacecolor='white')
pd_line2 = mlines.Line2D([], [], 0.8, '-', 'b',  marker='<',label='传统融合跟踪算法', markersize=5,
                        markerfacecolor='white')
# pd_line3 = mlines.Line2D([], [], 1, '-', 'b',   marker='^',label='APP and Ori', markersize=5,
#                         markerfacecolor='white')
# pd_line4 = mlines.Line2D([], [], 1, '-', 'g', marker='v', label='Multi Asso', markersize=5,
#                         markerfacecolor='white')
pd_line5 = mlines.Line2D([],[],  1, '-','r', marker='o', label='提出的融合跟踪算法', markersize=5, markerfacecolor='white')

plt.legend(loc='upper left', handles=[pd_line1, pd_line2, pd_line5], fontsize=10)
plt.tick_params(axis='both',which='major')
ax=plt.gca()

x_major_locator = MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig('./mota.png', bbox_inches='tight', dpi=800)


plt.show()