# 台风路径深度学习集合预报方案
<font color="#dd0000"> 本仓库正在完善中 </font><br />
## 快速启动
（1）所需python包安装：\
进入项目根目录，`pip install -r requirements.txt`\
（2）数据下载及模型下载：\
链接：https://pan.baidu.com/s/1fIaPOhf5Aid3dYSSlOZnIA?pwd=tp23  在百度网盘分享中下载模型训练和测试所需的文件，其中：\
'.npy'格式文件为数据，数据文件在网盘中的路径即为项目中的存放路径 \
`model`文件夹中存放的是已经训练好的深度学习模型权重文件，其中`model_demo.pth`是深度学习预测方法的展示模型，其余模型可用于10个成员以内的集合预报模型训练，权重文件请置于`./model/`下\
（3）深度学习模型的训练和测试：\
在`demo.ipynb`中对深度学习模型进行训练和测试，若训练，则模型权重会保存至`./model/model_demo.pth`，若测试，则需要权重文件`./model/model_demo.pth`存在 \
在`draw.ipynb`中对深度学习模型2021年至今的台风预测结果进行绘制，绘制的台风是按样本时间长度从大到小排列的 \
在`other_model.ipynb`中查看其他传统深度学习模型的训练和预测效果 \
（4）集合预报模型的训练和测试：\
在`ens_train.ipynb`中训练和测试集合预报模型，模型采用多个深度学习模型的预测结果为数据集，需要下载`./pred/processed/`中的多个数据文件
