# MyCode
代码有三部分，分别是ResNet模型、DenseNet模型和test.py(读取已保存的DenseNet model（DenseNet.h5）并生成提交文件)代码，要验证得分的话只需打开第三部分tes.py即可（需将DenseNet.h5与tes.py保存在同一工作路径下）  
读取测试集数据的路径在test.py中指定，可能需要修改（修改path_test即可），本人是在E:\SJTU\sjtu-m3dv-medical-3d-voxel-classification\test路径下存放的测试集  
读取的model和生成的submission文件均为默认路径（也可在test.py中修改），即为python的根目录
生成的Submission文件由于格式问题需手动将表格第一列（序号列）删除，然后将旁边两列向左移动一列即可  
