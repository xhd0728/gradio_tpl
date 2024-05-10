import gradio as gr
import gr_demo


def create_tabbed_interface():

    demo = gr.TabbedInterface(
        [
            gr_demo.text2image_gr(),
            gr_demo.text2image_gr(),
            gr_demo.upload2db_gr(),
            gr_demo.show_dataset_gr(),
            gr_demo.extract_caption_gr(),
            gr_demo.text2image_gr(),
            gr_demo.text2image_gr(),
            gr_demo.text2image_gr(),
            gr_demo.text2image_gr(),
            gr_demo.text2image_gr(),
            gr_demo.db_manage(),
        ],
        [
            "文本检索图像",
            "图像检索文本",
            "上传图片",
            "数据集展示",
            "图像文本生成",
            "扩散生成",
            "细节增强",
            "图文特征特征提取",
            "模型训练",
            "模型微调",
            "向量数据存储",
        ],
    )
    return demo
