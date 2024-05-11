import gradio as gr
import gr_demo


def create_tabbed_interface():

    demo = gr.TabbedInterface(
        [
            gr_demo.data_enhance(),
            gr_demo.extract_caption_gr(),
            gr_demo.db_manage(),
        ],
        [
            "数据增强",
            "图文特征特征提取",
            "向量数据存储",
        ],
    )
    return demo
