import gradio as gr
import gr_demo


def create_tabbed_interface():

    demo = gr.TabbedInterface(
        [
            gr_demo.data_enhance(),
            gr_demo.vector_embedding(),
            gr_demo.db_manage(),
        ],
        [
            "数据增强",
            "嵌入向量编码",
            "重排序和两阶段检索",
        ],
    )
    return demo
