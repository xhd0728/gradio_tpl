import gradio as gr


def header_gr():
    title = "<h1 align='center'>ITRS-DL: An Image and Text Retrieval System based on Deep Learning</h1>"
    description = "<h3 align='center'>基于深度学习的图文检索系统</h3>"
    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
    return demo
