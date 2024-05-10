import gradio as gr
from gr_header import header_gr
from gr_function import upload_func, gen_df, get_test_gallery


def text2image_gr():

    with gr.Blocks() as demo:
        # header_gr()
        with gr.Row():
            with gr.Column(scale=1):
                query_text = gr.Textbox(
                    value="a yellow dog", label="检索文本", elem_id=0, interactive=True
                )

                model_name = gr.Dropdown(
                    label="模型选择",
                    choices=[
                        "clip-vit-base-patch16",
                        "clip-vit-base-patch32",
                        "clip-vit-large-patch14",
                        "chinese-clip-vit-base-patch16",
                        "chinese-clip-vit-large-patch14-336px",
                    ],
                    elem_id=3,
                )

                topk = gr.Slider(
                    minimum=1, maximum=20, step=1, value=10, label="top_k", elem_id=2
                )

                btn1 = gr.Button("检索")

            with gr.Column(scale=100):
                out1 = gr.Gallery(label="检索结果", columns=5)

        inputs = [query_text, topk, model_name]

        examples = [
            ["a yellow dog", 10, "clip-vit-base-patch32"],
            ["many apples", 10, "clip-vit-large-patch14"],
        ]

        gr.Examples(examples, inputs=inputs)

        btn1.click(fn=get_test_gallery, inputs=inputs, outputs=out1)

    return demo


def upload2db_gr():
    with gr.Blocks() as demo:
        # header_gr()
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil")
                label_input = gr.Textbox(label="文本描述")

                gr.ClearButton(img_input)
                btn = gr.Button("上传图文信息")

            with gr.Column(scale=100):
                img_show = gr.Gallery(
                    label="上传图片预览",
                    columns=5,
                    height=256,
                )
                doc_df = gr.Dataframe(
                    label="关联文档",
                    headers=["id", "length", "context"],
                )

        btn.click(
            fn=upload_func,
            inputs=[img_input, label_input],
            outputs=[img_show, doc_df],
        )
    return demo


def show_dataset_gr():
    with gr.Blocks() as demo:
        # header_gr()
        with gr.Row():
            with gr.Column(scale=1):
                dataset_path = gr.Dropdown(
                    choices=["config/dataset"], label="数据集配置文件路径"
                )

                dataset_name = gr.Dropdown(
                    choices=[
                        "coco2014.json",
                        "coco2017.json",
                        "flickr30k.json",
                        "mini-imagenet.json",
                        "coco-cn.json",
                        "flickr30k-cn.json",
                    ],
                    label="数据集配置文件名称",
                )

                is_train = gr.Dropdown(
                    choices=["train", "test", "val"],
                    label="数据集划分",
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="展示数量",
                )

                btn = gr.Button("加载")

            with gr.Column(scale=100):
                img_out = gr.Gallery(label="数据集预览:", columns=5)

            inputs = [dataset_path, dataset_name, is_train, top_k]
            btn.click(fn=None, inputs=inputs, outputs=img_out)

    return demo


def extract_caption_gr():
    with gr.Blocks() as demo:
        # header_gr()
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil")

                model_name = gr.Dropdown(
                    label="模型选择",
                    choices=[
                        "llava-v1.6-34b",
                        "llava-v1.6-mistral-7b",
                        "llava-v1.6-vicuna-7b",
                        "llava-v1.6-vicuna-13b",
                    ],
                    elem_id=3,
                )

                max_token = gr.Slider(
                    minimum=128,
                    maximum=512,
                    step=10,
                    value=256,
                    label="最大生成长度",
                    elem_id=2,
                )

                list_num = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="生成数量",
                )

                gr.ClearButton(img)
                btn = gr.Button("上传图片")

            with gr.Column(scale=100):
                output_df = gr.Dataframe(
                    label="生成结果",
                    headers=["模型名称", "生成结果"],
                )

        inputs = [model_name, img, max_token, list_num]
        btn.click(fn=gen_df, inputs=inputs, outputs=output_df)

    return demo


def db_manage():
    with gr.Blocks() as demo:
        # header_gr()
        with gr.Row():
            with gr.Column(scale=1):

                db_addr = gr.Dropdown(
                    choices=[
                        "127.0.0.1",
                        "8.217.103.200",
                        "20.37.112.170",
                    ],
                    label="Milvus地址",
                )

                db_port = gr.Dropdown(
                    choices=[
                        "19530",
                    ],
                    label="Milvus端口",
                )

                db_port = gr.Dropdown(
                    choices=[
                        "default",
                    ],
                    label="数据库",
                )

                redis_addr = gr.Dropdown(
                    choices=[
                        "127.0.0.1",
                        "8.217.103.200",
                        "20.37.112.170",
                    ],
                    label="Redis地址",
                )

                redis_port = gr.Dropdown(
                    choices=[
                        "19530",
                    ],
                    label="Redis端口",
                )

                use_redis = gr.Dropdown(
                    choices=["True", "False"],
                    label="启用Redis缓存",
                )

                btn = gr.Button("刷新")

            with gr.Column(scale=100):
                collection_df = gr.Dataframe(
                    label="Collection列表",
                    headers=["名称", "状态", "大约的Entity数量", "描述", "创建时间"],
                )

                with gr.Row():
                    collection = gr.Dropdown(
                        choices=[
                            "vit_b_p32_coco2014_train",
                            "vit_b_p32_coco2017_val",
                            "vit_l_p14_flickr30k",
                            "vit_l_p14_mini_imagenet_train",
                        ],
                        label="Collection选择",
                    )

                    btn = gr.Button("加载")

                data_df = gr.Dataframe(
                    label="数据",
                    headers=["id", "embedding", "catagory"],
                )

            inputs = [db_addr, db_port, redis_addr, redis_port, use_redis]
            btn.click(fn=None, inputs=inputs, outputs=collection_df)

    return demo
