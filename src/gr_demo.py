import gradio as gr
import gr_function
from utils import load_yaml


def data_enhance():
    dataset_config = load_yaml("config/dataset.yaml")
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                dataset_path = gr.Dropdown(
                    choices=[dataset_config["dataset_path"]],
                    label="数据集路径",
                )
                dataset_name = gr.Dropdown(
                    choices=[item["name"] for item in dataset_config["dataset_name"]],
                    value=[item["file"] for item in dataset_config["dataset_name"]],
                    label="数据集名称",
                    allow_custom_value=True,
                )
                dataset_split = gr.Dropdown(
                    choices=dataset_config["dataset_split"],
                    value=[0, 1, 2],
                    label="数据集划分",
                    allow_custom_value=True,
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="展示数量",
                )
                btn_1 = gr.Button("数据加载")
                btn_2 = gr.Button("数据增强")
            with gr.Column(scale=50):
                dataset_df_1 = gr.Dataframe(
                    label="原始数据",
                    headers=["图像", "文本"],
                )
            with gr.Column(scale=50):
                dataset_df_2 = gr.Dataframe(
                    label="增强数据",
                    headers=["图像", "文本"],
                )
            inputs = [dataset_path, dataset_name, dataset_split, top_k]
            btn_1.click(
                fn=gr_function.get_origin_dataset,
                inputs=inputs,
                outputs=dataset_df_1,
            )
            btn_2.click(
                fn=gr_function.get_enhance_dataset,
                inputs=inputs,
                outputs=dataset_df_2,
            )
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
        btn.click(fn=gr_function.gen_df, inputs=inputs, outputs=output_df)

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
