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
                    value=dataset_config["dataset_path"],
                )
                dataset_name = gr.Dropdown(
                    choices=dataset_config["dataset_name"],
                    value=dataset_config["dataset_name"][0],
                    label="数据集名称",
                )
                dataset_split = gr.Dropdown(
                    choices=dataset_config["dataset_split"],
                    value=dataset_config["dataset_split"][0],
                    label="数据集划分",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="top_k",
                )
                btn_0 = gr.Button("图像加载")
                btn_1 = gr.Button("文本加载")
                btn_2 = gr.Button("文本增强")
            with gr.Column(scale=100):
                with gr.Row():
                    gallery = gr.Gallery(
                        label="检索结果",
                        columns=5,
                        height=220,
                    )
                with gr.Row():
                    with gr.Column(scale=50):
                        dataset_df_1 = gr.Dataframe(
                            label="原始数据",
                            headers=["id", "字符数量", "原始文本"],
                        )
                    with gr.Column(scale=50):
                        dataset_df_2 = gr.Dataframe(
                            label="增强数据",
                            headers=["id", "字符数量", "增强文本"],
                        )
            inputs = [dataset_path, dataset_name, dataset_split, top_k]
            btn_0.click(
                fn=gr_function.get_dataset_image,
                inputs=inputs,
                outputs=gallery,
            )
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


def vector_embedding():
    def clear_df_2():
        return [["", ""]]

    def clear_df_3():
        return [["", "", ""]]

    embedding_config = load_yaml("config/embedding.yaml")
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                vision_embedding_model = gr.CheckboxGroup(
                    choices=embedding_config["vision_embedding_model"],
                    label="图像嵌入编码模型",
                    value=[embedding_config["vision_embedding_model"][0]],
                )
                text_embedding_model = gr.CheckboxGroup(
                    choices=embedding_config["text_embedding_model"],
                    label="文本嵌入编码模型",
                    value=[embedding_config["text_embedding_model"][0]],
                )
                test_case = gr.Dropdown(
                    choices=["case_1", "case_2", "case_3"],
                    label="测试用例",
                )
                btn_0 = gr.Button("计算图像编码")
                btn_1 = gr.Button("计算文本编码")
                btn_2 = gr.Button("计算融合编码")
            with gr.Column(scale=100):
                with gr.Row():
                    with gr.Column(scale=30):
                        image_input = gr.Image(label="图像输入")
                    with gr.Column(scale=70):
                        image_embedding_df = gr.Dataframe(
                            headers=["嵌入编码模型", "向量表示"],
                        )
                with gr.Row():
                    with gr.Column(scale=30):
                        text_input = gr.Textbox(label="文本输入")
                    with gr.Column(scale=70):
                        text_embedding_df = gr.Dataframe(
                            headers=["嵌入编码模型", "向量表示"],
                        )
                with gr.Row():
                    combined_embedding_df = gr.Dataframe(
                        headers=[
                            "图像嵌入编码模型",
                            "文本嵌入编码模型",
                            "融合向量表示",
                        ],
                    )

        test_case.change(
            fn=gr_function.fill_case_data,
            inputs=[test_case],
            outputs=[image_input, text_input],
        )
        test_case.change(fn=clear_df_2, inputs=None, outputs=[image_embedding_df])
        test_case.change(fn=clear_df_2, inputs=None, outputs=[text_embedding_df])
        test_case.change(fn=clear_df_3, inputs=None, outputs=[combined_embedding_df])
        inputs_0 = [vision_embedding_model, test_case]
        inputs_1 = [text_embedding_model, test_case]
        inputs_2 = [vision_embedding_model, text_embedding_model, test_case]
        btn_0.click(
            fn=gr_function.get_vision_embedding,
            inputs=inputs_0,
            outputs=image_embedding_df,
        )
        btn_1.click(
            fn=gr_function.get_text_embedding,
            inputs=inputs_1,
            outputs=text_embedding_df,
        )
        btn_2.click(
            fn=gr_function.get_combined_embedding,
            inputs=inputs_2,
            outputs=combined_embedding_df,
        )
    return demo


def retrieval():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                dataset_name = gr.Dropdown(
                    choices=["webqa"],
                    value="webqa",
                    label="数据集名称",
                )
                vision_embedding_model = gr.Dropdown(
                    choices=["vit-base-patch16-224"],
                    value="vit-base-patch16-224",
                    label="图像嵌入编码模型",
                )
                text_embedding_model = gr.Dropdown(
                    choices=["bge-base-en-v1.5"],
                    value="bge-base-en-v1.5",
                    label="文本嵌入编码模型",
                )
                top_k1 = gr.Slider(
                    minimum=30,
                    maximum=150,
                    step=5,
                    value=50,
                    label="top_k1",
                )
                top_k2 = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=1,
                    label="top_k2",
                )
                btn_1 = gr.Button("一阶段检索 (粗排)")
                btn_2 = gr.Button("二阶段检索 (精排)")
                btn_3 = gr.Button("计算检索指标")
            with gr.Column(scale=100):
                with gr.Row():
                    with gr.Column(scale=70):
                        with gr.Row():
                            with gr.Column(scale=1):
                                test_case = gr.Dropdown(
                                    choices=["case_1"],
                                    label="测试用例",
                                )
                            with gr.Column(scale=1):
                                query_text = gr.Textbox(label="图像描述")
                        with gr.Row():
                            with gr.Column(scale=100):
                                query_df = gr.Dataframe(
                                    headers=["id", "score", "文本"],
                                    label="检索结果",
                                )
                    with gr.Column(scale=30):
                        query_image = gr.Image(label="图像输入")
                with gr.Row():
                    with gr.Column(scale=50):
                        retrieval_1_df = gr.Dataframe(
                            headers=["id", "score", "文本"],
                            label="一阶段检索结果",
                        )
                    with gr.Column(scale=50):
                        retrieval_2_df = gr.Dataframe(
                            headers=["id", "score", "文本"],
                            label="二阶段检索结果",
                        )
                with gr.Row():
                    with gr.Column(scale=100):
                        metric_df = gr.Dataframe(
                            headers=[
                                "重排序模型",
                                "Recall@100",
                                "MRR@10",
                                "NDCG@10",
                            ],
                            label="检索指标",
                        )
        inputs_1 = [
            dataset_name,
            vision_embedding_model,
            text_embedding_model,
            top_k1,
            test_case,
        ]
        inputs_2 = [
            dataset_name,
            vision_embedding_model,
            text_embedding_model,
            top_k2,
            test_case,
        ]
        inputs_3 = [
            dataset_name,
            vision_embedding_model,
            text_embedding_model,
            test_case,
        ]
        inputs_4 = [
            query_image,
            query_text,
            vision_embedding_model,
            text_embedding_model,
            test_case,
        ]
        btn_1.click(fn=None, inputs=inputs_1, outputs=retrieval_1_df)
        btn_2.click(fn=None, inputs=inputs_2, outputs=retrieval_2_df)
        btn_3.click(fn=None, inputs=inputs_3, outputs=metric_df)
    return demo
