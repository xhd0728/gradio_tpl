import gradio as gr
from gr_interface import create_tabbed_interface
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=65001, type=int)
    parser.add_argument("--queue", default=True, type=bool)
    args = parser.parse_args()

    demo = create_tabbed_interface()

    if args.queue:
        demo.queue().launch(server_name=args.ip, server_port=args.port)
    else:
        demo.launch(server_name=args.ip, server_port=args.port)

    print("exit")
